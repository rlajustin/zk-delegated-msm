#include "blst.h"
#include "consts.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t vec256[4];

static size_t reverse_bits(size_t x, size_t log_n) {
  size_t res = 0;
  for (size_t i = 0; i < log_n; i++) {
    res = (res << 1) | (x & 1);
    x >>= 1;
  }
  return res;
}

static void ntt_core_low_mem(vec256 *a, size_t log_n, const vec256 root) {
  size_t N = (size_t)1 << log_n;

  // 1. Bit-Reversal Permutation
  for (size_t i = 0; i < N; i++) {
    size_t rev = reverse_bits(i, log_n);
    if (i < rev) {
      vec256 tmp;
      memcpy(tmp, a[i], sizeof(vec256));
      memcpy(a[i], a[rev], sizeof(vec256));
      memcpy(a[rev], tmp, sizeof(vec256));
    }
  }

  // 2. Cooley-Tukey Butterfly
  for (size_t len = 2; len <= N; len <<= 1) {
    size_t half = len >> 1;
    size_t step = N / len;

    for (size_t i = 0; i < N; i += len) {
      for (size_t j = 0; j < half; j++) {
        vec256 u, v;
        memcpy(u, a[i + j], sizeof(vec256));

        // v = a[i + j + half] * roots[j * step]
        mul_mont_sparse_256(v, a[i + j + half], roots[j * step], BLS12_381_r,
                            r0);

        // a[i + j] = u + v
        add_mod_256(a[i + j], u, v, BLS12_381_r);

        // a[i + j + half] = u - v
        sub_mod_256(a[i + j + half], u, v, BLS12_381_r);
      }
    }
  }
}

static void ntt_core(vec256 *a, size_t log_n, const vec256 *roots) {
  size_t N = (size_t)1 << log_n;

  // 1. Bit-Reversal Permutation
  for (size_t i = 0; i < N; i++) {
    size_t rev = reverse_bits(i, log_n);
    if (i < rev) {
      vec256 tmp;
      memcpy(tmp, a[i], sizeof(vec256));
      memcpy(a[i], a[rev], sizeof(vec256));
      memcpy(a[rev], tmp, sizeof(vec256));
    }
  }

  // 2. Cooley-Tukey Butterfly
  for (size_t len = 2; len <= N; len <<= 1) {
    size_t half = len >> 1;
    size_t step = N / len;

    for (size_t i = 0; i < N; i += len) {
      for (size_t j = 0; j < half; j++) {
        vec256 u, v;
        memcpy(u, a[i + j], sizeof(vec256));

        // v = a[i + j + half] * roots[j * step]
        mul_mont_sparse_256(v, a[i + j + half], roots[j * step], BLS12_381_r,
                            r0);

        // a[i + j] = u + v
        add_mod_256(a[i + j], u, v, BLS12_381_r);

        // a[i + j + half] = u - v
        sub_mod_256(a[i + j + half], u, v, BLS12_381_r);
      }
    }
  }
}

void compute_lpn_toeplitz_ntt_c(vec256 *z, const vec256 *err, const vec256 *x,
                                const vec256 *toeplitz_vec,
                                const vec256 *fwd_roots,
                                const vec256 *inv_roots, const vec256 inv_N,
                                size_t n, size_t kappa, size_t log_n) {
  size_t N = (size_t)1 << log_n;

  // Allocate padded arrays (initialized to 0 via calloc)
  vec256 *X_pad = calloc(N, sizeof(vec256));
  vec256 *C = calloc(N, sizeof(vec256));

  // 1. Embed 'x' into X_pad
  for (size_t j = 0; j < kappa; j++) {
    memcpy(X_pad[j], x[j], sizeof(vec256));
  }

  // 2. Embed toeplitz_vec into the Circulant vector 'C'
  // This perfectly replicates your original inner loop branching logic in an
  // O(N log N) space.
  memcpy(C[0], toeplitz_vec[0], sizeof(vec256));

  for (size_t k = 1; k < kappa; k++) {
    memcpy(C[N - k], toeplitz_vec[k], sizeof(vec256));
  }
  for (size_t k = 1; k < n; k++) {
    memcpy(C[k], toeplitz_vec[kappa - k], sizeof(vec256));
  }

  // 3. Forward NTT
  ntt_core(X_pad, log_n, fwd_roots);
  ntt_core(C, log_n, fwd_roots);

  // 4. Pointwise Multiplication
  for (size_t i = 0; i < N; i++) {
    mul_mont_sparse_256(X_pad[i], X_pad[i], C[i], BLS12_381_r, r0);
  }

  // 5. Inverse NTT
  ntt_core(X_pad, log_n, inv_roots);

  // 6. Scale by 1/N and extract first 'n' elements, adding to 'err'
  for (size_t i = 0; i < n; i++) {
    vec256 scaled_val;
    mul_mont_sparse_256(scaled_val, X_pad[i], inv_N, BLS12_381_r, r0);
    add_mod_256(z[i], scaled_val, err[i], BLS12_381_r);
  }

  // Cleanup
  free(X_pad);
  free(C);
}

size_t sample_errors_and_affines_c(
    uint8_t *err_bytes_out,         // Full array: [n * 32] bytes
    uint8_t *dense_err_scalars_out, // Dense array: [max_t * 32] bytes
    void *dense_err_affines_out,    // Dense array: blst_p2_affine[max_t]
    const void *bases_in,           // Full array: blst_p2_affine[n]
    size_t n, double noise_rate, unsigned int seed) {
  // Seed the standard C RNG
  srand(seed);

  size_t t_count = 0;

  // Zero-initialize the full err_bytes array natively
  memset(err_bytes_out, 0, n * 32);

  blst_p2_affine *out_affines = (blst_p2_affine *)dense_err_affines_out;
  const blst_p2_affine *in_bases = (const blst_p2_affine *)bases_in;

  for (size_t i = 0; i < n; i++) {
    // Standard C coin flip
    double coin = (double)rand() / (double)RAND_MAX;

    if (coin < noise_rate) {
      uint8_t scalar[32];
      // Fill 32 bytes using rand()
      for (int j = 0; j < 32; j++) {
        scalar[j] = rand() & 0xFF;
      }

      // Copy into the full array for the LPN field hook
      memcpy(&err_bytes_out[i * 32], scalar, 32);

      // Copy into the dense arrays for the Pippenger MSM
      memcpy(&dense_err_scalars_out[t_count * 32], scalar, 32);
      out_affines[t_count] = in_bases[i];

      t_count++;
    }
  }

  return t_count;
}

void compute_lpn_toeplitz(vec256 *z, const vec256 *err, const vec256 *x,
                          const vec256 *toeplitz_vec, size_t n, size_t kappa) {
  size_t i, j;
  for (i = 0; i < n; i++) {
    vec_zero(z[i], sizeof(vec256));
  }

  for (j = 0; j < kappa; j++) {
    for (i = 0; i < n; i++) {
      vec256 tmp;
      size_t idx = (j >= i) ? (j - i) : (kappa + j - i);
      mul_mont_sparse_256(tmp, toeplitz_vec[idx], x[j], BLS12_381_r, r0);
      add_mod_256(z[i], z[i], tmp, BLS12_381_r);
    }
  }

  for (i = 0; i < n; i++) {
    add_mod_256(z[i], z[i], err[i], BLS12_381_r);
    add_mod_256(z[i], z[i], x[i], BLS12_381_r);
  }
}

void compute_lpn_mask_native(vec256 *z, const vec256 *err, const vec256 *x,
                             const vec256 *matrix_A, const vec256 *s, size_t n,
                             size_t kappa) {
  size_t i, j;
  vec256 tmp;

  for (i = 0; i < n; i++) {
    vec_zero(z[i], sizeof(vec256));
  }

  for (j = 0; j < kappa; j++) {
    for (i = 0; i < n; i++) {
      mul_mont_sparse_256(tmp, matrix_A[j * n + i], s[j], BLS12_381_r, r0);
      add_mod_256(z[i], z[i], tmp, BLS12_381_r);
    }
  }

  for (i = 0; i < n; i++) {
    add_mod_256(z[i], z[i], err[i], BLS12_381_r);
    add_mod_256(z[i], z[i], x[i], BLS12_381_r);
  }
}

void precompute_rho_super(vec256 rho_super,
                          const unsigned long long *rho_bytes) {
  vec256 tmp;
  mul_mont_sparse_256(tmp, rho_bytes, BLS12_381_rRR, BLS12_381_r, r0);
  mul_mont_sparse_256(rho_super, tmp, BLS12_381_rRR, BLS12_381_r, r0);
}

void compute_lpn_mask(vec256 *z, const vec256 *err, const vec256 *x,
                      const vec256 *matrix_A, const vec256 *s, size_t n,
                      size_t kappa, matrix_type_t mat_type) {
  if (mat_type == MATRIX_TOEPLITZ) {
    compute_lpn_toeplitz(z, err, x, matrix_A, n, kappa);
  } else {
    compute_lpn_mask_native(z, err, x, matrix_A, s, n, kappa);
  }
}

/* * The optimized inner product.
 * 'x' is your raw array of standard scalars.
 * 'rho_super' is your precomputed array of vec256 limbs.
 */
void blst_fr_inner_product_fast(void *out, const unsigned long long *x,
                                const vec256 *rho_super, size_t n) {
  size_t i;
  vec256 acc, tmp;

  /* Initialize accumulator to zero */
  vec_zero(acc, sizeof(acc));

  for (i = 0; i < n; i++) {
    mul_mont_sparse_256(tmp, x + i * 32, rho_super[i], BLS12_381_r, r0);
    /* acc = acc + tmp (mod r) */
    add_mod_256(acc, acc, tmp, BLS12_381_r);
  }

  /* * Because acc is already in Montgomery form (Value * R),
   * blst_scalar_from_fr will perfectly remove the R and write the final bytes!
   */
  blst_scalar_from_fr(out, (blst_fr *)acc);
}

void toeplitz_matrix_vector_mul(vec256 *out, const vec256 *toeplitz_vec,
                                const vec256 *s, size_t n) {
  size_t i, j;
  for (i = 0; i < n; i++) {
    vec_zero(out[i], sizeof(vec256));
    for (j = 0; j < n; j++) {
      vec256 tmp;
      if (j >= i) {
        mul_mont_sparse_256(tmp, toeplitz_vec[j - i], s[j], BLS12_381_r, r0);
      } else {
        mul_mont_sparse_256(tmp, toeplitz_vec[n + j - i], s[j], BLS12_381_r,
                            r0);
      }
      add_mod_256(out[i], out[i], tmp, BLS12_381_r);
    }
  }
}
