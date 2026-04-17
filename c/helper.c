#include "blst.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Helper: Standard -> Montgomery
static void to_montgomery(blst_fr *ret, const blst_scalar *val) {
  blst_fr_from_scalar(ret, val);
}

// Helper: Montgomery -> Standard
// static void from_montgomery(blst_scalar *ret, const blst_fr *val) {
//   blst_scalar_from_fr(ret, val);
// }

static size_t reverse_bits(size_t x, size_t log_n) {
  size_t res = 0;
  for (size_t i = 0; i < log_n; i++) {
    res = (res << 1) | (x & 1);
    x >>= 1;
  }
  return res;
}

/* NTT using blst native field arithmetic */
static void ntt_core_low_mem(blst_fr *a, size_t log_n,
                             const blst_fr *principal_root) {
  size_t N = (size_t)1 << log_n;

  // 1. Bit-Reversal Permutation
  for (size_t i = 0; i < N; i++) {
    size_t rev = reverse_bits(i, log_n);
    if (i < rev) {
      blst_fr tmp = a[i];
      a[i] = a[rev];
      a[rev] = tmp;
    }
  }

  // Get Montgomery '1'
  blst_scalar one_scalar = {0};
  one_scalar.b[0] = 1;
  blst_fr mont_one;
  blst_fr_from_scalar(&mont_one, &one_scalar);

  // 2. Cooley-Tukey Butterfly
  for (size_t len = 2, layer = 1; len <= N; len <<= 1, layer++) {
    size_t half = len >> 1;

    blst_fr w_layer = *principal_root;
    size_t squares = log_n - layer;
    for (size_t k = 0; k < squares; k++) {
      blst_fr_sqr(&w_layer, &w_layer);
    }

    blst_fr w = mont_one;

    for (size_t j = 0; j < half; j++) {
      for (size_t i = 0; i < N; i += len) {
        blst_fr u = a[i + j];
        blst_fr v;

        // v = a[i + j + half] * w
        blst_fr_mul(&v, &a[i + j + half], &w);

        // a[i + j] = u + v
        blst_fr_add(&a[i + j], &u, &v);

        // a[i + j + half] = u - v
        blst_fr_sub(&a[i + j + half], &u, &v);
      }
      // Update w for next j
      blst_fr_mul(&w, &w, &w_layer);
    }
  }
}

void precompute_rho_super(blst_fr *rho_super, const blst_scalar *rho_scalar) {
  blst_fr rho_mont;
  blst_fr_from_scalar(&rho_mont, rho_scalar);

  blst_scalar tmp;
  blst_scalar_from_fr(&tmp, &rho_mont);
  blst_fr_from_scalar(rho_super, &tmp);
}

void blst_fr_inner_product_fast(blst_scalar *out, const blst_scalar *x,
                                const blst_fr *rho_super, size_t n) {
  blst_fr acc;
  memset(&acc, 0, sizeof(blst_fr));

  for (size_t i = 0; i < n; i++) {
    blst_fr tmp;

    blst_fr_mul(&tmp, (const blst_fr *)&x[i], &rho_super[i]);
    blst_fr_add(&acc, &acc, &tmp);
  }

  // Convert back from Montgomery (acc) to standard bytes (out)
  blst_scalar_from_fr(out, &acc);
}

void compute_lpn_toeplitz_ntt_c(blst_scalar *z, const blst_scalar *err,
                                const blst_scalar *s, const blst_scalar *x,
                                const blst_scalar *toeplitz_vec,
                                const blst_scalar *fwd_root,
                                const blst_scalar *inv_root,
                                const blst_scalar *inv_N, size_t n,
                                size_t kappa, size_t log_n) {
  size_t N = (size_t)1 << log_n;

  blst_fr *S_pad = calloc(N, sizeof(blst_fr));
  blst_fr *C = calloc(N, sizeof(blst_fr));

  for (size_t j = 0; j < kappa; j++)
    to_montgomery(&S_pad[j], &s[j]);

  to_montgomery(&C[0], &toeplitz_vec[0]);
  for (size_t k = 1; k < kappa; k++)
    to_montgomery(&C[N - k], &toeplitz_vec[k]);
  for (size_t k = 1; k < n; k++)
    to_montgomery(&C[k], &toeplitz_vec[kappa - 1 + k]);

  blst_fr f_root, i_root, i_n;
  blst_fr_from_scalar(&f_root, fwd_root);
  blst_fr_from_scalar(&i_root, inv_root);
  blst_fr_from_scalar(&i_n, inv_N);

  ntt_core_low_mem(S_pad, log_n, &f_root);
  ntt_core_low_mem(C, log_n, &f_root);

  for (size_t i = 0; i < N; i++)
    blst_fr_mul(&S_pad[i], &S_pad[i], &C[i]);

  ntt_core_low_mem(S_pad, log_n, &i_root);

  for (size_t i = 0; i < n; i++) {
    blst_fr val, err_fr, x_fr;

    blst_fr_mul(&val, &S_pad[i], &i_n);

    // Convert to standard scalars for final addition if your logic requires
    // Or keep in Fr for speed until the very end.
    blst_fr_from_scalar(&err_fr, &err[i]);
    blst_fr_from_scalar(&x_fr, &x[i]);

    blst_fr_add(&val, &val, &err_fr);
    blst_fr_add(&val, &val, &x_fr);

    blst_scalar_from_fr(&z[i], &val);
  }

  free(S_pad);
  free(C);
}

size_t sample_errors_and_affines_c(blst_scalar *err_scalars_out,
                                   blst_scalar *dense_err_scalars_out,
                                   blst_p2_affine *dense_err_affines_out,
                                   const blst_p2_affine *bases_in, size_t n,
                                   double noise_rate, unsigned int seed) {
  srand(seed);
  size_t t_count = 0;

  // Zero out the sparse scalar array using the struct size
  memset(err_scalars_out, 0, n * sizeof(blst_scalar));

  for (size_t i = 0; i < n; i++) {
    double coin = (double)rand() / (double)RAND_MAX;
    if (coin < noise_rate) {
      blst_scalar s;

      // Sample random bytes for the scalar
      for (size_t j = 0; j < 31; j++) {
        s.b[j] = (uint8_t)rand();
      }
      // Mask top byte to ensure s < BLS12-381 order 'r'
      s.b[31] = (uint8_t)rand() & 0x3F;

      // Direct struct assignment is highly optimized by modern compilers
      err_scalars_out[i] = s;             // Sparse output
      dense_err_scalars_out[t_count] = s; // Dense output
      dense_err_affines_out[t_count] = bases_in[i];

      t_count++;
    }
  }
  return t_count;
}
