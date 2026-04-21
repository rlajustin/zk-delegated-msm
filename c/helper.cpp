#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/mat_ZZ_p.h>

#include "blst.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

using namespace NTL;

// -----------------------------------------------------------------------------
// NTL Configuration & Utilities
// -----------------------------------------------------------------------------

static bool ntl_inited = false;

static const uint8_t BLS12_381_R[32] = {
    0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0x02, 0xa4, 0xbd, 0x53, 0x05, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
    0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
};

static void init_ntl_bls12_381() {
  if (ntl_inited)
    return;

  ZZ p;
  ZZFromBytes(p, BLS12_381_R, 32);
  ZZ_p::init(p);

  ntl_inited = true;
}

static void blst_scalar_to_zz_p(ZZ_p &out, const blst_scalar *s) {
  byte buf[32];
  blst_lendian_from_scalar(buf, s); // scalar → LE bytes
  ZZ tmp;
  ZZFromBytes(tmp, buf, 32); // LE bytes → ZZ
  out = to_ZZ_p(tmp);
}

static void zz_p_to_blst_scalar(blst_scalar *out, const ZZ_p &v) {
  byte buf[32];
  ZZ tmp = rep(v);
  BytesFromZZ(buf, tmp, 32);          // ZZ → LE bytes
  blst_scalar_from_lendian(out, buf); // LE bytes → scalar
}

extern "C" {

// Trapdoor matrix generation (unchanged logic)
void generate_trapdoor_matrix_c(blst_scalar *matrix_out, size_t n, size_t kappa,
                                unsigned int seed) {
  srand(seed);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < kappa; j++) {
      blst_scalar s;
      for (size_t k = 0; k < 31; k++) {
        s.b[k] = (uint8_t)rand();
      }
      s.b[31] = (uint8_t)(rand() & 0x3F);
      matrix_out[i * kappa + j] = s;
    }
  }
}

// Compute standard trapdoor product
void compute_trapdoor_product_c(blst_scalar *output,
                                const blst_scalar *matrix_in,
                                const blst_scalar *x, size_t n, size_t kappa) {
  init_ntl_bls12_381();

  vec_ZZ_p ntl_x;
  ntl_x.SetLength(n);
  for (size_t i = 0; i < n; i++) {
    blst_scalar_to_zz_p(ntl_x[i], &x[i]);
  }

  // Iterate over columns of the matrix (kappa columns)
  for (size_t j = 0; j < kappa; j++) {
    ZZ_p acc = to_ZZ_p(0);
    for (size_t i = 0; i < n; i++) {
      ZZ_p mij;
      blst_scalar_to_zz_p(mij, &matrix_in[i * kappa + j]);
      acc += ntl_x[i] * mij;
    }
    zz_p_to_blst_scalar(&output[j], acc);
  }
}

// Fast Toeplitz Matrix-Vector Multiplication using NTL's FFT-based Polynomial
// Math
void compute_toeplitz_ntl_c(blst_scalar *output,
                            const blst_scalar *toeplitz_vec,
                            const blst_scalar *x, size_t n, size_t kappa) {
  init_ntl_bls12_381();

  ZZ_pX poly_toeplitz, poly_x, result;

  // Build the polynomial for the Toeplitz vector
  for (size_t i = 0; i < n + kappa - 1; i++) {
    ZZ_p coeff;
    blst_scalar_to_zz_p(coeff, &toeplitz_vec[i]);
    SetCoeff(poly_toeplitz, i, coeff);
  }

  // Build the polynomial for the input vector 'x'.
  // To evaluate a Toeplitz matrix using convolution, we reverse the input
  // vector.
  for (size_t i = 0; i < kappa; i++) {
    ZZ_p coeff;
    blst_scalar_to_zz_p(coeff, &x[kappa - 1 - i]);
    SetCoeff(poly_x, i, coeff);
  }

  // NTL automatically dispatches to FFT (NTT) for large degrees
  mul(result, poly_toeplitz, poly_x);

  // Extract the proper window of coefficients for the result
  for (size_t i = 0; i < n; i++) {
    zz_p_to_blst_scalar(&output[i], coeff(result, i + kappa - 1));
  }
}

// LPN Sampling completely offloaded to NTL
void compute_lpn_sampling_ntl_c(blst_scalar *blinded_x, const blst_scalar *err,
                                const blst_scalar *s, const blst_scalar *x,
                                const blst_scalar *toeplitz_vec, size_t n,
                                size_t kappa) {
  init_ntl_bls12_381();

  ZZ_pX poly_toeplitz, poly_s, result;

  // 1. Setup Toeplitz polynomial
  for (size_t i = 0; i < n + kappa - 1; i++) {
    ZZ_p coeff;
    blst_scalar_to_zz_p(coeff, &toeplitz_vec[i]);
    SetCoeff(poly_toeplitz, i, coeff);
  }

  // 2. Setup Secret 's' polynomial (reversed)
  for (size_t i = 0; i < kappa; i++) {
    ZZ_p coeff;
    blst_scalar_to_zz_p(coeff, &s[kappa - 1 - i]);
    SetCoeff(poly_s, i, coeff);
  }

  // 3. Fast polynomial multiplication (NTT happens natively here inside NTL)
  mul(result, poly_toeplitz, poly_s);

  // 4. Extract window, add error, add x
  for (size_t i = 0; i < n; i++) {
    ZZ_p val = coeff(result, i + kappa - 1);

    ZZ_p err_zz_p, x_zz_p;
    blst_scalar_to_zz_p(err_zz_p, &err[i]);
    blst_scalar_to_zz_p(x_zz_p, &x[i]);

    val += err_zz_p + x_zz_p;

    zz_p_to_blst_scalar(&blinded_x[i], val);
  }
}

// -----------------------------------------------------------------------------
// Legacy & Error Sampling Utilities
// -----------------------------------------------------------------------------

size_t sample_errors_and_affines_c(blst_scalar *err_scalars_out,
                                   blst_scalar *dense_err_scalars_out,
                                   blst_p2_affine *dense_err_affines_out,
                                   const blst_p2_affine *bases_in, size_t n,
                                   double noise_rate, unsigned int seed) {
  srand(seed);
  size_t t_count = 0;

  memset(err_scalars_out, 0, n * sizeof(blst_scalar));

  for (size_t i = 0; i < n; i++) {
    double coin = (double)rand() / (double)RAND_MAX;
    if (coin < noise_rate) {
      blst_scalar s;
      for (size_t j = 0; j < 31; j++) {
        s.b[j] = (uint8_t)rand();
      }
      s.b[31] = (uint8_t)rand() & 0x3F;

      err_scalars_out[i] = s;
      dense_err_scalars_out[t_count] = s;
      dense_err_affines_out[t_count] = bases_in[i];

      t_count++;
    }
  }
  return t_count;
}

// Inner product logic utilizing blst (left in for utility/small lengths)
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
    blst_fr_from_scalar(&tmp, &x[i]);
    blst_fr_mul(&tmp, &tmp, &rho_super[i]);
    blst_fr_add(&acc, &acc, &tmp);
  }

  blst_scalar_from_fr(out, &acc);
}

// -----------------------------------------------------------------------------
// NTT-based Toeplitz Multiplication (from old.c)
// -----------------------------------------------------------------------------

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

void compute_lpn_toeplitz_ntt_c(blst_scalar *blinded_x, const blst_scalar *err,
                                const blst_scalar *s, const blst_scalar *x,
                                const blst_scalar *toeplitz_vec,
                                const blst_scalar *fwd_root,
                                const blst_scalar *inv_root,
                                const blst_scalar *inv_N, size_t n,
                                size_t kappa, size_t log_n) {
  size_t N = (size_t)1 << log_n;

  blst_fr *S_pad = (blst_fr *)calloc(N, sizeof(blst_fr));
  blst_fr *C = (blst_fr *)calloc(N, sizeof(blst_fr));

  for (size_t k = 0; k < (n + kappa - 1); k++) {
    blst_fr_from_scalar(&C[k], &toeplitz_vec[k]);
  }

  blst_fr f_root, i_root, i_n;
  blst_fr_from_scalar(&f_root, fwd_root);
  blst_fr_from_scalar(&i_root, inv_root);
  blst_fr_from_scalar(&i_n, inv_N);
  ntt_core_low_mem(C, log_n, &f_root);

  // can't be precomputed

  for (size_t j = 0; j < kappa; j++) {
    blst_fr_from_scalar(&S_pad[j], &s[kappa - 1 - j]);
  }

  ntt_core_low_mem(S_pad, log_n, &f_root);

  for (size_t i = 0; i < N; i++) {
    blst_fr_mul(&S_pad[i], &S_pad[i], &C[i]);
  }

  ntt_core_low_mem(S_pad, log_n, &i_root);

  for (size_t i = 0; i < n; i++) {
    blst_fr val, err_fr, x_fr;

    blst_fr_mul(&val, &S_pad[i + kappa - 1], &i_n);

    blst_fr_from_scalar(&err_fr, &err[i]);
    blst_fr_from_scalar(&x_fr, &x[i]);

    blst_fr_add(&val, &val, &err_fr);
    blst_fr_add(&val, &val, &x_fr);

    blst_scalar_from_fr(&blinded_x[i], &val);
  }

  free(S_pad);
  free(C);
}
}
