#include "blst.h"
#include <NTL/ZZ_pX.h>
#include <stdlib.h>
#include <string.h>

using namespace NTL;

static const uint8_t BLS12_381_R[32] = {
    0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0x02, 0xa4, 0xbd, 0x53, 0x05, 0xd8, 0xa1, 0x09, 0x08, 0xd8,
    0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7, 0xed, 0x73,
};

extern "C" void init_ntl_bls12_381(void) {
  static bool ntl_inited = false;
  if (ntl_inited)
    return;
  ZZ p;
  ZZFromBytes(p, BLS12_381_R, 32);
  ZZ_p::init(p);
  ntl_inited = true;
}

// Helpers local to this file
static void b2z(ZZ_p &out, const blst_scalar *s) {
  byte buf[32];
  blst_lendian_from_scalar(buf, s);
  ZZ tmp;
  ZZFromBytes(tmp, buf, 32);
  out = to_ZZ_p(tmp);
}

static void z2b(blst_scalar *out, const ZZ_p &v) {
  byte buf[32];
  BytesFromZZ(buf, rep(v), 32);
  blst_scalar_from_lendian(out, buf);
}

extern "C" {

void compute_toeplitz_ntl_c(blst_scalar *output,
                            const blst_scalar *toeplitz_vec,
                            const blst_scalar *x, size_t n, size_t kappa) {
  init_ntl_bls12_381();
  ZZ_pX poly_toeplitz, poly_x, result;

  for (size_t i = 0; i < n + kappa - 1; i++) {
    ZZ_p coeff;
    b2z(coeff, &toeplitz_vec[i]);
    SetCoeff(poly_toeplitz, i, coeff);
  }

  for (size_t i = 0; i < kappa; i++) {
    ZZ_p coeff;
    b2z(coeff, &x[kappa - 1 - i]);
    SetCoeff(poly_x, i, coeff);
  }

  mul(result, poly_toeplitz, poly_x);

  for (size_t i = 0; i < n; i++) {
    z2b(&output[i], coeff(result, i + kappa - 1));
  }
}

void compute_lpn_sampling_ntl_c(blst_scalar *blinded_x, const blst_scalar *err,
                                const blst_scalar *s, const blst_scalar *x,
                                const blst_scalar *toeplitz_vec, size_t n,
                                size_t kappa) {
  init_ntl_bls12_381();
  ZZ_pX poly_toeplitz, poly_s, result;

  for (size_t i = 0; i < n + kappa - 1; i++) {
    ZZ_p c;
    b2z(c, &toeplitz_vec[i]);
    SetCoeff(poly_toeplitz, i, c);
  }

  for (size_t i = 0; i < kappa; i++) {
    ZZ_p c;
    b2z(c, &s[kappa - 1 - i]);
    SetCoeff(poly_s, i, c);
  }

  mul(result, poly_toeplitz, poly_s);

  for (size_t i = 0; i < n; i++) {
    ZZ_p val = coeff(result, i + kappa - 1);
    ZZ_p e_zz, x_zz;
    b2z(e_zz, &err[i]);
    b2z(x_zz, &x[i]);
    val += e_zz + x_zz;
    z2b(&blinded_x[i], val);
  }
}

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

// void blst_fr_inner_product_fast(blst_scalar *out, const blst_scalar *x,
//                                 const blst_fr *rho_super, size_t n) {
//   blst_fr acc;
//   memset(&acc, 0, sizeof(blst_fr));
//
//   for (size_t i = 0; i < n; i++) {
//     blst_fr tmp;
//     blst_fr_from_scalar(&tmp, &x[i]);
//     blst_fr_mul(&tmp, &tmp, &rho_super[i]);
//     blst_fr_add(&acc, &acc, &tmp);
//   }
//
//   blst_scalar_from_fr(out, &acc);
// }

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
