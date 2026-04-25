#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/vec_ZZ_p.h>
#include <blst/blst.h>
#include <random>
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

static void fr2z(ZZ_p &out, const blst_fr *f) {
  byte buf[32];
  blst_scalar s;              // Change from *s to s
  blst_scalar_from_fr(&s, f); // Pass address
  blst_lendian_from_scalar(buf, &s);
  ZZ tmp;
  ZZFromBytes(tmp, buf, 32);
  out = to_ZZ_p(tmp);
}

static void z2b(blst_scalar *out, const ZZ_p &v) {
  byte buf[32];
  BytesFromZZ(buf, rep(v), 32);
  blst_scalar_from_lendian(out, buf);
}

size_t sample_lpn_err(vec_ZZ_p err_scalars_zz_p,
                      blst_scalar *dense_err_scalars_out,
                      blst_p2_affine *dense_err_affines_out,
                      blst_p2_affine *bases_in, double noise_rate, size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::geometric_distribution<size_t> dist(noise_rate);

  size_t i = dist(gen);
  size_t t_count = 0;

  while (i < n) {
    ZZ_p r;
    do {
      random(r);
    } while (IsZero(r));

    err_scalars_zz_p[i] = r;
    blst_scalar s;
    z2b(&s, r);

    dense_err_scalars_out[t_count] = s;
    dense_err_affines_out[t_count] = bases_in[i];

    t_count++;

    i += (dist(gen) + 1);
  }
  return t_count;
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

// Optimized version that takes NTL types directly
void ntl_inner_product(blst_scalar *out, const vec_ZZ_p &x,
                       const vec_ZZ_p &rho) {
  ZZ_p res;
  InnerProduct(res, x, rho);
  z2b(out, res);
}

size_t delegate_toeplitz_ntl(
    blst_scalar *blinded_x, blst_scalar *inner_product_out,
    blst_scalar *err_scalars_out, blst_scalar *dense_err_scalars_out,
    blst_p2_affine *dense_err_affines_out, const blst_scalar *x,
    const blst_scalar *s, const blst_p2_affine *bases_in,
    const blst_scalar *toeplitz_vec, const blst_fr *rho, size_t n, size_t kappa,
    double noise_rate, size_t seed) {

  init_ntl_bls12_381();

  // CONVERSION
  vec_ZZ_p x_vec, rho_vec;
  x_vec.SetLength(n);
  rho_vec.SetLength(n);

  ZZ_pX poly_toeplitz, poly_s;

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
  for (size_t i = 0; i < n; i++) {
    b2z(x_vec[i], &x[i]);
  }
  for (size_t i = 0; i < n; i++) {
    fr2z(rho_vec[i], &rho[i]);
  }

  // NOISE SAMPLING
  std::mt19937 gen(seed);
  std::geometric_distribution<size_t> dist(noise_rate);

  vec_ZZ_p err;
  err.SetLength(n);

  size_t t_count = 0;
  size_t next_err_idx = dist(gen);

  while (next_err_idx < n) {
    ZZ_p r;
    do {
      random(r);
    } while (IsZero(r));

    err[next_err_idx] = r;

    blst_scalar err;
    z2b(&err, r);
    if (err_scalars_out) {
      err_scalars_out[next_err_idx] = err;
    }
    dense_err_scalars_out[t_count] = err;
    dense_err_affines_out[t_count] = bases_in[next_err_idx];

    t_count++;
    next_err_idx += (dist(gen) + 1);
  }

  // COMPUTATION
  ZZ_pX result;
  mul(result, poly_toeplitz, poly_s);

  vec_ZZ_p blinded_x_vec;
  blinded_x_vec.SetLength(n);

  for (size_t i = 0; i < n; i++) {
    // val = (T * s)_i + e_i + x_i
    ZZ_p val = coeff(result, i + kappa - 1);
    val += err[i];
    val += x_vec[i];

    blinded_x_vec[i] = val;
    z2b(&blinded_x[i], val);
  }

  ntl_inner_product(inner_product_out, blinded_x_vec, rho_vec);

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
}
