#include "trapdoor.hpp"
#include <NTL/mat_ZZ_p.h>
#include <NTL/vec_ZZ_p.h>
#include <openssl/evp.h>
#include <random>

using namespace NTL;

struct TrapdoorMatrix {
  mat_ZZ_p B; // (n-kappa) x kappa
  mat_ZZ_p R; // kappa x (n-kappa)
  mat_ZZ_p A; // n x kappa
  size_t n;
  size_t kappa;
};

extern "C" void init_ntl_bls12_381(void);

// --- Fast ChaCha20 Stream for Matrix Generation ---
class ChaChaStream {
private:
  EVP_CIPHER_CTX *ctx;
  unsigned char block[4096];
  int block_pos;

public:
  ChaChaStream(uint32_t seed) {
    ctx = EVP_CIPHER_CTX_new();
    unsigned char key[32] = {0};
    memcpy(key, &seed, 4); // Expand 32-bit seed into 256-bit key
    unsigned char iv[16] = {0};

    EVP_EncryptInit_ex(ctx, EVP_chacha20(), NULL, key, iv);
    block_pos = 4096; // Force initial refill
  }

  ~ChaChaStream() { EVP_CIPHER_CTX_free(ctx); }

  void refill() {
    static const unsigned char zeros[4096] = {0};
    int outlen;
    EVP_EncryptUpdate(ctx, block, &outlen, zeros, 4096);
    block_pos = 0;
  }

  void next_zz_p(ZZ_p &out) {
    if (block_pos + 32 > 4096)
      refill();
    unsigned char *buf = &block[block_pos];
    buf[31] &= 0x1F; // Mask to fit safely inside BLS12-381 scalar field

    ZZ tmp;
    ZZFromBytes(tmp, buf, 32);
    out = to_ZZ_p(tmp);
    block_pos += 32;
  }

  void next_binary(ZZ_p &out) {
    if (block_pos + 1 > 4096)
      refill();
    out = to_ZZ_p(block[block_pos] & 1);
    block_pos += 1;
  }
};

// --- Standard PRG for Sparse Delegation Noise ---
static ZZ_p sample_field_element(std::mt19937 &gen) {
  unsigned char buf[32];
  for (int i = 0; i < 8; i++) {
    uint32_t r = gen();
    memcpy(buf + (i * 4), &r, 4);
  }
  buf[31] &= 0x1F;

  ZZ tmp;
  ZZFromBytes(tmp, buf, 32);
  return to_ZZ_p(tmp);
}

// --- Helpers ---
static void zz_p_to_blst_scalar(blst_scalar *out, const ZZ_p &v) {
  unsigned char buf[32];
  BytesFromZZ(buf, rep(v), 32);
  blst_scalar_from_lendian(out, buf);
}

static void blst_scalar_to_zz_p(ZZ_p &out, const blst_scalar *s) {
  unsigned char buf[32];
  blst_lendian_from_scalar(buf, s);
  ZZ tmp;
  ZZFromBytes(tmp, buf, 32);
  out = to_ZZ_p(tmp);
}

static void fr2z(ZZ_p &out, const blst_fr *f) {
  unsigned char buf[32];
  blst_scalar s;
  blst_scalar_from_fr(&s, f);
  blst_lendian_from_scalar(buf, &s);
  ZZ tmp;
  ZZFromBytes(tmp, buf, 32);
  out = to_ZZ_p(tmp);
}

extern "C" {

TrapdoorMatrix *trapdoor_generate(size_t n, size_t kappa, unsigned int seed) {
  init_ntl_bls12_381();
  if (n <= kappa)
    return nullptr;

  ChaChaStream stream(seed);
  TrapdoorMatrix *td = new TrapdoorMatrix();
  td->n = n;
  td->kappa = kappa;
  size_t extra = n - kappa;

  // 1. Sample B: (n-kappa) x kappa
  td->B.SetDims(extra, kappa);
  for (long i = 0; i < (long)extra; i++) {
    for (long j = 0; j < (long)kappa; j++) {
      stream.next_zz_p(td->B[i][j]);
    }
  }

  // 2. Sample R: kappa x (n-kappa). Small values (0, 1)
  td->R.SetDims(kappa, extra);
  for (long i = 0; i < (long)kappa; i++) {
    for (long j = 0; j < (long)extra; j++) {
      stream.next_binary(td->R[i][j]);
    }
  }

  // 3. Construct A = [ RB + I ]
  //                  [    B   ]
  mat_ZZ_p RB = td->R * td->B;
  td->A.SetDims(n, kappa);

  for (long j = 0; j < (long)kappa; j++) {
    // Top section: RB + I
    for (long i = 0; i < (long)kappa; i++) {
      td->A[i][j] = RB[i][j] + (i == j ? to_ZZ_p(1) : to_ZZ_p(0));
    }
    // Bottom section: B
    for (long i = 0; i < (long)extra; i++) {
      td->A[kappa + i][j] = td->B[i][j];
    }
  }

  return td;
}

void trapdoor_free(TrapdoorMatrix *td) { delete td; }

void preprocess_zk_trapdoor_ntl(blst_scalar *A_out, TrapdoorMatrix **td_out,
                                size_t n, size_t kappa, unsigned int seed) {
  *td_out = trapdoor_generate(n, kappa, seed);
  if (!(*td_out))
    return;

  // Export A for Rust's MSM precomputation
  for (long i = 0; i < (long)n; i++) {
    for (long j = 0; j < (long)kappa; j++) {
      zz_p_to_blst_scalar(&A_out[i * kappa + j], (*td_out)->A[i][j]);
    }
  }
}

size_t delegate_trapdoor_ntl(const TrapdoorMatrix *td, blst_scalar *blinded_x,
                             blst_scalar *s_out, blst_scalar *inner_product_out,
                             blst_scalar *dense_err_scalars_out,
                             blst_p2_affine *dense_err_affines_out,
                             const blst_scalar *x,
                             const blst_p2_affine *bases_in, const blst_fr *rho,
                             double noise_rate, unsigned int seed) {

  init_ntl_bls12_381();
  std::mt19937 gen(seed);

  size_t n = td->n;
  size_t kappa = td->kappa;

  // 1. Error Generation (Geometric Skipping)
  std::geometric_distribution<size_t> dist(noise_rate);
  vec_ZZ_p err;
  err.SetLength(n); // Defaults to zero

  size_t t_count = 0;
  size_t next_err_idx = dist(gen);

  while (next_err_idx < n) {
    ZZ_p r = sample_field_element(gen);
    if (IsZero(r))
      r = to_ZZ_p(1);

    err[next_err_idx] = r;
    zz_p_to_blst_scalar(&dense_err_scalars_out[t_count], r);
    dense_err_affines_out[t_count] = bases_in[next_err_idx];

    t_count++;
    next_err_idx += (dist(gen) + 1);
  }

  // 2. Sample LPN secret 's' (size kappa)
  vec_ZZ_p s;
  s.SetLength(kappa);
  for (size_t i = 0; i < kappa; i++) {
    s[i] = sample_field_element(gen);
  }

  // 3. Compute blinded_x = As + err + x
  vec_ZZ_p Bu = td->B * s;
  vec_ZZ_p RBu = td->R * Bu;
  vec_ZZ_p blinded_x_vec;
  blinded_x_vec.SetLength(n);

  for (size_t i = 0; i < n; i++) {
    ZZ_p x_zz;
    blst_scalar_to_zz_p(x_zz, &x[i]);

    if (i < (long)kappa) {
      // Top: (RB + I)s + err + x
      blinded_x_vec[i] = RBu[i] + s[i] + err[i] + x_zz;
    } else {
      // Bottom: Bs + err + x
      blinded_x_vec[i] = Bu[i - kappa] + err[i] + x_zz;
    }
    zz_p_to_blst_scalar(&blinded_x[i], blinded_x_vec[i]);
  }

  // 4. Export Secret u (for Rust postprocess)
  for (size_t i = 0; i < kappa; i++) {
    zz_p_to_blst_scalar(&s_out[i], s[i]);
  }

  // 5. Inner Product with rho
  vec_ZZ_p rho_vec;
  rho_vec.SetLength(n);
  for (size_t i = 0; i < n; i++) {
    fr2z(rho_vec[i], &rho[i]);
  }

  ZZ_p res;
  InnerProduct(res, blinded_x_vec, rho_vec);
  zz_p_to_blst_scalar(inner_product_out, res);

  return t_count;
}
}
