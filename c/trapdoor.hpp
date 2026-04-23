#pragma once

#include "blst.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to keep NTL out of the public header
typedef struct TrapdoorMatrix TrapdoorMatrix;
typedef struct LpnSample LpnSample;

TrapdoorMatrix *trapdoor_generate(size_t n, size_t kappa, unsigned int seed);
void trapdoor_free(TrapdoorMatrix *td);

LpnSample *trapdoor_sample(const TrapdoorMatrix *td, const blst_scalar *x,
                           const blst_p2_affine *bases, double noise_rate,
                           unsigned int seed);

void trapdoor_sample_free(LpnSample *s);

void trapdoor_export_sample(const LpnSample *s, blst_scalar *b_out,
                            blst_scalar *s_out, blst_scalar *e_out);

size_t trapdoor_dense_count(const LpnSample *s);

void trapdoor_export_dense(const LpnSample *s, blst_scalar *dense_e_out,
                           blst_p2_affine *dense_bases_out,
                           size_t *dense_count_out);

void trapdoor_export_A(const TrapdoorMatrix *td, blst_scalar *A_out);

#ifdef __cplusplus
}
#endif
