use blst::{blst_fr, blst_p2_affine, blst_scalar};

// ── Opaque Types for Type Safety ──────────────────────────────────────────
// These structs have no size in Rust but provide distinct types for the linker.
#[repr(C)]
pub struct TrapdoorMatrix {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct LpnSample {
    _unused: [u8; 0],
}

extern "C" {
    // ── Trapdoor Lifecycle ──────────────────────────────────────────────────
    pub fn delegate_trapdoor_ntl(
        td: *const TrapdoorMatrix,
        blinded_x: *mut blst_scalar,
        s_out: *mut blst_scalar,
        // e_out: *mut blst_scalar,
        inner_product_out: *mut blst_scalar,
        dense_err_scalars_out: *mut blst_scalar,
        dense_err_affines_out: *mut blst_p2_affine,
        x: *const blst_scalar,
        bases_in: *const blst_p2_affine,
        rho: *const blst_fr, // Ensure this matches your blst_fr representation
        noise_rate: f64,
        seed: u32,
    ) -> usize;
    pub fn preprocess_zk_trapdoor_ntl(
        A_out: *mut blst_scalar,
        td_out: *mut *mut TrapdoorMatrix, // This must be a double pointer
        n: usize,
        kappa: usize,
        seed: u32,
    );
    pub fn trapdoor_generate(n: usize, kappa: usize, seed: u32) -> *mut TrapdoorMatrix;
    pub fn trapdoor_free(td: *mut TrapdoorMatrix);

    // ── Precomputation & Math Utils ─────────────────────────────────────────
    pub fn precompute_rho_super(ret: *mut blst_fr, rho: *const blst_scalar);

    pub fn blst_fr_inner_product_fast(
        ret: *mut blst_scalar,
        x: *const blst_scalar,
        rho_super: *const blst_fr,
        len: usize,
    );

    pub fn delegate_toeplitz_ntl(
        blinded_x: *mut blst_scalar,
        inner_product_out: *mut blst_scalar,
        err_scalars_out: *mut blst_scalar,
        dense_err_scalars_out: *mut blst_scalar,
        dense_err_affines_out: *mut blst_p2_affine,
        x: *const blst_scalar,
        s: *const blst_scalar,
        bases_in: *const blst_p2_affine,
        toeplitz_vec: *const blst_scalar,
        rho: *const blst_fr,
        n: usize,
        kappa: usize,
        noise_rate: f64,
        seed: u32,
    ) -> usize;
}
