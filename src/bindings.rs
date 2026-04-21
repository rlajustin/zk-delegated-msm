use blst::{blst_fr, blst_p2_affine, blst_scalar};

extern "C" {
    pub fn precompute_rho_super(ret: *mut blst_fr, rho: *const blst_scalar);

    pub fn blst_fr_inner_product_fast(
        ret: *mut blst_scalar,
        x: *const blst_scalar,
        rho_super: *const blst_fr,
        len: usize,
    );

    pub fn compute_lpn_toeplitz_ntt_c(
        blinded_x: *mut blst_scalar,
        err: *const blst_scalar,
        s: *const blst_scalar,
        x: *const blst_scalar,
        toeplitz_vec: *const blst_scalar,
        fwd_root: *const blst_scalar,
        inv_root: *const blst_scalar,
        inv_N: *const blst_scalar,
        n: usize,
        kappa: usize,
        log_n: usize,
    );

    pub fn sample_errors_and_affines_c(
        err_scalars_out: *mut blst_scalar,
        dense_err_scalars_out: *mut blst_scalar,
        dense_err_affines_out: *mut blst_p2_affine,
        bases_in: *const blst_p2_affine,
        n: usize,
        noise_rate: f64,
        seed: u32,
    ) -> usize;

    pub fn generate_trapdoor_matrix_c(
        matrix_out: *mut blst_scalar,
        n: usize,
        kappa: usize,
        seed: u32,
    );

    pub fn compute_trapdoor_product_c(
        output: *mut blst_scalar,
        matrix_in: *const blst_scalar,
        x: *const blst_scalar,
        n: usize,
        kappa: usize,
    );

    pub fn compute_toeplitz_ntl_c(
        output: *mut blst_scalar,
        toeplitz_vec: *const blst_scalar,
        x: *const blst_scalar,
        n: usize,
        kappa: usize,
    );

    pub fn compute_lpn_sampling_ntl_c(
        blinded_x: *mut blst_scalar,
        err: *const blst_scalar,
        s: *const blst_scalar,
        x: *const blst_scalar,
        toeplitz_vec: *const blst_scalar,
        n: usize,
        kappa: usize,
    );
}
