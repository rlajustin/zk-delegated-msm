use std::sync::mpsc::Sender;

pub use protocol::types::{
    DelegatedMsmAux, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmSk, ZkParams,
};
use std::sync::mpsc::channel;

use blst::{
    blst_fr, blst_p2_add_or_double, blst_p2_from_affine, blst_p2_generator, blst_p2_is_equal,
    blst_p2_mult, blst_scalar_from_lendian, p2_affines, MultiPoint,
};

use rand::RngCore;
use rayon::prelude::*;

extern "C" {
    pub fn precompute_rho_super(ret: *mut blst_fr, rho: *const blst_scalar);

    pub fn blst_fr_inner_product_fast(
        ret: *mut blst_scalar,
        x: *const blst_scalar, // Changed to blst_scalar for clarity
        rho_super: *const blst_fr,
        len: usize,
    );

    pub fn compute_lpn_toeplitz_ntt_c(
        z: *mut blst_scalar,
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
}

pub mod client;
pub mod io;
pub mod protocol;
pub mod server;
pub mod timer;

pub use blst::{blst_p2, blst_p2_affine, blst_scalar};

pub use client::{MsmClient, MsmClientState};
pub use protocol::khabbazian::Khabbazian2G2T;
pub use protocol::zk_delegated_msm::ZkDelegatedMsm;
pub use protocol::DelegatedMsmProtocol;
pub use server::MsmServer;

use crate::io::ClientRequest;
use crate::timer::Timer;

pub fn generate_bases(n: usize) -> Vec<blst_p2> {
    let mut bases = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let mut ctr = 0;
    for i in 0..n {
        if i / (n / 100) > ctr {
            println!("{:}/100", ctr);
            ctr += 1;
        }
        let mut base_scalar = [0u8; 32];
        rng.fill_bytes(&mut base_scalar);
        base_scalar[31] &= 0x7F;
        let mut point = blst_p2::default();
        unsafe {
            blst_p2_mult(&mut point, blst_p2_generator(), base_scalar.as_ptr(), 256);
        }
        bases.push(point);
    }
    bases
}

pub fn generate_scalars(n: usize) -> Vec<blst_scalar> {
    let mut result: Vec<blst_scalar> = vec![blst_scalar::default(); n];
    for v in result.iter_mut() {
        *v = random_scalar();
    }
    result
}

pub fn random_scalar() -> blst_scalar {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes[31] &= 0x3F; // Ensure within range for bls12-381
    let mut scalar = blst_scalar::default();
    unsafe {
        blst_scalar_from_lendian(&mut scalar, bytes.as_ptr());
    }
    scalar
}

pub fn fast_inner_product_safe(
    x_scalars: &[blst_scalar],
    rho_super: &[blst_fr],
    n: usize,
) -> blst_scalar {
    let mut out_scalar = blst_scalar::default();
    unsafe {
        blst_fr_inner_product_fast(&mut out_scalar, x_scalars.as_ptr(), rho_super.as_ptr(), n);
    }
    out_scalar
}

pub fn compute_msm(bases: &p2_affines, scalars: &[blst_scalar]) -> blst_p2 {
    // Cast the scalar slice to a byte slice
    let scalar_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const u8, scalars.len() * 32) };
    bases.mult(scalar_bytes, 256)
}

pub fn compute_msm_slice(bases: &[blst_p2_affine], scalars: &[blst_scalar]) -> blst_p2 {
    let scalar_bytes =
        unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const u8, scalars.len() * 32) };
    bases.mult(scalar_bytes, 256)
}
pub fn preprocess_2g2t_logic(
    bases: &p2_affines,
    n: usize,
    r: &blst_scalar,
) -> (Vec<blst_fr>, blst_p2, Vec<blst_p2>) {
    let mut rho_standard: Vec<blst_scalar> = vec![blst_scalar::default(); n];
    let mut rho_super: Vec<blst_fr> = vec![blst_fr::default(); n];
    let mut t_bases = Vec::with_capacity(n);

    for i in 0..n {
        let rho_i = random_scalar();
        rho_standard.push(rho_i);
        unsafe {
            precompute_rho_super(rho_super[i..].as_mut_ptr(), &rho_standard[i]);
        }
    }

    let mut q_point = blst_p2::default();
    let q_scalar = random_scalar();
    unsafe {
        blst_p2_mult(&mut q_point, blst_p2_generator(), q_scalar.b.as_ptr(), 256);
    }

    for i in 0..n {
        let mut rho_q = blst_p2::default();
        let mut r_p = blst_p2::default();
        let mut base_p = blst_p2::default();
        unsafe {
            blst_p2_from_affine(&mut base_p, &bases[i]);
            blst_p2_mult(&mut rho_q, &q_point, rho_standard[i].b.as_ptr(), 256);
            blst_p2_mult(&mut r_p, &base_p, r.b.as_ptr(), 256);
            let mut t_i = blst_p2::default();
            blst_p2_add_or_double(&mut t_i, &r_p, &rho_q);
            t_bases.push(t_i);
        }
    }

    (rho_super, q_point, t_bases)
}

pub fn compute_toeplitz_mt_p(
    toeplitz_vec: &[blst_scalar],
    bases: &p2_affines,
    n: usize,
    kappa: usize,
) -> Vec<blst_p2> {
    let process_column = |k: usize| {
        let column_scalars = &toeplitz_vec[k..(k + n)];
        compute_msm(bases, column_scalars)
    };

    (0..kappa).into_par_iter().map(process_column).collect()
}

pub fn compute_mt_p_server_aided(
    toeplitz_vec: &[blst_scalar],
    server: &Sender<ClientRequest>,
    timer: &mut Timer,
    n: usize,
    kappa: usize,
    sk: &DelegatedMsmSk,
) -> Vec<blst_p2> {
    let mut mt_p_results = Vec::with_capacity(kappa);

    for k in 0..kappa {
        let column_scalars = toeplitz_vec[k..(k + n)].to_vec();
        let mut r_a = blst_p2::default();
        let mut s_q = blst_p2::default();
        let mut expected_b = blst_p2::default();

        let (resp_tx, resp_rx) = channel();

        timer.pause();

        server
            .send(ClientRequest::Compute(column_scalars.clone(), resp_tx))
            .expect("Failed to send MSM request to server");

        let proof = resp_rx
            .recv()
            .expect("Server disconnected during preprocessing");

        timer.start();

        let inner_product = fast_inner_product_safe(&column_scalars, &sk.rho_super, n);
        unsafe {
            blst_p2_mult(&mut r_a, &proof.a, sk.r.b.as_ptr(), 256);
            blst_p2_mult(&mut s_q, &sk.q_point, inner_product.b.as_ptr(), 256);
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
        }
        if unsafe { blst_p2_is_equal(&proof.b, &expected_b) } {
            mt_p_results.push(proof.a);
        } else {
            panic!("Server messed up in preprocessing mt_p")
        }
    }

    mt_p_results
}
