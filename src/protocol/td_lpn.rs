use crate::bindings::{
    delegate_trapdoor_ntl, preprocess_zk_trapdoor_ntl, trapdoor_free, trapdoor_generate,
    TrapdoorMatrix,
};
use crate::io::{load_td_sk, point_to_hex, save_td_sk, ClientRequest};
use crate::protocol::{HasMsmBase, MsmBase};
use crate::timer::Timer;
use crate::{
    compute_msm, compute_msm_slice, compute_mt_p_trapdoor_server_aided, preprocess_2g2t_logic,
    random_scalar, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol, LatticeParams,
};
use blst::{
    blst_p2, blst_p2_add_or_double, blst_p2_affine, blst_p2_cneg, blst_p2_is_equal, blst_p2_mult,
    blst_scalar, p2_affines,
};
use rand::Rng;
use std::sync::mpsc::Sender;
use std::time::Duration;

pub struct TrapdoorPtr(pub *mut TrapdoorMatrix);

impl Drop for TrapdoorPtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { trapdoor_free(self.0) };
        }
    }
}

#[derive(Default)]
pub struct TdSk {
    pub base: MsmBase,
    pub mt_p: Option<p2_affines>,
    // Opaque NTL TrapdoorMatrix — None until preprocess_zk runs
    pub trapdoor: Option<TrapdoorPtr>,
    // Seed stored so we can cheaply reconstruct on load (no serialization needed)
    pub trapdoor_seed: Option<u32>,
    pub n: usize,
    pub kappa: usize,
}

#[derive(Default)]
pub struct TdAux {
    pub inner_product: blst_scalar,
    pub corr: blst_p2,
}

pub struct TdMsm {
    pub kappa: usize,
    pub noise_rate: f64,
}

impl TdMsm {
    pub fn new(kappa: usize, noise_rate: f64) -> Self {
        Self { kappa, noise_rate }
    }
}

impl HasMsmBase for TdSk {
    fn from_base(base: MsmBase) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }
    fn base(&self) -> &MsmBase {
        &self.base
    }
}

impl DelegatedMsmProtocol for TdMsm {
    type SecretKey = TdSk;
    type Auxiliary = TdAux;

    fn load_secret_key(base_dir: &str, params: LatticeParams) -> std::io::Result<TdSk> {
        // Loads seed from disk, rebuilds TrapdoorMatrix in-memory — no matrix I/O
        let mut sk = load_td_sk(base_dir, params)?;

        if let Some(seed) = sk.trapdoor_seed {
            let timer = Timer::new();
            println!("Generating trapdoor matrix");
            let ptr = unsafe { trapdoor_generate(sk.n, sk.kappa, seed) };
            assert!(!ptr.is_null(), "trapdoor_generate failed on load");
            println!("Trapdoor matrix generation took {:?}", timer.elapsed());
            sk.trapdoor = Some(TrapdoorPtr(ptr));
        }
        Ok(sk)
    }

    fn save_secret_key(base_dir: &str, sk: &Self::SecretKey) -> std::io::Result<()> {
        save_td_sk(base_dir, sk)
    }

    fn preprocess(&self, n: usize, bases: &p2_affines) -> (MsmBase, DelegatedMsmPk, Duration) {
        let timer = Timer::new();
        let r = random_scalar();
        let (rho_super, q_point, t_bases_vec) = preprocess_2g2t_logic(bases, n, &r);
        (
            MsmBase {
                r,
                rho_super,
                q_point,
            },
            DelegatedMsmPk {
                t_bases: p2_affines::from(&t_bases_vec),
            },
            timer.elapsed(),
        )
    }

    fn preprocess_zk(
        &self,
        n: usize,
        kappa: usize,
        _bases: &p2_affines,
        server: &Sender<ClientRequest>,
        sk: &mut Self::SecretKey,
        _pk: &mut DelegatedMsmPk,
    ) -> Duration {
        let mut timer = Timer::new();
        let seed: u32 = rand::thread_rng().gen();

        let mut a_matrix_flat = vec![blst_scalar::default(); n * kappa];
        let mut td_ptr: *mut TrapdoorMatrix = std::ptr::null_mut();

        unsafe {
            preprocess_zk_trapdoor_ntl(a_matrix_flat.as_mut_ptr(), &mut td_ptr, n, kappa, seed);
        }
        assert!(
            !td_ptr.is_null(),
            "delegate_trapdoor_preprocess_ntl returned null"
        );

        let mt_p_vec =
            compute_mt_p_trapdoor_server_aided(&a_matrix_flat, server, &mut timer, n, kappa, sk);

        // 4. Update the Secret Key
        sk.trapdoor = Some(TrapdoorPtr(td_ptr));
        sk.trapdoor_seed = Some(seed);
        sk.n = n;
        sk.kappa = kappa;
        sk.mt_p = Some(p2_affines::from(&mt_p_vec));

        timer.elapsed()
    }

    fn delegate(
        &self,
        kappa: usize,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        x_scalars: &[blst_scalar],
    ) -> (Vec<blst_scalar>, Self::Auxiliary, Duration) {
        let timer = Timer::new();
        let n = x_scalars.len();

        let td_ptr = sk.trapdoor.as_ref().expect("trapdoor not initialized").0;
        let samp_seed: u32 = rand::thread_rng().gen();

        // Prepare Output Buffers
        let mut blinded_x = vec![blst_scalar::default(); n];
        let mut s_vec = vec![blst_scalar::default(); kappa];
        // let mut e_vec = vec![blst_scalar::default(); n];
        let mut inner_product = blst_scalar::default();

        // Allocate maximum possible dense size (n) to be safe
        let mut dense_err_scalars = vec![blst_scalar::default(); n];
        let mut dense_err_affines = vec![blst_p2_affine::default(); n];

        // Perform all NTL logic in one C FFI call
        let actual_t = unsafe {
            delegate_trapdoor_ntl(
                td_ptr,
                blinded_x.as_mut_ptr(),
                s_vec.as_mut_ptr(),
                // e_vec.as_mut_ptr(),
                &mut inner_product,
                dense_err_scalars.as_mut_ptr(),
                dense_err_affines.as_mut_ptr(),
                x_scalars.as_ptr(),
                bases.as_slice().as_ptr(),
                sk.base.rho_super.as_ptr() as *const ark_bls12_381::Fr as *const _,
                self.noise_rate,
                samp_seed,
            )
        };

        // Shrink the dense arrays to the actual number of non-zero errors generated
        dense_err_scalars.truncate(actual_t);
        dense_err_affines.truncate(actual_t);

        // Core MSMs
        let s_mtp = compute_msm(sk.mt_p.as_ref().unwrap(), &s_vec);
        let e_p = compute_msm_slice(&dense_err_affines, &dense_err_scalars);

        let mut corr = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut corr, &s_mtp, &e_p);
            blst_p2_cneg(&mut corr, true);
        }

        (
            blinded_x,
            TdAux {
                inner_product,
                corr,
            },
            timer.elapsed(),
        )
    }
    fn compute(
        &self,
        bases: &p2_affines,
        pk: &DelegatedMsmPk,
        message: &[blst_scalar],
    ) -> DelegatedMsmPf {
        DelegatedMsmPf {
            a_result: compute_msm(bases, message),
            b_result: compute_msm(&pk.t_bases, message),
        }
    }

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: DelegatedMsmPf,
    ) -> (Result<blst_p2, ()>, Duration) {
        let timer = Timer::new();
        let mut r_a = blst_p2::default();
        let mut s_q = blst_p2::default();
        let mut expected_b = blst_p2::default();

        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, sk.base.r.b.as_ptr(), 256);
            blst_p2_mult(
                &mut s_q,
                &sk.base.q_point,
                aux.inner_product.b.as_ptr(),
                256,
            );
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);

            if !blst_p2_is_equal(&proof.b_result, &expected_b) {
                return (Err(()), timer.elapsed());
            }

            let mut res = blst_p2::default();
            blst_p2_add_or_double(&mut res, &proof.a_result, &aux.corr);
            println!("final result: {}", point_to_hex(&res));
            (Ok(res), timer.elapsed())
        }
    }

    fn protocol_name() -> &'static str {
        "trapdoor"
    }
}
