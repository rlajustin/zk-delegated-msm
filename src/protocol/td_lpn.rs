use crate::bindings::{
    compute_trapdoor_product_c, generate_trapdoor_matrix_c, sample_errors_and_affines_c,
};
use crate::io::{point_to_hex, save_td_sk, ClientRequest};
use crate::protocol::{HasMsmBase, MsmBase};
use crate::timer::Timer;
use crate::{
    compute_msm, compute_msm_slice, fast_inner_product_safe, preprocess_2g2t_logic, random_scalar,
    DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol, LatticeParams,
};
use blst::{
    blst_fr, blst_fr_add, blst_fr_from_scalar, blst_p2, blst_p2_add_or_double, blst_p2_affine,
    blst_p2_cneg, blst_p2_is_equal, blst_p2_mult, blst_scalar, blst_scalar_from_fr, p2_affines,
};
use std::time::Duration;

use rand::Rng;
use std::sync::mpsc::Sender;

#[derive(Default)]
pub struct TdSk {
    pub base: MsmBase,
    pub mt_p: Option<p2_affines>,
    pub trapdoor_matrix: Option<Vec<blst_scalar>>,
    pub trapdoor_seed: Option<u32>,
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

fn generate_scalar_vector(n: usize) -> Vec<blst_scalar> {
    let mut s_vec: Vec<blst_scalar> = vec![blst_scalar::default(); n];
    for s in s_vec.iter_mut() {
        *s = random_scalar();
    }
    s_vec
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
        crate::io::load_td_sk(base_dir, params)
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
        _server: &Sender<ClientRequest>,
        sk: &mut Self::SecretKey,
        _pk: &mut DelegatedMsmPk,
    ) -> Duration {
        let timer = Timer::new();

        let matrix_size = n * kappa;
        let mut trapdoor_matrix: Vec<blst_scalar> = vec![blst_scalar::default(); matrix_size];
        let seed: u32 = rand::thread_rng().gen();

        unsafe {
            generate_trapdoor_matrix_c(trapdoor_matrix.as_mut_ptr(), n, kappa, seed);
        }

        sk.trapdoor_matrix = Some(trapdoor_matrix);
        sk.trapdoor_seed = Some(seed);

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

        let n = bases.as_slice().len();

        let s_scalars = generate_scalar_vector(kappa);
        let mut err_scalars = vec![blst_scalar::default(); n];
        let mut dense_err_scalars = vec![blst_scalar::default(); n];
        let mut dense_err_affines = vec![blst_p2_affine::default(); n];
        let seed: u32 = rand::thread_rng().gen();

        let actual_t = unsafe {
            sample_errors_and_affines_c(
                err_scalars.as_mut_ptr(),
                dense_err_scalars.as_mut_ptr(),
                dense_err_affines.as_mut_ptr(),
                bases.as_slice().as_ptr(),
                n,
                self.noise_rate,
                seed,
            )
        };

        dense_err_scalars.truncate(actual_t);
        dense_err_affines.truncate(actual_t);

        let mut z_output: Vec<blst_scalar> = vec![blst_scalar::default(); kappa];

        unsafe {
            compute_trapdoor_product_c(
                z_output.as_mut_ptr(),
                sk.trapdoor_matrix.as_ref().unwrap().as_ptr(),
                x_scalars.as_ptr(),
                n,
                kappa,
            );
        }

        let mut blinded_x: Vec<blst_scalar> = vec![blst_scalar::default(); n];
        for i in 0..n {
            let z_idx = i % kappa;
            let mut sum_fr = blst_fr::default();

            unsafe {
                let z_fr_ptr = &z_output[z_idx];
                let mut z_fr = blst_fr::default();
                blst_fr_from_scalar(&mut z_fr, z_fr_ptr);

                if i < actual_t {
                    let mut err_fr = blst_fr::default();
                    blst_fr_from_scalar(&mut err_fr, &err_scalars[i]);
                    blst_fr_add(&mut sum_fr, &z_fr, &err_fr);
                } else {
                    sum_fr = z_fr;
                }

                blst_scalar_from_fr(&mut blinded_x[i], &sum_fr);
            }
        }

        let inner_product = fast_inner_product_safe(&blinded_x, &sk.base.rho_super, n);
        let s_mtp = compute_msm(sk.mt_p.as_ref().unwrap(), &s_scalars);
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
        let corr = aux.corr;

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
            blst_p2_add_or_double(&mut res, &proof.a_result, &corr);

            println!("\n=== POSTPROCESS DEBUG ===");
            println!("final result = a_result + corr: {}", point_to_hex(&res));
            (Ok(res), timer.elapsed())
        }
    }

    fn protocol_name() -> &'static str {
        "trapdoor"
    }
}
