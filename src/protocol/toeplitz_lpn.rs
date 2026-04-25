use crate::bindings::delegate_toeplitz_ntl;
use crate::io::{load_toeplitz_sk, point_to_hex, save_toeplitz_sk, ClientRequest};
use crate::protocol::{HasMsmBase, MsmBase};
use crate::timer::Timer;
use crate::{
    compute_msm, compute_msm_slice, compute_mt_p_toeplitz_server_aided, compute_toeplitz_mt_p,
    preprocess_2g2t_logic, random_scalar, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol,
    LatticeParams,
};
use ark_bls12_381::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use blst::{
    blst_p2, blst_p2_add_or_double, blst_p2_affine, blst_p2_cneg, blst_p2_is_equal, blst_p2_mult,
    blst_scalar, p2_affines,
};
use rand::Rng;
use std::sync::mpsc::Sender;
use std::time::Duration;

#[derive(Default)]
pub struct ToeplitzSk {
    pub base: MsmBase,
    pub mt_p: Option<p2_affines>,
    pub m_matrix_toeplitz: Option<Vec<blst_scalar>>,
    pub ntt_fwd_root: Option<blst_scalar>,
    pub ntt_inv_root: Option<blst_scalar>,
    pub ntt_inv_n: Option<blst_scalar>,
}

#[derive(Default)]
pub struct ToeplitzAux {
    pub inner_product: blst_scalar,
    pub corr: blst_p2,
}

static USE_SERVER_PREPROCESSING: bool = false;

pub fn get_log_n(n: usize, kappa: usize) -> usize {
    let ntt_size = (n + kappa - 1).next_power_of_two();
    ntt_size.trailing_zeros() as usize
}

fn generate_scalar_vector(n: usize) -> Vec<blst_scalar> {
    let mut s_vec: Vec<blst_scalar> = vec![blst_scalar::default(); n];
    for s in s_vec.iter_mut() {
        *s = random_scalar();
    }
    s_vec
}

pub struct ToeplitzMsm {
    pub kappa: usize,
    pub noise_rate: f64,
}

impl ToeplitzMsm {
    pub fn new(kappa: usize, noise_rate: f64) -> Self {
        Self { kappa, noise_rate }
    }
}

impl HasMsmBase for ToeplitzSk {
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

impl DelegatedMsmProtocol for ToeplitzMsm {
    type SecretKey = ToeplitzSk;
    type Auxiliary = ToeplitzAux;

    fn load_secret_key(base_dir: &str, params: LatticeParams) -> std::io::Result<ToeplitzSk> {
        load_toeplitz_sk(base_dir, params)
    }

    fn save_secret_key(base_dir: &str, sk: &Self::SecretKey) -> std::io::Result<()> {
        save_toeplitz_sk(base_dir, sk)
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
        bases: &p2_affines,
        server: &Sender<ClientRequest>,
        sk: &mut Self::SecretKey,
        _pk: &mut DelegatedMsmPk,
    ) -> Duration {
        let mut timer = Timer::new();
        let num_elements = n + kappa - 1;
        let toeplitz_vector = generate_scalar_vector(num_elements);

        // Delegate expensive MSM part to server
        let mt_p_vec = {
            if USE_SERVER_PREPROCESSING {
                compute_mt_p_toeplitz_server_aided(
                    &toeplitz_vector,
                    server,
                    &mut timer,
                    n,
                    kappa,
                    sk,
                )
            } else {
                compute_toeplitz_mt_p(&toeplitz_vector, bases, n, kappa)
            }
        };

        // Compute domain constants using arkworks
        let ntt_size = (n + kappa - 1).next_power_of_two();
        let domain = Radix2EvaluationDomain::<Fr>::new(ntt_size).unwrap();

        // Store only the minimal 32-byte field elements
        let fwd_root_bytes = domain.group_gen.into_bigint().to_bytes_le();
        let inv_root_bytes = domain.group_gen_inv.into_bigint().to_bytes_le();
        let inv_n_bytes = domain.size_inv().into_bigint().to_bytes_le();

        sk.ntt_fwd_root = Some(blst_scalar {
            b: fwd_root_bytes.try_into().unwrap(),
        });
        sk.ntt_inv_root = Some(blst_scalar {
            b: inv_root_bytes.try_into().unwrap(),
        });
        sk.ntt_inv_n = Some(blst_scalar {
            b: inv_n_bytes.try_into().unwrap(),
        });
        sk.mt_p = Some(p2_affines::from(&mt_p_vec));
        sk.m_matrix_toeplitz = Some(toeplitz_vector);

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

        let mut inner_product = blst_scalar::default();
        let mut blinded_x: Vec<blst_scalar> = vec![blst_scalar::default(); n];
        let mut err_scalars = vec![blst_scalar::default(); n];
        let mut dense_err_scalars = vec![blst_scalar::default(); n];
        let mut dense_err_affines = vec![blst_p2_affine::default(); n];
        let seed: u32 = rand::thread_rng().gen();

        let actual_t = unsafe {
            delegate_toeplitz_ntl(
                blinded_x.as_mut_ptr(),
                &mut inner_product,
                err_scalars.as_mut_ptr(),
                dense_err_scalars.as_mut_ptr(),
                dense_err_affines.as_mut_ptr(),
                x_scalars.as_ptr(),
                s_scalars.as_ptr(),
                bases.as_slice().as_ptr(),
                sk.m_matrix_toeplitz.as_ref().unwrap().as_ptr(),
                sk.base.rho_super.as_ptr(),
                n,
                kappa,
                self.noise_rate,
                seed,
            )
        };

        dense_err_scalars.truncate(actual_t);
        dense_err_affines.truncate(actual_t);

        let s_mtp = compute_msm(sk.mt_p.as_ref().unwrap(), &s_scalars);
        let e_p = compute_msm_slice(&dense_err_affines, &dense_err_scalars);

        let mut corr = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut corr, &s_mtp, &e_p);
            blst_p2_cneg(&mut corr, true);
        }

        (
            blinded_x, // Now sending x + z
            ToeplitzAux {
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
        "toeplitz"
    }
}
