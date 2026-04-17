use std::time::Duration;

use crate::io::{point_to_hex, ClientRequest};
use crate::protocol::{
    DelegatedMsmAux, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol, DelegatedMsmSk,
};
use crate::timer::Timer;
use crate::{
    compute_lpn_toeplitz_ntt_c, compute_msm, compute_msm_slice, compute_mt_p_server_aided,
    compute_toeplitz_mt_p, fast_inner_product_safe, preprocess_2g2t_logic, random_scalar,
    sample_errors_and_affines_c,
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

static USE_SERVER_PREPROCESSING: bool = true;

pub struct ZkDelegatedMsm {
    pub kappa: usize,
    pub noise_rate: f64,
}

impl ZkDelegatedMsm {
    pub fn new(kappa: usize, noise_rate: f64) -> Self {
        Self { kappa, noise_rate }
    }
}

pub fn get_log_n(n: usize, kappa: usize) -> usize {
    let ntt_size = (n + kappa - 1).next_power_of_two();
    ntt_size.trailing_zeros() as usize
}

fn generate_scalar_vector(n: usize) -> Vec<blst_scalar> {
    let mut s = Vec::with_capacity(n);
    for _ in 0..n {
        s.push(random_scalar());
    }
    s
}

impl<'a> DelegatedMsmProtocol<'a> for ZkDelegatedMsm {
    type SecretKey = DelegatedMsmSk;
    type PublicKey = DelegatedMsmPk;
    type BlindedMessage = Vec<blst_scalar>;
    type Auxiliary = DelegatedMsmAux;
    type Proof = DelegatedMsmPf;

    fn preprocess(
        &self,
        n: usize,
        bases: &p2_affines,
    ) -> (Self::SecretKey, Self::PublicKey, Duration) {
        let timer = Timer::new();
        let r = random_scalar();
        let (rho_super, q_point, t_bases_vec) = preprocess_2g2t_logic(bases, n, &r);

        (
            DelegatedMsmSk {
                r,
                rho_super,
                q_point,
                mt_p: None,
                m_matrix_toeplitz: None,
                ntt_fwd_root: None,
                ntt_inv_root: None,
                ntt_inv_n: None,
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
        _pk: &mut Self::PublicKey,
    ) -> Duration {
        let mut timer = Timer::new();
        let num_elements = n + kappa - 1;
        let toeplitz_vector = generate_scalar_vector(num_elements);

        let mt_p_vec = {
            if USE_SERVER_PREPROCESSING {
                compute_mt_p_server_aided(&toeplitz_vector, server, &mut timer, n, kappa, sk)
            } else {
                compute_toeplitz_mt_p(&toeplitz_vector, bases, n, kappa)
            }
        };
        // Delegate expensive MSM part to server

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
        let mut m_matrix_toeplitz = vec![blst_scalar::default(); num_elements];
        for s in toeplitz_vector {
            m_matrix_toeplitz.push(s);
        }
        sk.m_matrix_toeplitz = Some(m_matrix_toeplitz);

        timer.elapsed()
    }

    fn delegate(
        &self,
        kappa: usize,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        x_scalars: &'a [blst_scalar],
    ) -> (Self::BlindedMessage, Self::Auxiliary, Duration) {
        let timer = Timer::new();

        let mt_p = sk.mt_p.as_ref().expect("mt_p missing");
        let m_matrix_toeplitz = sk
            .m_matrix_toeplitz
            .as_ref()
            .expect("m_matrix_toeplitz missing");
        let ntt_fwd_root = sk.ntt_fwd_root.as_ref().expect("ntt_fwd_root missing");
        let ntt_inv_root = sk.ntt_inv_root.as_ref().expect("ntt_inv_root missing");
        let ntt_inv_n = sk.ntt_inv_n.as_ref().expect("ntt_inv_n missing");

        let n = bases.as_slice().len();
        if x_scalars.len() != n {
            panic!(
                "Input scalars length ({}) does not match initialized state length ({})",
                x_scalars.len(),
                n
            );
        }
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

        let mut z_scalars: Vec<blst_scalar> = vec![blst_scalar::default(); n];
        let log_n = get_log_n(n, kappa);
        unsafe {
            compute_lpn_toeplitz_ntt_c(
                z_scalars.as_mut_ptr(),
                err_scalars.as_ptr(),
                s_scalars.as_ptr(),
                x_scalars.as_ptr(),
                m_matrix_toeplitz.as_ptr(),
                ntt_fwd_root,
                ntt_inv_root,
                ntt_inv_n,
                n,
                kappa,
                log_n,
            );
        }

        let inner_product = fast_inner_product_safe(&z_scalars, &sk.rho_super, n);
        let s_mtp = compute_msm(mt_p, &s_scalars);
        let e_p = compute_msm_slice(&dense_err_affines, &dense_err_scalars);

        let mut corr = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut corr, &s_mtp, &e_p);
            blst_p2_cneg(&mut corr, true);
        }

        println!("\n=== DELEGATE DEBUG ===");
        println!("n: {}", n);
        println!("s_mtp: {}", point_to_hex(&s_mtp));
        println!("e_p: {}", point_to_hex(&e_p));
        println!("corr = -(s_mtp + e_p): {}", point_to_hex(&corr));
        println!("=== END DEBUG ===\n");
        (
            z_scalars,
            DelegatedMsmAux {
                inner_product,
                corr,
            },
            timer.elapsed(),
        )
    }

    fn compute(
        &self,
        bases: &p2_affines,
        pk: &Self::PublicKey,
        message: &Self::BlindedMessage,
    ) -> Self::Proof {
        DelegatedMsmPf {
            a_result: compute_msm(bases, message),
            b_result: compute_msm(&pk.t_bases, message),
        }
    }

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> (Result<blst_p2, ()>, Duration) {
        let timer = Timer::new();
        let corr = aux.corr;

        println!("\n=== POSTPROCESS DEBUG ===");
        println!("proof.a_result: {}", point_to_hex(&proof.a_result));
        println!("proof.b_result: {}", point_to_hex(&proof.b_result));
        println!("sk.q_point: {}", point_to_hex(&sk.q_point));
        println!("aux.corr: {}", point_to_hex(&corr));

        let mut r_a = blst_p2::default();
        let mut s_q = blst_p2::default();
        let mut expected_b = blst_p2::default();

        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, sk.r.b.as_ptr(), 256);
            blst_p2_mult(&mut s_q, &sk.q_point, aux.inner_product.b.as_ptr(), 256);
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
            if !blst_p2_is_equal(&proof.b_result, &expected_b) {
                return (Err(()), timer.elapsed());
            }
            let mut res = blst_p2::default();
            blst_p2_add_or_double(&mut res, &proof.a_result, &corr);
            println!("final result = a_result + corr: {}", point_to_hex(&res));
            println!("=== END DEBUG ===\n");
            (Ok(res), timer.elapsed())
        }
    }

    fn protocol_name() -> &'static str {
        "ZK-2G2T-Delegated-MSM"
    }
}
