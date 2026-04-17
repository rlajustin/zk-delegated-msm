use crate::io::ClientRequest;
use crate::protocol::{
    DelegatedMsmAux, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol, DelegatedMsmSk,
};
use crate::timer::Timer;
use crate::{compute_msm, fast_inner_product_safe, preprocess_2g2t_logic, random_scalar};
use blst::{
    blst_p2, blst_p2_add_or_double, blst_p2_is_equal, blst_p2_mult, blst_scalar, p2_affines,
};
use std::sync::mpsc::Sender;
use std::time::Duration;

pub struct Khabbazian2G2T;

impl Default for Khabbazian2G2T {
    fn default() -> Self {
        Self
    }
}

impl Khabbazian2G2T {
    pub fn new() -> Self {
        Self
    }
}

impl<'a> DelegatedMsmProtocol<'a> for Khabbazian2G2T {
    type SecretKey = DelegatedMsmSk;
    type PublicKey = DelegatedMsmPk;
    type BlindedMessage = &'a [blst_scalar];
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
        let t_bases = p2_affines::from(&t_bases_vec);

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
            DelegatedMsmPk { t_bases },
            timer.elapsed(),
        )
    }

    fn preprocess_zk(
        &self,
        _kappa: usize,
        _n: usize,
        _bases: &p2_affines,
        _server: &Sender<ClientRequest>,
        _sk: &mut Self::SecretKey,
        _pk: &mut Self::PublicKey,
    ) -> Duration {
        let timer = Timer::new();
        timer.elapsed()
        // Basic khabbazian doesn't have ZK extension
    }

    fn delegate(
        &self,
        _kappa: usize,
        _bases: &p2_affines,
        sk: &Self::SecretKey,
        x: &'a [blst_scalar],
    ) -> (Self::BlindedMessage, Self::Auxiliary, Duration) {
        let timer = Timer::new();
        let n = x.len();
        let inner_product = fast_inner_product_safe(x, &sk.rho_super, n);
        (
            x,
            DelegatedMsmAux {
                inner_product,
                corr: blst_p2::default(),
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
        let a = compute_msm(bases, message);
        let b = compute_msm(&pk.t_bases, message);
        DelegatedMsmPf {
            a_result: a,
            b_result: b,
        }
    }

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> (Result<blst_p2, ()>, Duration) {
        let timer = Timer::new();
        let mut r_a = blst_p2::default();
        let mut s_q = blst_p2::default();
        let mut expected_b = blst_p2::default();

        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, sk.r.b.as_ptr(), 256);
            blst_p2_mult(&mut s_q, &sk.q_point, aux.inner_product.b.as_ptr(), 256);
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
        }

        if unsafe { blst_p2_is_equal(&proof.b_result, &expected_b) } {
            (Ok(proof.a_result), timer.elapsed())
        } else {
            (Err(()), timer.elapsed())
        }
    }

    fn protocol_name() -> &'static str {
        "2G2T-Delegated-MSM"
    }
}
