use crate::DelegatedMsmProtocol;
use crate::{
    blst_p2, blst_p2_add_or_double, blst_p2_is_equal, blst_p2_mult, blst_scalar, compute_msm,
    fast_inner_product_safe, p2_affines, preprocess_2g2t_logic, random_scalar,
};

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

pub struct TwoGTwoTSk {
    pub r: blst_scalar,
    pub rho_bytes: Vec<u8>,
    pub q_point: blst_p2,
}

pub struct TwoGTwoTPk {
    pub t_bases: p2_affines,
}

pub struct TwoGTwoTMessage<'a> {
    pub msg: &'a [u8],
}

pub struct TwoGTwoTAux {
    pub inner_product: blst_scalar,
}

#[derive(Default)]
pub struct TwoGTwoTProof {
    pub a_result: blst_p2,
    pub b_result: blst_p2,
}

impl<'a> DelegatedMsmProtocol<'a, blst_p2> for Khabbazian2G2T {
    type SecretKey = TwoGTwoTSk;
    type PublicKey = TwoGTwoTPk;
    type Message = TwoGTwoTMessage<'a>;
    type Auxiliary = TwoGTwoTAux;
    type Proof = TwoGTwoTProof;

    fn preprocess(&self, n: usize, bases: &p2_affines) -> (Self::SecretKey, Self::PublicKey) {
        let r = random_scalar();

        let (rho_bytes, q_point, t_bases) = preprocess_2g2t_logic(bases, n, &r);
        let t_bases = p2_affines::from(&t_bases);

        (
            TwoGTwoTSk {
                r,
                rho_bytes,
                q_point,
            },
            TwoGTwoTPk { t_bases },
        )
    }

    fn delegate(
        &self,
        _bases: &p2_affines,
        sk: &Self::SecretKey,
        x: &'a [u8], // Tied to lifetime 'a
    ) -> (Self::Message, Self::Auxiliary) {
        let n = x.len() / 32;

        let inner_product = fast_inner_product_safe(x, &sk.rho_bytes, n);

        (TwoGTwoTMessage { msg: x }, TwoGTwoTAux { inner_product })
    }

    fn compute(
        &self,
        bases: &p2_affines,
        pk: &Self::PublicKey,
        message: &Self::Message,
    ) -> Self::Proof {
        let a = compute_msm(bases, message.msg);
        let b = compute_msm(&pk.t_bases, message.msg);

        TwoGTwoTProof {
            a_result: a,
            b_result: b,
        }
    }

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> Result<blst_p2, ()> {
        let mut r_a = blst_p2::default();
        let mut s_q = blst_p2::default();
        let mut expected_b = blst_p2::default();

        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, sk.r.b.as_ptr(), 255);
            blst_p2_mult(&mut s_q, &sk.q_point, aux.inner_product.b.as_ptr(), 255);
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
        }

        if unsafe { blst_p2_is_equal(&proof.b_result, &expected_b) } {
            Ok(proof.a_result)
        } else {
            Err(())
        }
    }

    fn protocol_name() -> &'static str {
        "2G2T-Delegated-MSM"
    }
}
