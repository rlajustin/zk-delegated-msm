use crate::{
    blst_fr_from_scalar, blst_p2, blst_p2_add_or_double, blst_p2_affine, blst_p2_cneg,
    blst_p2_from_affine, blst_p2_generator, blst_p2_is_equal, blst_p2_mult, blst_scalar,
    compute_dense_mt_p, compute_lpn_toeplitz_ntt_c, compute_msm, compute_toeplitz_mt_p,
    fast_inner_product_safe, p2_affines, preprocess_2g2t_logic, random_scalar,
    sample_errors_and_affines_c, DelegatedMsmProtocol, MatrixType, USE_PARALLELISM,
};
use rand::Rng;

use ark_bls12_381::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

pub struct ZkDelegatedMsm {
    pub kappa: usize,
    pub noise_rate: f64,
    pub mat_type: MatrixType,
}

impl ZkDelegatedMsm {
    pub fn new(kappa: usize, noise_rate: f64, mat_type: MatrixType) -> Self {
        Self {
            kappa,
            noise_rate,
            mat_type,
        }
    }
}

pub struct ZkSk {
    pub r: blst_scalar,
    pub rho_bytes: Vec<u8>,
    pub q_point: blst_p2,
    pub mt_p: p2_affines,
    pub m_matrix_flat: Vec<u8>,
    pub kappa: usize,
    pub ntt_inv_roots_flat: Vec<u8>,
    pub ntt_roots_flat: Vec<u8>,
    pub ntt_inv_n: Vec<u8>,
    pub log_n: usize,
}

pub struct ZkPk {
    pub t_bases: p2_affines,
}

pub struct ZkMessage<'a> {
    pub m_masked: Vec<u8>,
    _phantom: std::marker::PhantomData<&'a u8>,
}

pub struct ZkAux {
    pub inner_product: blst_scalar,
    pub corr: blst_p2,
}

pub struct ZkProof {
    pub a_result: blst_p2,
    pub b_result: blst_p2,
}

pub struct ZkSkZk {
    pub zk_r: blst_scalar,
    pub zk_rho_bytes: Vec<u8>,
    pub zk_q_point: blst_p2,
    pub zk_t_bases: p2_affines,
}

pub struct ZkPkZk {
    pub zk_t_bases: p2_affines,
}

pub struct ZkMessageZk<'a> {
    pub m_masked: Vec<u8>,
    _phantom: std::marker::PhantomData<&'a u8>,
}

pub struct ZkAuxZk {
    pub inner_product: blst_scalar,
    pub corr: blst_p2,
}

pub struct ZkProofZk {
    pub a_result: blst_p2,
    pub b_result: blst_p2,
}

pub struct ZkDelegatedComputeResult {
    pub message: ZkMessageZk<'static>,
    pub auxiliary: ZkAuxZk,
    pub proof: ZkProofZk,
}

fn generate_scalar_vector(n: usize) -> Vec<blst_scalar> {
    let mut s = Vec::with_capacity(n);
    for _ in 0..n {
        s.push(random_scalar());
    }
    s
}

impl<'a> DelegatedMsmProtocol<'a, blst_p2> for ZkDelegatedMsm {
    type SecretKey = ZkSk;
    type PublicKey = ZkPk;
    type Message = ZkMessage<'a>;
    type Auxiliary = ZkAux;
    type Proof = ZkProof;

    fn preprocess(&self, n: usize, bases: &p2_affines) -> (Self::SecretKey, Self::PublicKey) {
        let kappa = self.kappa;
        let r = random_scalar();
        let (rho_bytes, q_point, t_bases_vec) = preprocess_2g2t_logic(bases, n, &r);

        let mt_p_vec: Vec<blst_p2>;
        let mut m_matrix_flat = Vec::new();

        match self.mat_type {
            MatrixType::Random => {
                let m_matrix_scalars = generate_scalar_vector(n * kappa);
                mt_p_vec = compute_dense_mt_p(&m_matrix_scalars, bases, n, kappa, USE_PARALLELISM);
                for s in m_matrix_scalars.iter() {
                    let mut mont_limb = [0u64; 4];
                    unsafe {
                        blst_fr_from_scalar(mont_limb.as_mut_ptr() as *mut _, s);
                    }
                    m_matrix_flat.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(mont_limb.as_ptr() as *const u8, 32)
                    });
                }
            }
            MatrixType::Toeplitz => {
                let num_elements = n + kappa - 1;
                let toeplitz_vector = generate_scalar_vector(num_elements);
                mt_p_vec =
                    compute_toeplitz_mt_p(&toeplitz_vector, bases, n, kappa, USE_PARALLELISM);
                for s in toeplitz_vector.iter() {
                    let mut mont_limb = [0u64; 4];
                    unsafe {
                        blst_fr_from_scalar(mont_limb.as_mut_ptr() as *mut _, s);
                    }
                    m_matrix_flat.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(mont_limb.as_ptr() as *const u8, 32)
                    });
                }
            }
        }

        let ntt_size = (n + kappa - 1).next_power_of_two();
        let log_n = ntt_size.trailing_zeros() as usize;
        let domain = Radix2EvaluationDomain::<Fr>::new(ntt_size).unwrap();

        let mut fwd_roots_flat = Vec::with_capacity(ntt_size * 32);
        for element in domain.elements() {
            fwd_roots_flat.extend_from_slice(&element.into_bigint().to_bytes_le());
        }

        let mut inv_roots_flat = Vec::with_capacity(ntt_size * 32);
        inv_roots_flat.extend_from_slice(&fwd_roots_flat[0..32]);
        for i in (1..ntt_size).rev() {
            inv_roots_flat.extend_from_slice(&fwd_roots_flat[i * 32..(i + 1) * 32]);
        }

        (
            ZkSk {
                r,
                rho_bytes,
                q_point,
                mt_p: p2_affines::from(&mt_p_vec),
                m_matrix_flat,
                kappa,
                ntt_inv_roots_flat: inv_roots_flat,
                ntt_roots_flat: fwd_roots_flat,
                ntt_inv_n: domain.size_inv().into_bigint().to_bytes_le(),
                log_n,
            },
            ZkPk {
                t_bases: p2_affines::from(&t_bases_vec),
            },
        )
    }

    fn delegate(
        &self,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        x_bytes: &'a [u8],
    ) -> (Self::Message, Self::Auxiliary) {
        let n = x_bytes.len() / 32;
        let s_scalars = generate_scalar_vector(sk.kappa);
        let mut err_bytes = vec![0u8; n * 32];
        let mut dense_err_scalars = vec![0u8; n * 32];
        let mut dense_err_affines = vec![blst_p2_affine::default(); n];
        let seed: u32 = rand::thread_rng().gen();

        let actual_t = unsafe {
            sample_errors_and_affines_c(
                err_bytes.as_mut_ptr(),
                dense_err_scalars.as_mut_ptr(),
                dense_err_affines.as_mut_ptr(),
                bases.as_slice().as_ptr(),
                n,
                self.noise_rate,
                seed,
            )
        };
        dense_err_scalars.truncate(actual_t * 32);
        dense_err_affines.truncate(actual_t);

        let mut s_bytes = vec![0u8; sk.kappa * 32];
        for (i, s) in s_scalars.iter().enumerate() {
            s_bytes[i * 32..(i + 1) * 32].copy_from_slice(&s.b);
        }

        let mut z_bytes = vec![0u8; n * 32];
        unsafe {
            compute_lpn_toeplitz_ntt_c(
                z_bytes.as_mut_ptr(),
                err_bytes.as_ptr(),
                s_bytes.as_ptr(),
                sk.m_matrix_flat.as_ptr(),
                sk.ntt_roots_flat.as_ptr(),
                sk.ntt_inv_roots_flat.as_ptr(),
                sk.ntt_inv_n.as_ptr(),
                n,
                sk.kappa,
                sk.log_n,
            );
        }

        let inner_product = fast_inner_product_safe(&z_bytes, &sk.rho_bytes, n);
        let s_mtp = compute_msm(&sk.mt_p, &s_bytes);
        let e_p = p2_affines::from(&dense_err_affines).mult(&dense_err_scalars, 255);

        let mut corr = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut corr, &s_mtp, &e_p);
            blst_p2_cneg(&mut corr, true);
        }

        (
            ZkMessage {
                m_masked: z_bytes,
                _phantom: std::marker::PhantomData,
            },
            ZkAux {
                inner_product,
                corr,
            },
        )
    }

    fn compute(
        &self,
        bases: &p2_affines,
        pk: &Self::PublicKey,
        msg: &Self::Message,
    ) -> Self::Proof {
        ZkProof {
            a_result: compute_msm(bases, &msg.m_masked),
            b_result: compute_msm(&pk.t_bases, &msg.m_masked),
        }
    }

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> Result<blst_p2, ()> {
        let (mut r_a, mut s_q, mut expected_b) =
            (blst_p2::default(), blst_p2::default(), blst_p2::default());
        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, sk.r.b.as_ptr(), 255);
            blst_p2_mult(&mut s_q, &sk.q_point, aux.inner_product.b.as_ptr(), 255);
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
            if !blst_p2_is_equal(&proof.b_result, &expected_b) {
                return Err(());
            }
            let mut res = blst_p2::default();
            blst_p2_add_or_double(&mut res, &proof.a_result, &aux.corr);
            Ok(res)
        }
    }

    fn protocol_name() -> &'static str {
        "ZK-2G2T-Delegated-MSM"
    }

    type ZkSecretKey = ZkSkZk;
    type ZkPublicKey = ZkPkZk;
    type ZkMessage<'b>
        = ZkMessageZk<'b>
    where
        Self: 'b;
    type ZkAuxiliary = ZkAuxZk;
    type ZkProof = ZkProofZk;

    fn preprocess_zk(
        &self,
        n: usize,
        bases: &p2_affines,
        _sk: &Self::SecretKey,
    ) -> (Self::ZkSecretKey, Self::ZkPublicKey) {
        let zk_r = random_scalar();
        let (zk_rho_bytes, zk_q_point, zk_t_bases_vec) = preprocess_2g2t_logic(bases, n, &zk_r);
        let zk_t_bases = p2_affines::from(&zk_t_bases_vec);
        (
            ZkSkZk {
                zk_r,
                zk_rho_bytes,
                zk_q_point,
                zk_t_bases: zk_t_bases.clone(),
            },
            ZkPkZk { zk_t_bases },
        )
    }

    fn delegate_zk(
        &self,
        bases: &p2_affines,
        soundness_sk: &Self::SecretKey,
        zk_sk: &Self::ZkSecretKey,
        x_bytes: &'a [u8],
    ) -> (Self::ZkMessage<'a>, Self::ZkAuxiliary, Self::ZkProof) {
        let result = self.delegate_zk_internal(bases, soundness_sk, zk_sk, x_bytes);
        (result.message, result.auxiliary, result.proof)
    }

    fn verify_zk(
        &self,
        _s_sk: &Self::SecretKey,
        zk_sk: &Self::ZkSecretKey,
        aux: &Self::ZkAuxiliary,
        proof: Self::ZkProof,
    ) -> Result<blst_p2, ()> {
        let (mut r_a, mut s_q, mut expected_b) =
            (blst_p2::default(), blst_p2::default(), blst_p2::default());
        unsafe {
            blst_p2_mult(&mut r_a, &proof.a_result, zk_sk.zk_r.b.as_ptr(), 255);
            blst_p2_mult(
                &mut s_q,
                &zk_sk.zk_q_point,
                aux.inner_product.b.as_ptr(),
                255,
            );
            blst_p2_add_or_double(&mut expected_b, &r_a, &s_q);
            if !blst_p2_is_equal(&proof.b_result, &expected_b) {
                return Err(());
            }
            let mut res = blst_p2::default();
            blst_p2_add_or_double(&mut res, &proof.a_result, &aux.corr);
            Ok(res)
        }
    }

    fn supports_zk_delegation() -> bool {
        true
    }
}

impl ZkDelegatedMsm {
    fn delegate_zk_internal(
        &self,
        bases: &p2_affines,
        soundness_sk: &ZkSk,
        zk_sk: &ZkSkZk,
        x_bytes: &[u8],
    ) -> ZkDelegatedComputeResult {
        let n = x_bytes.len() / 32;
        let s_scalars = generate_scalar_vector(soundness_sk.kappa);
        let mut err_bytes = vec![0u8; n * 32];
        let mut dense_err_scalars = vec![0u8; n * 32];
        let mut dense_err_affines = vec![blst_p2_affine::default(); n];
        let seed: u32 = rand::thread_rng().gen();

        unsafe {
            sample_errors_and_affines_c(
                err_bytes.as_mut_ptr(),
                dense_err_scalars.as_mut_ptr(),
                dense_err_affines.as_mut_ptr(),
                bases.as_slice().as_ptr(),
                n,
                self.noise_rate,
                seed,
            );
        }

        let mut s_bytes = vec![0u8; soundness_sk.kappa * 32];
        for (i, s) in s_scalars.iter().enumerate() {
            s_bytes[i * 32..(i + 1) * 32].copy_from_slice(&s.b);
        }

        let mut z_bytes = vec![0u8; n * 32];
        unsafe {
            compute_lpn_toeplitz_ntt_c(
                z_bytes.as_mut_ptr(),
                err_bytes.as_ptr(),
                s_bytes.as_ptr(),
                soundness_sk.m_matrix_flat.as_ptr(),
                soundness_sk.ntt_roots_flat.as_ptr(),
                soundness_sk.ntt_inv_roots_flat.as_ptr(),
                soundness_sk.ntt_inv_n.as_ptr(),
                n,
                soundness_sk.kappa,
                soundness_sk.log_n,
            );
        }

        let zk_inner_product = fast_inner_product_safe(&z_bytes, &zk_sk.zk_rho_bytes, n);
        let s_mtp = compute_msm(&soundness_sk.mt_p, &s_bytes);
        let e_p = p2_affines::from(&dense_err_affines).mult(&dense_err_scalars, 255);

        let mut corr = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut corr, &s_mtp, &e_p);
            blst_p2_cneg(&mut corr, true);
        }

        ZkDelegatedComputeResult {
            message: ZkMessageZk {
                m_masked: z_bytes.clone(),
                _phantom: std::marker::PhantomData,
            },
            auxiliary: ZkAuxZk {
                inner_product: zk_inner_product,
                corr,
            },
            proof: ZkProofZk {
                a_result: compute_msm(bases, &z_bytes),
                b_result: compute_msm(&zk_sk.zk_t_bases, &z_bytes),
            },
        }
    }
}
