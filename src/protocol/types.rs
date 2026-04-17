use blst::{blst_fr, blst_p2, blst_scalar, p2_affines};

pub struct DelegatedMsmSk {
    pub r: blst_scalar,
    pub rho_super: Vec<blst_fr>,
    pub q_point: blst_p2,
    pub mt_p: Option<p2_affines>,
    pub m_matrix_toeplitz: Option<Vec<blst_scalar>>,
    pub ntt_fwd_root: Option<blst_scalar>,
    pub ntt_inv_root: Option<blst_scalar>,
    pub ntt_inv_n: Option<blst_scalar>,
}

impl Default for DelegatedMsmSk {
    fn default() -> Self {
        Self {
            r: blst_scalar::default(),
            rho_super: vec![blst_fr::default(); 0],
            q_point: blst_p2::default(),
            mt_p: None,
            m_matrix_toeplitz: None,
            ntt_fwd_root: None,
            ntt_inv_root: None,
            ntt_inv_n: None,
        }
    }
}

pub struct DelegatedMsmPk {
    pub t_bases: p2_affines,
}

#[derive(Default)]
pub struct DelegatedMsmAux {
    pub inner_product: blst_scalar,
    pub corr: blst_p2,
}

#[derive(Default)]
pub struct DelegatedMsmPf {
    pub a_result: blst_p2,
    pub b_result: blst_p2,
}

pub struct ZkParams {
    pub n: usize,
    pub kappa: usize,
}
