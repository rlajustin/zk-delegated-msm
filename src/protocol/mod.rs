use crate::io::ClientRequest;
use blst::{blst_fr, blst_p2, blst_scalar, p2_affines};
use std::sync::mpsc::Sender;
use std::time::Duration;

pub struct DelegatedMsmPk {
    pub t_bases: p2_affines,
}

#[derive(Default)]
pub struct DelegatedMsmPf {
    pub a_result: blst_p2,
    pub b_result: blst_p2,
}

pub trait HasMsmBase {
    fn from_base(base: MsmBase) -> Self;
    fn base(&self) -> &MsmBase;
}

#[derive(Default)]
pub struct MsmBase {
    pub r: blst_scalar,
    pub rho_super: Vec<blst_fr>,
    pub q_point: blst_p2,
}

pub trait DelegatedMsmProtocol {
    type SecretKey: HasMsmBase;
    type Auxiliary;

    fn load_secret_key(base_dir: &str, params: LatticeParams) -> std::io::Result<Self::SecretKey>;

    fn save_secret_key(base_dir: &str, sk: &Self::SecretKey) -> std::io::Result<()>;

    fn preprocess(&self, n: usize, bases: &p2_affines) -> (MsmBase, DelegatedMsmPk, Duration);

    fn preprocess_zk(
        &self,
        n: usize,
        kappa: usize,
        bases: &crate::p2_affines,
        server: &Sender<ClientRequest>,
        sk: &mut Self::SecretKey,
        pk: &mut DelegatedMsmPk,
    ) -> Duration;

    fn delegate(
        &self,
        kappa: usize,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        x: &[blst_scalar],
    ) -> (Vec<blst_scalar>, Self::Auxiliary, Duration);

    fn compute(
        &self,
        bases: &p2_affines,
        pk: &DelegatedMsmPk,
        message: &[blst_scalar],
    ) -> DelegatedMsmPf;

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: DelegatedMsmPf,
    ) -> (Result<blst_p2, ()>, Duration);

    fn protocol_name() -> &'static str;
}

pub struct LatticeParams {
    pub n: usize,
    pub kappa: usize,
}

mod td_lpn;
mod toeplitz_lpn;
pub use td_lpn::*;
pub use toeplitz_lpn::*;
