use std::time::Duration;

use crate::io::ClientRequest;
use blst::{blst_p2, blst_scalar, p2_affines};
use std::sync::mpsc::Sender;
pub use types::{DelegatedMsmAux, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmSk};

pub trait DelegatedMsmProtocol<'a> {
    type SecretKey;
    type PublicKey;
    type BlindedMessage;
    type Auxiliary;
    type Proof;

    fn preprocess(
        &self,
        n: usize,
        bases: &p2_affines,
    ) -> (Self::SecretKey, Self::PublicKey, Duration);

    fn preprocess_zk(
        &self,
        n: usize,
        kappa: usize,
        bases: &crate::p2_affines,
        server: &Sender<ClientRequest>,
        sk: &mut Self::SecretKey,
        pk: &mut Self::PublicKey,
    ) -> Duration;

    fn delegate(
        &self,
        kappa: usize,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        x: &'a [blst_scalar],
    ) -> (Self::BlindedMessage, Self::Auxiliary, Duration);

    fn compute(
        &self,
        bases: &p2_affines,
        pk: &Self::PublicKey,
        message: &Self::BlindedMessage,
    ) -> Self::Proof;

    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> (Result<blst_p2, ()>, Duration);

    fn protocol_name() -> &'static str;
}

pub mod khabbazian;
pub mod types;
pub mod zk_delegated_msm;
