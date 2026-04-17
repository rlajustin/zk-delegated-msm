use crate::io::{
    init_level, load_2g2t_sk, load_bases_subset, load_pk, load_zk_sk, point_to_hex, save_2g2t_sk,
    save_pk, save_zk_sk, ClientRequest, CommStats,
};
use crate::{
    compute_msm, DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol, DelegatedMsmSk,
    ZkDelegatedMsm, ZkParams,
};
use blst::{blst_p2, blst_scalar, p2_affines};
use std::ops::Add;

use std::sync::mpsc::{channel, Sender};

pub struct MsmClientState {
    pub bases: p2_affines,
    pub sk: DelegatedMsmSk,
    pub pk: DelegatedMsmPk,
    pub n: usize,
    pub kappa: usize,
}

pub struct MsmClient {
    pub protocol: ZkDelegatedMsm,
    pub base_path: String,
    pub state: Option<MsmClientState>,
}

impl MsmClient {
    pub fn new(protocol: ZkDelegatedMsm, base_path: &str) -> Self {
        Self {
            protocol,
            base_path: base_path.to_string(),
            state: None,
        }
    }

    pub fn load_state(&mut self, params: ZkParams) -> std::io::Result<()> {
        load_zk_sk(&self.base_path, params)?;

        Ok(())
    }

    pub fn init_client(&mut self, n: usize, kappa: usize) -> std::io::Result<()> {
        let global_bases = load_bases_subset(&self.base_path, n)?;
        let sk;
        let pk;
        match init_level(&self.base_path) {
            0 => {
                let (s, p, preprocess_time) = self.protocol.preprocess(n, &global_bases);
                sk = s;
                pk = p;
                println!("2G2T preprocess took {:?}", preprocess_time);

                println!("Saving 2G2T State");
                save_2g2t_sk(&self.base_path, &sk)?;
                save_pk(&self.base_path, &pk)?;
            }
            1..=u8::MAX => {
                sk = load_2g2t_sk(&self.base_path)?;
                pk = load_pk(&self.base_path, n)?;
            }
        }

        self.state = Some(MsmClientState {
            bases: global_bases,
            sk,
            pk,
            n,
            kappa,
        });
        Ok(())
    }

    pub fn init_client_zk(&mut self, server: &Sender<ClientRequest>) -> std::io::Result<()> {
        let state = self.state.as_mut().expect("Client state missing");
        if init_level(&self.base_path) <= 1 {
            let preprocess_zk_time = self.protocol.preprocess_zk(
                state.n,
                state.kappa,
                &state.bases,
                server,
                &mut state.sk,
                &mut state.pk,
            );

            println!("ZK preprocess took {:?}", preprocess_zk_time);

            save_zk_sk(&self.base_path, &state.sk)?;

            println!("Session initialized and keys persisted to storage");
            Ok(())
        } else {
            state.sk = load_zk_sk(
                &self.base_path,
                ZkParams {
                    n: state.n,
                    kappa: state.kappa,
                },
            )?;
            state.pk = load_pk(&self.base_path, state.n)?;

            println!("Client fully initialized from files");

            Ok(())
        }
    }

    pub fn request(
        &self,
        server_tx: &Sender<ClientRequest>,
        scalars: &[blst_scalar],
        stats: &mut CommStats,
    ) -> (Result<blst_p2, String>, std::time::Duration) {
        let state = self.state.as_ref().expect("Client not booted");

        println!(
            "Expected: {}",
            point_to_hex(&compute_msm(&state.bases, scalars))
        );

        let (msg, aux, delegate_time) =
            self.protocol
                .delegate(state.kappa, &state.bases, &state.sk, scalars);

        stats.record_outbound_scalars(&msg);

        let (resp_tx, resp_rx) = channel();

        server_tx
            .send(ClientRequest::Compute(msg, resp_tx))
            .unwrap();

        let response = resp_rx.recv().expect("Server thread died");

        stats.record_inbound_points(2);

        // 5. Verification
        let proof = DelegatedMsmPf {
            a_result: response.a,
            b_result: response.b,
        };
        let (res, postprocess_time) = self.protocol.postprocess(&state.sk, &aux, proof);
        let total_time = delegate_time.add(postprocess_time);
        if res.is_err() {
            return (
                Err("Verification failed: Server provided invalid proof".to_string()),
                total_time,
            );
        }
        (Ok(res.unwrap()), total_time)
    }
}
