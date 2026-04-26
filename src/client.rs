use crate::io::{
    init_level, load_2g2t_sk, load_bases_subset, load_pk, save_2g2t_sk, save_pk, ClientRequest,
};
use crate::protocol::{HasMsmBase, LatticeParams};
use crate::{DelegatedMsmPf, DelegatedMsmPk, DelegatedMsmProtocol};
use blst::{blst_p2, blst_scalar, p2_affines};
use std::ops::Add;
use std::sync::mpsc::{channel, SyncSender, TrySendError};
use std::thread::sleep;
use std::time::{Duration, Instant};

pub struct MsmClientState<P: DelegatedMsmProtocol> {
    pub bases: p2_affines,
    pub sk: P::SecretKey,
    pub pk: DelegatedMsmPk,
    pub n: usize,
    pub kappa: usize,
}

pub struct MsmClient<P: DelegatedMsmProtocol> {
    pub protocol: P,
    pub base_dir: String,
    pub state: Option<MsmClientState<P>>,
}

impl<P: DelegatedMsmProtocol> MsmClient<P> {
    pub fn new(protocol: P, base_dir: &str) -> Self {
        Self {
            protocol,
            base_dir: base_dir.to_string(),
            state: None,
        }
    }

    pub fn init_client(&mut self, n: usize, kappa: usize) -> std::io::Result<()> {
        let global_bases = load_bases_subset(&self.base_dir, n)?;
        if init_level::<P>(&self.base_dir) <= 1 {
            let (base, p, preprocess_time) = self.protocol.preprocess(n, &global_bases);
            println!("2G2T preprocess took {:?}", preprocess_time);

            println!("Saving 2G2T State");
            save_2g2t_sk(&self.base_dir, &base)?;
            save_pk(&self.base_dir, &p)?;
        }

        let pk = load_pk(&self.base_dir, n)?;
        let sk = P::SecretKey::from_base(load_2g2t_sk(&self.base_dir)?);

        self.state = Some(MsmClientState {
            bases: global_bases,
            sk,
            pk,
            n,
            kappa,
        });
        Ok(())
    }

    pub fn init_client_zk(&mut self, server: &SyncSender<ClientRequest>) -> std::io::Result<()> {
        let state = self.state.as_mut().expect("Client state missing");
        if init_level::<P>(&self.base_dir) <= 1 {
            println!("Missing preprocessed lattice data, generating...");
            let preprocess_zk_time = self.protocol.preprocess_zk(
                state.n,
                state.kappa,
                &state.bases,
                server,
                &mut state.sk,
                &mut state.pk,
            );

            println!("ZK preprocess took {:?}", preprocess_zk_time);

            P::save_secret_key(&self.base_dir, &state.sk)?;

            println!("Session initialized and keys persisted to storage");
            Ok(())
        } else {
            state.sk = P::load_secret_key(
                &self.base_dir,
                LatticeParams {
                    n: state.n,
                    kappa: state.kappa,
                },
            )?;
            state.pk = load_pk(&self.base_dir, state.n)?;

            println!("Client fully initialized from files");

            Ok(())
        }
    }

    pub fn request(
        &self,
        server_tx: &SyncSender<ClientRequest>,
        scalars: &[blst_scalar],
    ) -> (Result<blst_p2, String>, std::time::Duration) {
        let state = self.state.as_ref().expect("Client not booted");

        let (msg, aux, delegate_time) =
            self.protocol
                .delegate(state.kappa, &state.bases, &state.sk, scalars);

        let (resp_tx, resp_rx) = channel();
        let start_send = Instant::now();
        let timeout = Duration::from_secs(3);

        println!("Sending request to server...");

        loop {
            match server_tx.try_send(ClientRequest::Compute(msg.clone(), resp_tx.clone())) {
                Ok(_) => break, // Successfully sent
                Err(TrySendError::Full(_)) => {
                    if start_send.elapsed() > timeout {
                        return (
                            Err("Send timeout: Server queue is full and not clearing".to_string()),
                            Duration::default(),
                        );
                    }
                    sleep(Duration::from_millis(100)); // Back off
                    continue;
                }
                Err(TrySendError::Disconnected(_)) => {
                    return (
                        Err("Server channel disconnected".to_string()),
                        Duration::default(),
                    );
                }
            }
        }
        let response = resp_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("Server took too long to respond");

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
