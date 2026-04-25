use crate::compute_msm;
use crate::io::{load_bases_subset, load_pk, ClientRequest, ServerResponse};
use blst::{blst_p2, blst_scalar, p2_affines};
use std::sync::mpsc::Receiver;

pub struct MsmServerState {
    pub bases: p2_affines,
    pub t_bases: p2_affines,
}

pub struct MsmServer {
    pub base_dir: String,
    pub state: Option<MsmServerState>,
}

impl MsmServer {
    pub fn new(base_dir: &str) -> Self {
        Self {
            base_dir: base_dir.to_string(),
            state: None,
        }
    }

    pub fn run(
        mut self,
        n: usize,
        rx: Receiver<ClientRequest>,
        ready_tx: std::sync::mpsc::Sender<()>,
    ) {
        self.boot(n);
        println!("Server: Booted and listening...");

        let _ = ready_tx.send(());

        while let Ok(msg) = rx.recv() {
            match msg {
                ClientRequest::Compute(z_scalars, tx_back) => {
                    println!("Server: received compute request");
                    let (a, b) = self.handle_request(z_scalars);
                    println!("Server: computed, sending response");
                    let _ = tx_back.send(ServerResponse { a, b });
                }
                ClientRequest::Shutdown => break,
            }
        }
        println!("Server: Shutting down.");
    }

    fn boot(&mut self, n: usize) {
        let bases =
            load_bases_subset(&self.base_dir, n).expect("Global bases file missing or too small.");

        let t_bases = load_pk(&self.base_dir, n)
            .expect("PK missing or too small")
            .t_bases;

        self.state = Some(MsmServerState { bases, t_bases })
    }

    pub fn handle_request(&self, z_scalars: Vec<blst_scalar>) -> (blst_p2, blst_p2) {
        let state = self.state.as_ref().expect("Server state missing");
        (
            compute_msm(&state.bases, &z_scalars),
            compute_msm(&state.t_bases, &z_scalars),
        )
    }
}
