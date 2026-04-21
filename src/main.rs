use std::sync::mpsc::channel;

use blst::p2_affines;
use zk_delegated_msm::io::{init_level, save_bases, ClientRequest, CommStats};
use zk_delegated_msm::protocol::{DelegatedMsmProtocol, TdMsm, ToeplitzMsm};
use zk_delegated_msm::{generate_bases, generate_scalars, MsmClient, MsmServer};

pub trait ProtocolNew {
    fn new(kappa: usize, noise_rate: f64) -> Self;
}

impl ProtocolNew for TdMsm {
    fn new(kappa: usize, noise_rate: f64) -> Self {
        TdMsm::new(kappa, noise_rate)
    }
}

impl ProtocolNew for ToeplitzMsm {
    fn new(kappa: usize, noise_rate: f64) -> Self {
        ToeplitzMsm::new(kappa, noise_rate)
    }
}

static BASE_DIR: &str = "data";

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let use_td = args.iter().any(|a| a == "--td");

    let n = 1 << 18;
    let kappa = 1 << 8;

    if use_td {
        run_client::<TdMsm>(n, kappa)
    } else {
        run_client::<ToeplitzMsm>(n, kappa)
    }
}

fn run_client<P: DelegatedMsmProtocol + ProtocolNew>(
    n: usize,
    kappa: usize,
) -> std::io::Result<()> {
    if init_level::<P>(BASE_DIR) == 0 {
        println!("Generating global bases file");
        let raw_bases = generate_bases(n);
        save_bases(BASE_DIR, p2_affines::from(&raw_bases).as_slice())
            .expect("Failed to save global bases");
    } else {
        println!("Global bases found");
    }

    let mut client = MsmClient::new(P::new(kappa, 1f64 / (kappa as f64)), BASE_DIR);

    println!("Initializing client-server setup");
    client.init_client(n, kappa)?;

    let (server_tx, server_rx) = channel();
    let (ready_tx, ready_rx) = channel();
    let server_handle = std::thread::spawn(move || {
        let server = MsmServer::new(BASE_DIR);
        server.run(n, server_rx, ready_tx);
    });
    ready_rx.recv().expect("Server thread panicked");
    client.init_client_zk(&server_tx)?;

    let mut stats = CommStats::default();
    let x_scalars = generate_scalars(n);
    let (res, time) = client.request(&server_tx, &x_scalars, &mut stats);
    match res {
        Ok(_) => {
            println!("\nMSM Computation Successful!");
            println!("\n Total Time: {:?}", time);
            print_report(n, &stats);
        }
        Err(e) => {
            println!("\nProtocol Error: {}", e);
            println!("\n Total Time: {:?}", time);
        }
    }
    server_tx.send(ClientRequest::Shutdown).unwrap();
    server_handle.join().unwrap();
    Ok(())
}

fn print_report(n: usize, stats: &CommStats) {
    println!("--- Final Report ---");
    println!("Size (n):  {}", n);
    println!(
        "Sent:      {:.2} MB",
        stats.total_bytes_sent as f64 / 1_000_000.0
    );
    println!(
        "Received:  {:.2} KB",
        stats.total_bytes_received as f64 / 1_000.0
    );
}
