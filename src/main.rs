use blst::p2_affines;
use std::sync::mpsc::channel;

use zk_delegated_msm::io::{init_level, save_bases, ClientRequest, CommStats};
use zk_delegated_msm::{generate_bases, generate_scalars, MsmClient, MsmServer, ZkDelegatedMsm};

static BASE_PATH: &str = "data.bin";

fn main() -> std::io::Result<()> {
    let n = 1 << 18;
    let kappa = 1 << 8; // 256
    if init_level(BASE_PATH) == 0 {
        println!("Generating global bases file");
        let raw_bases = generate_bases(n);
        save_bases(
            &BASE_PATH.to_string(),
            p2_affines::from(&raw_bases).as_slice(),
        )
        .expect("Failed to save global bases");
    } else {
        println!("Global bases found");
    }

    println!("Initializing client-server setup");
    let protocol = ZkDelegatedMsm::new(256, 1f64 / (kappa as f64));
    let mut client = MsmClient::new(protocol, BASE_PATH);
    client.init_client(n, kappa)?;

    let (server_tx, server_rx) = channel();

    let (ready_tx, ready_rx) = channel();
    let server_handle = std::thread::spawn(move || {
        let server = MsmServer::new(BASE_PATH);
        server.run(n, server_rx, ready_tx);
    });

    println!("Main: Waiting for server to initialize...");
    ready_rx
        .recv()
        .expect("Server thread panicked or dropped before booting");
    println!("Main: Server is up! Starting ZK initialization.");
    client.init_client_zk(&server_tx)?;

    let mut stats = CommStats::default();
    println!("Testing client request");
    let x_scalars = generate_scalars(n);

    // This is the simplified API you wanted:
    let (res, time) = client.request(&server_tx, &x_scalars, &mut stats);
    match res {
        Ok(_result) => {
            println!("\n✅ MSM Computation Successful!");
            println!("\n Total Time: {:?}", time);
            print_report(n, &stats);
        }
        Err(e) => {
            println!("\n❌ Protocol Error: {}", e);
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
