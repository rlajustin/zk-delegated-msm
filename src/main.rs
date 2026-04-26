use std::sync::mpsc::{channel, sync_channel};

use blst::p2_affines;
use zk_delegated_msm::io::{
    init_level, load_bases_subset, point_to_hex, save_bases, ClientRequest, CommStats,
};
use zk_delegated_msm::protocol::{DelegatedMsmProtocol, TdMsm, ToeplitzMsm};
use zk_delegated_msm::timer::Timer;
use zk_delegated_msm::{compute_msm, generate_bases, generate_scalars, MsmClient, MsmServer};

use std::collections::HashMap;
use std::io::Write;

static DEFAULT_KAPPA: usize = 1 << 9;
static ITERATIONS: usize = 10;

use cap::Cap;
use std::alloc::System;

fn get_base_dir(protocol: &str, n: usize, kappa: usize) -> String {
    let dir = format!("data_{}_{}_{}", protocol, n, kappa);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

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

fn main() -> std::io::Result<()> {
    ALLOCATOR.set_limit(2 * 1024 * 1024 * 1024).unwrap();
    let args: Vec<String> = std::env::args().collect();
    let use_td = args.iter().any(|a| a == "--td");

    println!("Running benchmark for n=2^16..=20, kappa=512");
    println!("{}", if use_td { "Trapdoor" } else { "Toeplitz" });
    println!();

    if use_td {
        run_benchmark::<TdMsm>("td")
    } else {
        run_benchmark::<ToeplitzMsm>("toeplitz")
    }
}

fn run_benchmark<P: DelegatedMsmProtocol + ProtocolNew>(protocol: &str) -> std::io::Result<()> {
    let mut results: Vec<HashMap<String, String>> = Vec::new();

    for n_exp in 16..=20 {
        let n = 1 << n_exp;
        let base_dir = get_base_dir(protocol, n, DEFAULT_KAPPA);
        println!("=== n = {} (2^{}) ===", n, n_exp);

        let mut result: HashMap<String, String> = HashMap::new();
        result.insert("n".to_string(), n.to_string());
        result.insert("n_exp".to_string(), n_exp.to_string());
        result.insert("kappa".to_string(), DEFAULT_KAPPA.to_string());

        if init_level::<P>(&base_dir) == 0 {
            println!("Generating global bases file");
            let raw_bases = generate_bases(n);
            save_bases(&base_dir, p2_affines::from(&raw_bases).as_slice())
                .expect("Failed to save global bases");
        }

        let mut client = MsmClient::new(
            P::new(DEFAULT_KAPPA, 1f64 / (DEFAULT_KAPPA as f64)),
            &base_dir,
        );

        let preprocess_timer = Timer::new();
        println!("Initializing client...");
        client.init_client(n, DEFAULT_KAPPA)?;
        let preprocess_time = preprocess_timer.elapsed();
        println!("Preprocess: {:?}", preprocess_time);
        result.insert(
            "preprocess_ms".to_string(),
            format!("{}", preprocess_time.as_millis()),
        );

        let (server_tx, server_rx) = sync_channel(1);
        let (ready_tx, ready_rx) = channel();
        let server_handle = std::thread::spawn({
            let base_dir = base_dir.clone();
            move || {
                let server = MsmServer::new(&base_dir);
                server.run(n, server_rx, ready_tx);
            }
        });
        ready_rx.recv().expect("Server thread panicked");

        let preprocess_zk_timer = Timer::new();
        client.init_client_zk(&server_tx)?;
        let preprocess_zk_time = preprocess_zk_timer.elapsed();
        println!("Preprocess ZK: {:?}", preprocess_zk_time);
        result.insert(
            "preprocess_zk_ms".to_string(),
            format!("{}", preprocess_zk_time.as_millis()),
        );

        let bases = load_bases_subset(&base_dir, n)?;

        let mut pippenger_times: Vec<u128> = Vec::new();
        let mut request_times: Vec<u128> = Vec::new();
        let mut verified_count: usize = 0;

        for i in 0..ITERATIONS {
            println!("--- iteration {}/{} ---", i + 1, ITERATIONS);

            let x_scalars = generate_scalars(n);
            println!("  scalars generated");

            let pippenger_timer = Timer::new();
            let expected = compute_msm(&bases, &x_scalars);
            let pippenger_time = pippenger_timer.elapsed();
            println!("  Pippenger MSM: {:?}", pippenger_time);
            println!("  Expected: {}", point_to_hex(&expected));
            pippenger_times.push(pippenger_time.as_millis());

            println!("  sending request...");
            let (res, request_time) = client.request(&server_tx, &x_scalars);

            match res {
                Ok(_) => {
                    println!("Protocol: OK");
                    println!("Total Request Time: {:?}", request_time);
                    request_times.push(request_time.as_millis());
                    verified_count += 1;
                }
                Err(e) => {
                    println!("Protocol Error: {}", e);
                }
            }
        }

        let avg_pippenger: u64 = pippenger_times.iter().sum::<u128>() as u64 / ITERATIONS as u64;
        let avg_request: u64 = if request_times.is_empty() {
            0
        } else {
            request_times.iter().sum::<u128>() as u64 / request_times.len() as u64
        };
        println!("\nAverages over {} iterations:", ITERATIONS);
        println!("  Pippenger: {} ms", avg_pippenger);
        println!("  Request: {} ms", avg_request);
        println!("  Verified: {}/{}", verified_count, ITERATIONS);

        result.insert("pippenger_ms".to_string(), avg_pippenger.to_string());
        result.insert("request_ms".to_string(), format!("{}", avg_request));
        result.insert(
            "verified".to_string(),
            format!("{}/{}", verified_count, ITERATIONS),
        );
        // result.insert(
        //     "sent_mb".to_string(),
        //     format!("{:.2}", total_sent as f64 / 1_000_000.0 / ITERATIONS as f64),
        // );
        // result.insert(
        //     "received_kb".to_string(),
        //     format!("{:.2}", total_recv as f64 / 1_000.0 / ITERATIONS as f64),
        // );

        println!();
        results.push(result);

        server_tx
            .send(ClientRequest::Shutdown)
            .map_err(|e| format!("Failed to send to server: {}", e))
            .unwrap();
        server_handle.join().unwrap();
    }

    println!("\n=== Summary ===");
    println!(
        "{:>6} {:>8} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "n", "kappa", "preproc", "preproc_zk", "pippenger", "request", "verified"
    );
    println!(
        "{:->6} {:->8} {:->10} {:->12} {:->12} {:->10} {:->10}",
        "", "", "", "", "", "", ""
    );
    for r in &results {
        println!(
            "{:>6} {:>8} {:>10} {:>12} {:>12} {:>10} {:>10}",
            r.get("n_exp").unwrap_or(&"".to_string()),
            r.get("kappa").unwrap_or(&"".to_string()),
            r.get("preprocess_ms").unwrap_or(&"".to_string()),
            r.get("preprocess_zk_ms").unwrap_or(&"".to_string()),
            r.get("pippenger_ms").unwrap_or(&"".to_string()),
            r.get("request_ms").unwrap_or(&"".to_string()),
            r.get("verified").unwrap_or(&"?".to_string())
        );
    }

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let log_filename = format!("{}_{}.csv", protocol, timestamp);
    let mut log_file = std::fs::File::create(&log_filename)?;
    writeln!(log_file, "n_exp,n,kappa,preprocess_ms,preprocess_zk_ms,pippenger_ms,request_ms,verified,sent_mb,received_kb")?;
    for r in &results {
        writeln!(
            log_file,
            "{},{},{},{},{},{},{},{},{},{}",
            r.get("n_exp").cloned().unwrap_or_default(),
            r.get("n").cloned().unwrap_or_default(),
            r.get("kappa").cloned().unwrap_or_default(),
            r.get("preprocess_ms").cloned().unwrap_or_default(),
            r.get("preprocess_zk_ms").cloned().unwrap_or_default(),
            r.get("pippenger_ms").cloned().unwrap_or_default(),
            r.get("request_ms").cloned().unwrap_or_default(),
            r.get("verified")
                .cloned()
                .unwrap_or_else(|| "?".to_string()),
            r.get("sent_mb").cloned().unwrap_or_default(),
            r.get("received_kb").cloned().unwrap_or_default()
        )?;
    }
    println!("\nResults saved to: {}", log_filename);

    Ok(())
}
