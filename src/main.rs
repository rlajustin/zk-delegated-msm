use std::io::{Read, Write};
use zk_delegated_msm::{blst_p2, MatrixType, TestHarness};

static KAPPA: usize = 256;
static ERROR_RATE: f64 = 0.01;
static BASES_FILE: &str = "bases.bin";

fn save_bases(bases: &[blst_p2], path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let count = bases.len() as u32;
    file.write_all(&count.to_le_bytes())?;
    for base in bases {
        // Transmuting to bytes for raw storage; blst_p2 is 288 bytes (uncompressed)
        let bytes: [u8; 288] = unsafe { std::mem::transmute(*base) };
        file.write_all(&bytes)?;
    }
    Ok(())
}

fn load_bases(path: &str) -> std::io::Result<Vec<blst_p2>> {
    let mut file = std::fs::File::open(path)?;
    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;
    let mut bases = Vec::with_capacity(count);
    let mut buf = [0u8; 288];
    for _ in 0..count {
        file.read_exact(&mut buf)?;
        let point: blst_p2 = unsafe { std::mem::transmute(buf) };
        bases.push(point);
    }
    Ok(bases)
}

fn main() {
    println!("\n{}\n", "*".repeat(80));
    println!("TESTING DELEGATED MSM\n");
    println!("{}\n", "*".repeat(80));

    let bases = match load_bases(BASES_FILE) {
        Ok(bases) => {
            println!("Loaded {} bases from {}", bases.len(), BASES_FILE);
            bases
        }
        Err(_) => {
            println!("Generating bases...");
            let bases = TestHarness::generate_bases(1 << 18);
            println!("Saving bases to {}", BASES_FILE);
            save_bases(&bases, BASES_FILE).expect("failed to save bases");
            bases
        }
    };

    let sizes: Vec<usize> = (10..=18).map(|exp| 1 << exp).collect();
    println!("Sizes: {:?}", sizes);

    let harness = TestHarness::new().verbose(true);
    println!("\n{}", "=".repeat(80));
    println!(
        "TEST: ZK-LPN Protocol (kappa={:}, error_rate={:}, type=Toeplitz)",
        KAPPA, ERROR_RATE
    );
    println!("{}", "=".repeat(80));

    let protocol =
        zk_delegated_msm::zk2g2t::ZkDelegatedMsm::new(KAPPA, ERROR_RATE, MatrixType::Toeplitz);

    harness.run_suite(&protocol, &bases, &sizes, 3);

    println!("\n{}", "*".repeat(80));
    println!("TESTING COMPLETE");
    println!("{}", "*".repeat(80));
}
