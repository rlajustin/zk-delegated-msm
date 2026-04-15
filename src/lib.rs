pub mod khabbazian2g2t;
pub mod zk2g2t;

use rand::RngCore;
use rayon::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixType {
    Random = 0,
    Toeplitz = 1,
}

pub static USE_PARALLELISM: bool = false;

extern "C" {
    pub fn precompute_rho_super(ret: *mut u8, rho: *const u8);
    pub fn blst_fr_inner_product_fast(ret: *mut u8, x: *const u8, rho_super: *const u8, len: usize);
    pub fn compute_lpn_mask(
        out: *mut u8,
        err: *const u8,
        x: *const u8,
        matrix_A: *const u8,
        s: *const u8,
        n: usize,
        kappa: usize,
        mat_type: MatrixType,
    );
    pub fn sample_errors_and_affines_c(
        err_bytes_out: *mut u8,
        dense_err_scalars_out: *mut u8,
        dense_err_affines_out: *mut blst_p2_affine,
        bases_in: *const blst_p2_affine,
        n: usize,
        noise_rate: f64,
        seed: u32,
    ) -> usize;
    pub fn compute_lpn_toeplitz_ntt_c(
        z: *mut u8,
        err: *const u8,
        x: *const u8,
        toeplitz_vec: *const u8,
        fwd_roots: *const u8,
        inv_roots: *const u8,
        inv_N: *const u8,
        n: usize,
        kappa: usize,
        log_n: usize,
    );
}

pub use blst::*;

pub trait DelegatedMsmProtocol<'a, G> {
    type SecretKey;
    type PublicKey;
    type Message: 'a;
    type Auxiliary;
    type Proof;

    fn preprocess(&self, n: usize, bases: &p2_affines) -> (Self::SecretKey, Self::PublicKey);
    fn delegate(
        &self,
        bases: &p2_affines,
        sk: &Self::SecretKey,
        scalars: &'a [u8],
    ) -> (Self::Message, Self::Auxiliary);
    fn compute(
        &self,
        bases: &p2_affines,
        pk: &Self::PublicKey,
        message: &Self::Message,
    ) -> Self::Proof;
    fn postprocess(
        &self,
        sk: &Self::SecretKey,
        aux: &Self::Auxiliary,
        proof: Self::Proof,
    ) -> Result<G, ()>;
    fn protocol_name() -> &'static str;

    type ZkSecretKey: 'static;
    type ZkPublicKey: 'static;
    type ZkMessage: 'a;
    type ZkAuxiliary: 'static;
    type ZkProof: 'static;

    fn preprocess_zk(
        &self,
        n: usize,
        bases: &p2_affines,
        sk: &Self::SecretKey,
    ) -> (Self::ZkSecretKey, Self::ZkPublicKey);

    fn delegate_zk(
        &self,
        bases: &p2_affines,
        soundness_sk: &Self::SecretKey,
        zk_sk: &Self::ZkSecretKey,
        scalars: &'a [u8],
    ) -> (Self::ZkMessage, Self::ZkAuxiliary, Self::ZkProof);

    fn verify_zk(
        &self,
        soundness_sk: &Self::SecretKey,
        zk_sk: &Self::ZkSecretKey,
        aux: &Self::ZkAuxiliary,
        proof: Self::ZkProof,
    ) -> Result<G, ()>;

    fn supports_zk_delegation() -> bool {
        false
    }
}

pub struct TestHarness {
    verbose: bool,
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl TestHarness {
    pub fn new() -> Self {
        Self { verbose: false }
    }
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    pub fn run_suite<P>(&self, protocol: &P, bases: &[blst_p2], sizes: &[usize], num_trials: usize)
    where
        for<'a> P: DelegatedMsmProtocol<'a, blst_p2>,
    {
        for &n in sizes {
            println!("\n--- Testing n={} (2^{:.1}) ---", n, (n as f64).log2());
            self.benchmark_protocol(protocol, bases, n, num_trials);
        }
    }

    fn benchmark_protocol<P>(&self, protocol: &P, p_bases: &[blst_p2], n: usize, num_trials: usize)
    where
        for<'a> P: DelegatedMsmProtocol<'a, blst_p2>,
    {
        let mut scalar_sets = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            scalar_sets.push(Self::generate_scalars(n));
        }
        let bases = p2_affines::from(&p_bases[..n]);

        let pippenger_time = Self::benchmark_pippenger(&bases, &scalar_sets);
        println!(
            "  Pippenger:       {:?} (avg over {} trials)",
            pippenger_time, num_trials
        );

        let (preprocess_time, sk, pk) = {
            let start = std::time::Instant::now();
            let result = protocol.preprocess(n, &bases);
            (start.elapsed(), result.0, result.1)
        };
        println!(
            "  Preprocess:      {:?} ({:.1}x slower)",
            preprocess_time,
            preprocess_time.as_secs_f64() / pippenger_time.as_secs_f64()
        );

        let mut delegate_times = Vec::with_capacity(num_trials);
        let mut compute_times = Vec::with_capacity(num_trials);
        let mut postprocess_times = Vec::with_capacity(num_trials);
        let mut verify_success = 0usize;

        for scalars in scalar_sets.iter() {
            let start = std::time::Instant::now();
            let (message, aux) = protocol.delegate(&bases, &sk, scalars);
            delegate_times.push(start.elapsed());

            let start = std::time::Instant::now();
            let proof = protocol.compute(&bases, &pk, &message);
            compute_times.push(start.elapsed());

            let start = std::time::Instant::now();
            if protocol.postprocess(&sk, &aux, proof).is_ok() {
                verify_success += 1
            }
            postprocess_times.push(start.elapsed());
        }

        println!("  Delegate:        {:?}", Self::avg_time(&delegate_times));
        println!("  Compute:         {:?}", Self::avg_time(&compute_times));
        println!(
            "  Postprocess:     {:?}",
            Self::avg_time(&postprocess_times)
        );
        println!(
            "  Verification:    {}/{} successful",
            verify_success, num_trials
        );

        let total_client_time: std::time::Duration = delegate_times
            .iter()
            .zip(postprocess_times.iter())
            .map(|(d, p)| *d + *p)
            .sum();
        let avg_client = total_client_time / num_trials as u32;
        println!(
            "  Total client:    {:?} ({:.1}x speedup)",
            avg_client,
            pippenger_time.as_secs_f64() / avg_client.as_secs_f64()
        );
    }

    pub fn generate_bases(n: usize) -> Vec<blst_p2> {
        let mut bases = Vec::with_capacity(n);
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            let mut base_scalar = [0u8; 32];
            rng.fill_bytes(&mut base_scalar);
            base_scalar[31] &= 0x7F;
            let mut point = blst_p2::default();
            unsafe {
                blst_p2_mult(&mut point, blst_p2_generator(), base_scalar.as_ptr(), 255);
            }
            bases.push(point);
        }
        bases
    }

    pub fn generate_scalars(n: usize) -> Vec<u8> {
        let mut scalars = vec![0u8; n * 32];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut scalars);
        for i in 0..n {
            scalars[i * 32 + 31] &= 0x7F;
        }
        scalars
    }

    pub fn benchmark_pippenger(bases: &p2_affines, scalar_sets: &[Vec<u8>]) -> std::time::Duration {
        let mut times = Vec::with_capacity(scalar_sets.len());
        for scalars in scalar_sets {
            let start = std::time::Instant::now();
            let _ = bases.mult(scalars, 255);
            times.push(start.elapsed());
        }
        Self::avg_time(&times)
    }

    fn avg_time(times: &[std::time::Duration]) -> std::time::Duration {
        if times.is_empty() {
            return std::time::Duration::ZERO;
        }
        times.iter().sum::<std::time::Duration>() / times.len() as u32
    }
}

pub fn random_scalar() -> blst_scalar {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes[31] &= 0x3F; // Ensure within range for bls12-381
    let mut scalar = blst_scalar::default();
    unsafe {
        blst_scalar_from_lendian(&mut scalar, bytes.as_ptr());
    }
    scalar
}

pub fn fast_inner_product_safe(x_bytes: &[u8], rho_super_bytes: &[u8], n: usize) -> blst_scalar {
    let mut out_scalar = blst_scalar::default();
    unsafe {
        blst_fr_inner_product_fast(
            &mut out_scalar as *mut _ as *mut u8,
            x_bytes.as_ptr(),
            rho_super_bytes.as_ptr(),
            n,
        );
    }
    out_scalar
}

pub fn compute_msm(bases: &p2_affines, scalar_bytes: &[u8]) -> blst_p2 {
    bases.mult(scalar_bytes, 255)
}

pub fn preprocess_2g2t_logic(
    bases: &p2_affines,
    n: usize,
    r: &blst_scalar,
) -> (Vec<u8>, blst_p2, Vec<blst_p2>) {
    let mut rho_standard = Vec::with_capacity(n);
    let mut rho_bytes = vec![0u8; n * 32];
    let mut t_bases = Vec::with_capacity(n);

    for i in 0..n {
        let rho_i = random_scalar();
        rho_standard.push(rho_i);
        unsafe {
            precompute_rho_super(rho_bytes[i * 32..].as_mut_ptr(), rho_standard[i].b.as_ptr());
        }
    }

    let mut q_point = blst_p2::default();
    let q_scalar = random_scalar();
    unsafe {
        blst_p2_mult(&mut q_point, blst_p2_generator(), q_scalar.b.as_ptr(), 255);
    }

    for i in 0..n {
        let mut rho_q = blst_p2::default();
        let mut r_p = blst_p2::default();
        let mut base_p = blst_p2::default();
        unsafe {
            blst_p2_from_affine(&mut base_p, &bases[i]);
            blst_p2_mult(&mut rho_q, &q_point, rho_standard[i].b.as_ptr(), 255);
            blst_p2_mult(&mut r_p, &base_p, r.b.as_ptr(), 255);
            let mut t_i = blst_p2::default();
            blst_p2_add_or_double(&mut t_i, &r_p, &rho_q);
            t_bases.push(t_i);
        }
    }

    (rho_bytes, q_point, t_bases)
}

pub fn compute_dense_mt_p(
    matrix_vec: &[blst_scalar],
    bases: &p2_affines,
    n: usize,
    kappa: usize,
    use_parallelism: bool,
) -> Vec<blst_p2> {
    let process_column = |k: usize| -> blst_p2 {
        let mut column_scalars = vec![0u8; n * 32];
        for i in 0..n {
            let scalar = &matrix_vec[i * kappa + k];
            column_scalars[i * 32..(i + 1) * 32].copy_from_slice(&scalar.b);
        }
        bases.mult(&column_scalars, 255)
    };

    if use_parallelism {
        (0..kappa).into_par_iter().map(process_column).collect()
    } else {
        (0..kappa).map(process_column).collect()
    }
}

pub fn compute_toeplitz_mt_p(
    toeplitz_vec: &[blst_scalar],
    bases: &p2_affines,
    n: usize,
    kappa: usize,
    use_parallelism: bool,
) -> Vec<blst_p2> {
    let mut all_scalars_bytes = vec![0u8; (n + kappa - 1) * 32];
    for (i, scalar) in toeplitz_vec.iter().enumerate() {
        all_scalars_bytes[i * 32..(i + 1) * 32].copy_from_slice(&scalar.b);
    }

    let process_column = |k: usize| {
        let column_scalars = &all_scalars_bytes[k * 32..(k + n) * 32];
        bases.mult(column_scalars, 255)
    };

    if use_parallelism {
        (0..kappa).into_par_iter().map(process_column).collect()
    } else {
        (0..kappa).map(process_column).collect()
    }
}
