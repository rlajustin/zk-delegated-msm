use crate::{DelegatedMsmPk, DelegatedMsmSk, ZkParams};
use blst::{
    blst_fr, blst_fr_from_scalar, blst_lendian_from_scalar, blst_p2, blst_p2_affine,
    blst_p2_affine_serialize, blst_p2_deserialize, blst_p2_from_affine, blst_p2_serialize,
    blst_scalar, blst_scalar_from_fr, blst_scalar_from_lendian, p2_affines, BLST_ERROR,
};
use std::sync::mpsc::Sender;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::mem::size_of;
use std::path::Path;

pub enum ClientRequest {
    Compute(Vec<blst_scalar>, Sender<ServerResponse>), // We include a return path
    Shutdown,
}

pub struct ServerResponse {
    pub a: blst_p2,
    pub b: blst_p2,
}

// --- Pathing & Types ---

pub enum PathType {
    Bases,
    SecretKey,
    ZkSecretKey,
    PublicKey,
}

impl PathType {
    fn prefix(&self) -> &'static str {
        match self {
            PathType::Bases => "bases",
            PathType::SecretKey => "sk",
            PathType::ZkSecretKey => "zk_sk",
            PathType::PublicKey => "pk",
        }
    }
}

/// Centralized path helper
fn get_path(base_name: &str, path_type: PathType) -> String {
    format!("data/{}_{base_name}", path_type.prefix())
}

pub const BLST_P2_AFFINE_SIZE: usize = size_of::<blst_p2_affine>();
pub const BLST_SCALAR_SIZE: usize = size_of::<blst_scalar>();
pub const BLST_P2_SIZE: usize = size_of::<blst_p2>();

// --- Generic Serialization Helpers ---

fn write_scalar<W: Write>(mut w: W, s: &blst_scalar) -> std::io::Result<()> {
    let mut bytes = [0u8; BLST_SCALAR_SIZE];
    unsafe { blst_lendian_from_scalar(bytes.as_mut_ptr(), s) };
    w.write_all(&bytes)
}

fn read_scalar<R: Read>(mut r: R) -> std::io::Result<blst_scalar> {
    let mut bytes = [0u8; BLST_SCALAR_SIZE];
    r.read_exact(&mut bytes)?;
    let mut s = blst_scalar::default();
    unsafe { blst_scalar_from_lendian(&mut s, bytes.as_ptr()) };
    Ok(s)
}

/// Converts Fr to Scalar then writes 32 bytes to disk
fn write_fr<W: Write>(w: W, fr: &blst_fr) -> std::io::Result<()> {
    let mut s = blst_scalar::default();
    unsafe { blst_scalar_from_fr(&mut s, fr) };
    write_scalar(w, &s)
}

/// Reads 32 bytes from disk to Scalar then converts to Fr
fn read_fr<R: Read>(r: R) -> std::io::Result<blst_fr> {
    let s = read_scalar(r)?;
    let mut fr = blst_fr::default();
    unsafe { blst_fr_from_scalar(&mut fr, &s) };
    Ok(fr)
}

fn write_affines<W: Write>(mut w: W, points: &[blst_p2_affine]) -> std::io::Result<()> {
    let mut buf = [0u8; BLST_P2_AFFINE_SIZE];
    for p in points {
        unsafe { blst_p2_affine_serialize(buf.as_mut_ptr(), p) };
        w.write_all(&buf)?;
    }
    Ok(())
}

fn read_points<R: Read>(mut r: R, n: usize) -> std::io::Result<Vec<blst_p2>> {
    let mut points = Vec::with_capacity(n);
    let mut buf = [0u8; BLST_P2_AFFINE_SIZE];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        let mut affine = blst_p2_affine::default();
        if unsafe { blst_p2_deserialize(&mut affine, buf.as_ptr()) } != BLST_ERROR::BLST_SUCCESS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Point deserialization failed",
            ));
        }
        let mut proj = blst_p2::default();
        unsafe { blst_p2_from_affine(&mut proj, &affine) };
        points.push(proj);
    }
    Ok(points)
}

pub fn point_to_hex(point: &blst_p2) -> String {
    let mut bytes = [0u8; BLST_P2_SIZE];
    unsafe { blst_p2_serialize(bytes.as_mut_ptr(), point) };
    hex::encode(&bytes[0..4])
}

// --- Primary IO Logic ---

pub fn save_bases(base_path: &str, bases: &[blst_p2_affine]) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_path, PathType::Bases))?);
    file.write_all(&(bases.len() as u32).to_le_bytes())?;
    write_affines(file, bases)
}

pub fn load_bases_subset(base_path: &str, n: usize) -> std::io::Result<p2_affines> {
    let mut file = BufReader::new(File::open(get_path(base_path, PathType::Bases))?);
    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    let total = u32::from_le_bytes(count_bytes) as usize;

    if n > total {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Insufficient bases",
        ));
    }
    Ok(p2_affines::from(&read_points(file, n)?))
}

pub fn save_2g2t_sk(base_path: &str, sk: &DelegatedMsmSk) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_path, PathType::SecretKey))?);
    write_scalar(&mut file, &sk.r)?;
    file.write_all(&(sk.rho_super.len() as u32).to_le_bytes())?;
    for rho in &sk.rho_super {
        write_fr(&mut file, rho)?;
    }

    let mut q_bytes = [0u8; BLST_P2_SIZE];
    unsafe { blst_p2_serialize(q_bytes.as_mut_ptr(), &sk.q_point) };
    file.write_all(&q_bytes)
}

pub fn load_2g2t_sk(base_path: &str) -> std::io::Result<DelegatedMsmSk> {
    let mut file = BufReader::new(File::open(get_path(base_path, PathType::SecretKey))?);
    let r = read_scalar(&mut file)?;

    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let rho_len = u32::from_le_bytes(len_buf) as usize;
    let rho_super = (0..rho_len)
        .map(|_| read_fr(&mut file))
        .collect::<std::io::Result<Vec<_>>>()?;

    let mut q_bytes = [0u8; BLST_P2_AFFINE_SIZE];
    file.read_exact(&mut q_bytes)?;
    let mut q_affine = blst_p2_affine::default();
    if unsafe { blst_p2_deserialize(&mut q_affine, q_bytes.as_ptr()) } != BLST_ERROR::BLST_SUCCESS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "q_point failed",
        ));
    }
    let mut q_point = blst_p2::default();
    unsafe { blst_p2_from_affine(&mut q_point, &q_affine) };

    Ok(DelegatedMsmSk {
        r,
        rho_super,
        q_point,
        ..Default::default()
    })
}

pub fn save_zk_sk(base_path: &str, sk: &DelegatedMsmSk) -> std::io::Result<ZkParams> {
    let mt_p = sk.mt_p.as_ref().ok_or(std::io::ErrorKind::InvalidInput)?;
    let toeplitz = sk
        .m_matrix_toeplitz
        .as_ref()
        .ok_or(std::io::ErrorKind::InvalidInput)?;

    let mut file = BufWriter::new(File::create(get_path(base_path, PathType::ZkSecretKey))?);
    write_affines(&mut file, mt_p.as_slice())?;

    // Write toeplitz matrix directly from memory
    for scalar in toeplitz {
        write_scalar(&mut file, scalar)?;
    }

    write_scalar(&mut file, sk.ntt_fwd_root.as_ref().unwrap())?;
    write_scalar(&mut file, sk.ntt_inv_root.as_ref().unwrap())?;
    write_scalar(&mut file, sk.ntt_inv_n.as_ref().unwrap())?;

    Ok(ZkParams {
        n: (toeplitz.len() - mt_p.as_slice().len() + 1),
        kappa: mt_p.as_slice().len(),
    })
}

pub fn load_zk_sk(base_path: &str, params: ZkParams) -> std::io::Result<DelegatedMsmSk> {
    let mut sk = load_2g2t_sk(base_path)?;
    let mut file = BufReader::new(File::open(get_path(base_path, PathType::ZkSecretKey))?);

    sk.mt_p = Some(p2_affines::from(&read_points(&mut file, params.kappa)?));

    let mut toeplitz: Vec<blst_scalar> = vec![blst_scalar::default(); params.n + params.kappa - 1];
    for _ in 0..params.n + params.kappa - 1 {
        toeplitz.push(read_scalar(&mut file)?);
    }

    sk.m_matrix_toeplitz = Some(toeplitz);
    sk.ntt_fwd_root = Some(read_scalar(&mut file)?);
    sk.ntt_inv_root = Some(read_scalar(&mut file)?);
    sk.ntt_inv_n = Some(read_scalar(&mut file)?);

    Ok(sk)
}

pub fn save_pk(base_path: &str, pk: &DelegatedMsmPk) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_path, PathType::PublicKey))?);
    let bases = pk.t_bases.as_slice();
    file.write_all(&(bases.len() as u32).to_le_bytes())?;
    write_affines(file, bases)
}

pub fn load_pk(base_path: &str, n: usize) -> std::io::Result<DelegatedMsmPk> {
    let mut file = BufReader::new(File::open(get_path(base_path, PathType::PublicKey))?);
    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    Ok(DelegatedMsmPk {
        t_bases: p2_affines::from(&read_points(file, n)?),
    })
}

pub fn init_level(base_path: &str) -> u8 {
    if !Path::new(&get_path(base_path, PathType::SecretKey)).exists() {
        0
    } else if !Path::new(&get_path(base_path, PathType::ZkSecretKey)).exists() {
        1
    } else {
        2
    }
}
#[derive(Debug, Default, Clone)]
pub struct CommStats {
    pub total_bytes_sent: usize,
    pub total_bytes_received: usize,
}

impl CommStats {
    pub fn record_inbound_scalars(&mut self, scalar: &[blst_scalar]) {
        self.total_bytes_received += scalar.len() * BLST_SCALAR_SIZE;
    }
    pub fn record_outbound_scalars(&mut self, scalar: &[blst_scalar]) {
        self.total_bytes_sent += scalar.len() * BLST_SCALAR_SIZE;
    }
    pub fn record_inbound_points(&mut self, n: usize) {
        self.total_bytes_received += BLST_SCALAR_SIZE * n;
    }
    pub fn record_outbound_points(&mut self, n: usize) {
        self.total_bytes_sent += BLST_SCALAR_SIZE * n;
    }
}
