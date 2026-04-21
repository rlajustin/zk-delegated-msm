use crate::protocol::{
    DelegatedMsmPk, LatticeParams, MsmBase, TdMsm, TdSk, ToeplitzMsm, ToeplitzSk,
};
use crate::DelegatedMsmProtocol;
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
    Compute(Vec<blst_scalar>, Sender<ServerResponse>),
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
    ProtocolSecretKey,
    PublicKey,
}

impl PathType {
    fn prefix(&self) -> &str {
        match self {
            PathType::Bases => "bases",
            PathType::SecretKey => "2g2t_sk",
            PathType::ProtocolSecretKey => "protocol_sk",
            PathType::PublicKey => "pk",
        }
    }
}

/// Centralized path helper
fn get_path(base_dir: &str, path_type: PathType) -> String {
    format!("{base_dir}/{}.bin", path_type.prefix())
}

fn get_path_protocol<P: DelegatedMsmProtocol>(base_dir: &str, path_type: PathType) -> String {
    format!(
        "{base_dir}/{}_{}.bin",
        P::protocol_name(),
        path_type.prefix(),
    )
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

fn write_fr<W: Write>(w: W, fr: &blst_fr) -> std::io::Result<()> {
    let mut s = blst_scalar::default();
    unsafe { blst_scalar_from_fr(&mut s, fr) };
    write_scalar(w, &s)
}

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

pub fn scalar_to_hex(scalar: &blst_scalar) -> String {
    let mut bytes = [0u8; BLST_SCALAR_SIZE];
    unsafe { blst_lendian_from_scalar(bytes.as_mut_ptr(), scalar) };
    hex::encode(&bytes[0..4])
}

// --- File Logic ---

pub fn save_bases(base_dir: &str, bases: &[blst_p2_affine]) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_dir, PathType::Bases))?);
    file.write_all(&(bases.len() as u32).to_le_bytes())?;
    write_affines(file, bases)
}

pub fn load_bases_subset(base_dir: &str, n: usize) -> std::io::Result<p2_affines> {
    let mut file = BufReader::new(File::open(get_path(base_dir, PathType::Bases))?);
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

pub fn save_2g2t_sk(base_dir: &str, base: &MsmBase) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_dir, PathType::SecretKey))?);
    write_scalar(&mut file, &base.r)?;
    file.write_all(&(base.rho_super.len() as u32).to_le_bytes())?;
    for rho in &base.rho_super {
        write_fr(&mut file, rho)?;
    }

    let mut q_bytes = [0u8; BLST_P2_SIZE];
    unsafe { blst_p2_serialize(q_bytes.as_mut_ptr(), &base.q_point) };
    file.write_all(&q_bytes)
}

pub fn load_2g2t_sk(base_dir: &str) -> std::io::Result<MsmBase> {
    let mut file = BufReader::new(File::open(get_path(base_dir, PathType::SecretKey))?);
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

    Ok(MsmBase {
        r,
        rho_super,
        q_point,
    })
}

pub fn save_toeplitz_sk(base_dir: &str, sk: &ToeplitzSk) -> std::io::Result<()> {
    let mt_p = sk.mt_p.as_ref().ok_or(std::io::ErrorKind::InvalidInput)?;
    let toeplitz = sk
        .m_matrix_toeplitz
        .as_ref()
        .ok_or(std::io::ErrorKind::InvalidInput)?;

    let mut file = BufWriter::new(File::create(get_path_protocol::<ToeplitzMsm>(
        base_dir,
        PathType::ProtocolSecretKey,
    ))?);
    write_affines(&mut file, mt_p.as_slice())?;

    // Write toeplitz matrix directly from memory
    for scalar in toeplitz {
        write_scalar(&mut file, scalar)?;
    }

    write_scalar(&mut file, sk.ntt_fwd_root.as_ref().unwrap())?;
    write_scalar(&mut file, sk.ntt_inv_root.as_ref().unwrap())?;
    write_scalar(&mut file, sk.ntt_inv_n.as_ref().unwrap())?;

    Ok(())
}

pub fn load_toeplitz_sk(base_dir: &str, params: LatticeParams) -> std::io::Result<ToeplitzSk> {
    let base = load_2g2t_sk(base_dir)?;
    let mut file = BufReader::new(File::open(get_path_protocol::<ToeplitzMsm>(
        base_dir,
        PathType::ProtocolSecretKey,
    ))?);

    let mt_p = Some(p2_affines::from(&read_points(&mut file, params.kappa)?));

    let mut toeplitz: Vec<blst_scalar> = vec![blst_scalar::default(); params.n + params.kappa - 1];
    for s in toeplitz.iter_mut() {
        *s = read_scalar(&mut file)?;
    }

    Ok(ToeplitzSk {
        base,
        mt_p,
        m_matrix_toeplitz: Some(toeplitz),
        ntt_fwd_root: Some(read_scalar(&mut file)?),
        ntt_inv_root: Some(read_scalar(&mut file)?),
        ntt_inv_n: Some(read_scalar(&mut file)?),
    })
}

pub fn save_td_sk(base_dir: &str, sk: &TdSk) -> std::io::Result<()> {
    let mt_p = sk.mt_p.as_ref().ok_or(std::io::ErrorKind::InvalidInput)?;
    let trapdoor_matrix = sk
        .trapdoor_matrix
        .as_ref()
        .ok_or(std::io::ErrorKind::InvalidInput)?;
    let seed = sk.trapdoor_seed.ok_or(std::io::ErrorKind::InvalidInput)?;

    let mut file = BufWriter::new(File::create(get_path_protocol::<TdMsm>(
        base_dir,
        PathType::ProtocolSecretKey,
    ))?);
    write_affines(&mut file, mt_p.as_slice())?;

    for scalar in trapdoor_matrix {
        write_scalar(&mut file, scalar)?;
    }

    file.write_all(&seed.to_le_bytes())
}

pub fn load_td_sk(base_dir: &str, params: LatticeParams) -> std::io::Result<TdSk> {
    let base = load_2g2t_sk(base_dir)?;
    let mut file = BufReader::new(File::open(get_path_protocol::<TdMsm>(
        base_dir,
        PathType::ProtocolSecretKey,
    ))?);

    let mt_p = Some(p2_affines::from(&read_points(&mut file, params.kappa)?));

    let trapdoor_len = params.n * params.kappa;
    let mut trapdoor: Vec<blst_scalar> = vec![blst_scalar::default(); trapdoor_len];
    for s in trapdoor.iter_mut() {
        *s = read_scalar(&mut file)?;
    }

    let mut seed_buf = [0u8; 4];
    file.read_exact(&mut seed_buf)?;

    Ok(TdSk {
        base,
        mt_p,
        trapdoor_matrix: Some(trapdoor),
        trapdoor_seed: Some(u32::from_le_bytes(seed_buf)),
    })
}

pub fn save_pk(base_dir: &str, pk: &DelegatedMsmPk) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(get_path(base_dir, PathType::PublicKey))?);
    let bases = pk.t_bases.as_slice();
    file.write_all(&(bases.len() as u32).to_le_bytes())?;
    write_affines(file, bases)
}

pub fn load_pk(base_dir: &str, n: usize) -> std::io::Result<DelegatedMsmPk> {
    let mut file = BufReader::new(File::open(get_path(base_dir, PathType::PublicKey))?);
    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    Ok(DelegatedMsmPk {
        t_bases: p2_affines::from(&read_points(file, n)?),
    })
}

// Helpers

pub fn init_level<P: DelegatedMsmProtocol>(base_dir: &str) -> u8 {
    if !Path::new(&get_path(base_dir, PathType::Bases)).exists() {
        0
    } else if !Path::new(&get_path_protocol::<P>(
        base_dir,
        PathType::ProtocolSecretKey,
    ))
    .exists()
    {
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
