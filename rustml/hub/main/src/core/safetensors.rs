//! SafeTensors format loader
//!
//! SafeTensors is a simple and safe format for storing tensors.
//! Format specification: https://github.com/huggingface/safetensors
//!
//! Provides two loading modes:
//! - `SafeTensorLoader::load()`: reads all data into memory, converts to F32
//! - `load_safetensors_mmap()`: zero-copy mmap, preserves original dtype

use crate::api::error::HubResult;
use rustml_core::{DType, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Errors specific to SafeTensors loading
#[derive(Error, Debug)]
pub enum SafeTensorsError {
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Data corruption: {0}")]
    DataCorruption(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// SafeTensors file loader (reads all data to F32)
#[derive(Debug, Default)]
pub struct SafeTensorLoader {
    /// Whether to convert all tensors to f32
    convert_to_f32: bool,
}

impl SafeTensorLoader {
    /// Create a new SafeTensor loader
    pub fn new() -> Self {
        Self { convert_to_f32: true }
    }

    /// Load tensors from a SafeTensors file (converts all to F32)
    pub fn load(&self, path: &Path) -> HubResult<HashMap<String, Tensor>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        self.load_from_reader(&mut reader)
    }

    /// Load tensors from a reader
    fn load_from_reader<R: Read + Seek>(&self, reader: &mut R) -> HubResult<HashMap<String, Tensor>> {
        // Read header size (8 bytes, little-endian u64)
        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_size];
        reader.read_exact(&mut header_bytes)?;
        let header: serde_json::Value = serde_json::from_slice(&header_bytes)
            .map_err(SafeTensorsError::from)?;

        // Data starts after header
        let data_start = 8 + header_size;

        let mut tensors = HashMap::new();

        // Parse tensor metadata from header
        if let serde_json::Value::Object(map) = header {
            for (name, info) in map {
                if name == "__metadata__" {
                    continue;
                }
                let tensor = self.load_tensor(reader, &info, data_start)?;
                tensors.insert(name, tensor);
            }
        }

        Ok(tensors)
    }

    /// Load a single tensor from the file
    fn load_tensor<R: Read + Seek>(
        &self,
        reader: &mut R,
        info: &serde_json::Value,
        data_start: usize,
    ) -> HubResult<Tensor> {
        let dtype = info["dtype"]
            .as_str()
            .ok_or_else(|| SafeTensorsError::InvalidHeader("Missing dtype".into()))?;

        let shape: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| SafeTensorsError::InvalidHeader("Missing shape".into()))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let data_offsets = info["data_offsets"]
            .as_array()
            .ok_or_else(|| SafeTensorsError::InvalidHeader("Missing data_offsets".into()))?;

        let start = data_offsets[0].as_u64().unwrap_or(0) as usize;
        let end = data_offsets[1].as_u64().unwrap_or(0) as usize;
        let byte_size = end - start;

        // Seek to tensor data
        reader.seek(SeekFrom::Start((data_start + start) as u64))?;

        // Read raw bytes
        let mut bytes = vec![0u8; byte_size];
        reader.read_exact(&mut bytes)?;

        // Convert to f32 based on dtype
        let data = self.bytes_to_f32(&bytes, dtype)?;

        Tensor::from_vec(data, shape).map_err(|e| e.into())
    }

    /// Convert raw bytes to f32 based on dtype
    fn bytes_to_f32(&self, bytes: &[u8], dtype: &str) -> Result<Vec<f32>, SafeTensorsError> {
        match dtype {
            "F32" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(data)
            }
            "F16" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let half_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half_to_f32(half_bits)
                    })
                    .collect();
                Ok(data)
            }
            "BF16" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16_to_f32(bf16_bits)
                    })
                    .collect();
                Ok(data)
            }
            "F64" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();
                Ok(data)
            }
            "I32" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        val as f32
                    })
                    .collect();
                Ok(data)
            }
            "I64" => {
                let data: Vec<f32> = bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();
                Ok(data)
            }
            _ => Err(SafeTensorsError::UnsupportedDtype(dtype.to_string())),
        }
    }
}

/// Load SafeTensors with zero-copy mmap, preserving original dtype.
///
/// This is the preferred loading method for large models as it:
/// - Avoids allocating memory for the entire file
/// - Returns tensors backed by the memory-mapped file
/// - Preserves F16/BF16 dtypes (convert with `tensor.to_f32()` when needed)
pub fn load_safetensors_mmap(path: &Path) -> HubResult<HashMap<String, Tensor>> {
    let file = File::open(path)?;
    // SAFETY: The file is opened read-only and lives for the duration of the
    // returned tensors (via Arc<Mmap>). Memory-mapped reads are safe as long as
    // no external process truncates/modifies the file while mapped.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mmap_arc = Arc::new(mmap);

    let st = safetensors::SafeTensors::deserialize(mmap_arc.as_ref())
        .map_err(|e| SafeTensorsError::InvalidHeader(e.to_string()))?;

    let mut tensors = HashMap::new();

    for (name, view) in st.tensors() {
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F16 => DType::F16,
            other => {
                return Err(SafeTensorsError::UnsupportedDtype(format!("{:?}", other)).into());
            }
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let data_len = view.data().len();

        // Calculate offset relative to mmap start
        let mmap_ptr = mmap_arc.as_ptr() as usize;
        let data_ptr = view.data().as_ptr() as usize;
        let offset = data_ptr - mmap_ptr;

        let tensor = Tensor::from_mmap(mmap_arc.clone(), offset, data_len, shape, dtype);
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

/// Convert IEEE 754 half-precision float to single-precision
fn half_to_f32(half: u16) -> f32 {
    let sign = (half >> 15) & 1;
    let exp = (half >> 10) & 0x1F;
    let frac = half & 0x3FF;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits((sign as u32) << 31)
        } else {
            let val = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let exp32 = (exp as i32) - 15 + 127;
        let frac32 = (frac as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | ((exp32 as u32) << 23) | frac32)
    }
}

/// Convert bfloat16 to single-precision float
fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// DType byte encoding for custom binary format
fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I8 => 3,
        DType::U8 => 4,
        DType::Q8_0 => 5,
        DType::Q4_0 => 6,
        DType::Q4_1 => 7,
    }
}

fn byte_to_dtype(b: u8) -> Result<DType, SafeTensorsError> {
    match b {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U8),
        5 => Ok(DType::Q8_0),
        6 => Ok(DType::Q4_0),
        7 => Ok(DType::Q4_1),
        _ => Err(SafeTensorsError::UnsupportedDtype(format!("Unknown dtype byte: {}", b))),
    }
}

/// Save tensors in custom binary format with trailing CRC32.
///
/// Format: \[NumTensors: u32\]
/// Per tensor: \[NameLen: u32, Name: bytes, DType: u8, NDim: u32, Shape: \[u32; NDim\], DataLen: u64, Data: bytes\]
/// Trailer: \[CRC32: u32\] (over all preceding bytes)
pub fn save_custom_bin(path: &Path, tensors: &HashMap<String, Tensor>) -> HubResult<()> {
    use std::io::Write;

    let mut buf: Vec<u8> = Vec::new();

    // NumTensors
    buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    for (name, tensor) in tensors {
        // NameLen + Name
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        // DType
        buf.push(dtype_to_byte(tensor.dtype()));

        // NDim
        let shape = tensor.shape();
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());

        // Shape
        for &dim in shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Data
        let data = tensor.as_raw_bytes()?;
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }

    // CRC32 over all preceding bytes
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let mut file = File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
}

/// Load tensors from custom binary format with CRC32 verification.
pub fn load_custom_bin(path: &Path) -> HubResult<HashMap<String, Tensor>> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() < 8 {
        return Err(SafeTensorsError::DataCorruption("File too small".into()).into());
    }

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;

    // Verify CRC32: last 4 bytes are the checksum
    if buffer.len() < 4 {
        return Err(SafeTensorsError::DataCorruption("File too small for CRC32".into()).into());
    }
    let payload = &buffer[..buffer.len() - 4];
    let stored_crc = u32::from_le_bytes(buffer[buffer.len() - 4..].try_into().unwrap());
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(SafeTensorsError::DataCorruption(format!(
            "CRC32 mismatch: expected {:08x}, got {:08x}",
            stored_crc, computed_crc
        ))
        .into());
    }

    let mut cursor = 0;

    let num_tensors = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
    cursor += 4;

    let mut tensors = HashMap::new();

    for _ in 0..num_tensors {
        // Name
        if cursor + 4 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let name_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        if cursor + name_len > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let name = String::from_utf8(payload[cursor..cursor + name_len].to_vec())
            .map_err(|e| SafeTensorsError::DataCorruption(e.to_string()))?;
        cursor += name_len;

        // DType
        if cursor + 1 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let dtype = byte_to_dtype(payload[cursor])?;
        cursor += 1;

        // NDim
        if cursor + 4 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let ndim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;

        // Shape
        let mut shape = Vec::new();
        for _ in 0..ndim {
            if cursor + 4 > payload.len() {
                return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
            }
            let dim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            shape.push(dim);
        }

        // Data
        if cursor + 8 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let data_len = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;

        if cursor + data_len > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let data = payload[cursor..cursor + data_len].to_vec();
        cursor += data_len;

        let tensor = Tensor::new(data, shape, dtype);
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        // 1.0 in bf16 is 0x3F80
        let one = bf16_to_f32(0x3F80);
        assert!((one - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_conversion() {
        // 1.0 in f16 is 0x3C00
        let one = half_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6);

        // 0.0 in f16
        let zero = half_to_f32(0x0000);
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_custom_bin_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.1, 0.2], vec![2]).unwrap(),
        );

        let dir = std::env::temp_dir().join("rustml_test_custom_bin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bin");

        save_custom_bin(&path, &tensors).unwrap();
        let loaded = load_custom_bin(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));

        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), DType::F32);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dtype_byte_roundtrip() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::Q8_0, DType::Q4_0] {
            let b = dtype_to_byte(dtype);
            let back = byte_to_dtype(b).unwrap();
            assert_eq!(back, dtype);
        }
    }
}
