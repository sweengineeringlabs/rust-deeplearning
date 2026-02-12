//! SafeTensors format loader
//!
//! SafeTensors is a simple and safe format for storing tensors.
//! Format specification: https://github.com/huggingface/safetensors

use crate::api::error::HubResult;
use rustml_core::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
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

/// SafeTensors file loader
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

    /// Load tensors from a SafeTensors file
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
                // Skip metadata key
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
                // Convert f16 to f32
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
                // Convert bf16 to f32
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

/// Convert IEEE 754 half-precision float to single-precision
fn half_to_f32(half: u16) -> f32 {
    let sign = (half >> 15) & 1;
    let exp = (half >> 10) & 0x1F;
    let frac = half & 0x3FF;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits((sign as u32) << 31)
        } else {
            // Subnormal
            let val = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        if frac == 0 {
            // Infinity
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            // NaN
            f32::NAN
        }
    } else {
        // Normalized
        let exp32 = (exp as i32) - 15 + 127;
        let frac32 = (frac as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | ((exp32 as u32) << 23) | frac32)
    }
}

/// Convert bfloat16 to single-precision float
fn bf16_to_f32(bf16: u16) -> f32 {
    // bfloat16 is just the top 16 bits of f32
    f32::from_bits((bf16 as u32) << 16)
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
}
