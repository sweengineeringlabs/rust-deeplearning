pub mod weight_map;
pub use weight_map::WeightMap;

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use crate::error::{LLMForgeError, Result};
use crate::core::tensor::{Tensor, DType};

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::sync::Arc;

pub struct ModelLoader;

fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I8 => 3,
        DType::U8 => 4,
        DType::Q8_0 => 5,
    }
}

fn byte_to_dtype(b: u8) -> Result<DType> {
    match b {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U8),
        5 => Ok(DType::Q8_0),
        _ => Err(LLMForgeError::UnknownDType(b)),
    }
}

impl ModelLoader {
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
        let file = File::open(path)?;
        // SAFETY: The file is opened read-only and lives for the duration of the
        // returned tensors (via Arc<Mmap>). Memory-mapped reads are safe as long as
        // no external process truncates/modifies the file while mapped. This is the
        // standard contract for model weight files which are read-only assets.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| LLMForgeError::Io(e))?;
        let mmap_arc = Arc::new(mmap);

        let safetensors = SafeTensors::deserialize(mmap_arc.as_ref())
            .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;

        let mut tensors = HashMap::new();

        for (name, view) in safetensors.tensors() {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => return Err(LLMForgeError::NotImplemented(format!("DType {:?} not supported", view.dtype()))),
            };

            let shape: Vec<usize> = view.shape().iter().map(|&x| x).collect();
            let data_len = view.data().len();

            // Calculate offset relative to mmap start
            let mmap_ptr = mmap_arc.as_ptr() as usize;
            let data_ptr = view.data().as_ptr() as usize;
            let offset = data_ptr - mmap_ptr;

            let tensor = Tensor::from_mmap(
                mmap_arc.clone(),
                offset,
                data_len,
                shape,
                dtype
            );

            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    /// Save tensors in custom binary format with trailing CRC32.
    ///
    /// Format: [NumTensors: u32]
    /// Per tensor: [NameLen: u32, Name: bytes, DType: u8, NDim: u32, Shape: [u32; NDim], DataLen: u64, Data: bytes]
    /// Trailer: [CRC32: u32] (over all preceding bytes)
    pub fn save_custom_bin<P: AsRef<Path>>(path: P, tensors: &HashMap<String, Tensor>) -> Result<()> {
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
    ///
    /// Format: [NumTensors: u32]
    /// Per tensor: [NameLen: u32, Name: bytes, DType: u8, NDim: u32, Shape: [u32; NDim], DataLen: u64, Data: bytes]
    /// Trailer: [CRC32: u32] (over all preceding bytes)
    pub fn load_custom_bin<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
        let metadata = std::fs::metadata(path.as_ref())?;
        if metadata.len() == 0 {
            return Err(LLMForgeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File is empty",
            )));
        }
        // Minimum: 4 bytes num_tensors + 4 bytes CRC32
        if metadata.len() < 8 {
            return Err(LLMForgeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File too small for header",
            )));
        }

        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Verify CRC32: last 4 bytes are the checksum
        if buffer.len() < 4 {
            return Err(LLMForgeError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small for CRC32 trailer",
            )));
        }
        let payload = &buffer[..buffer.len() - 4];
        let stored_crc = u32::from_le_bytes(
            buffer[buffer.len() - 4..].try_into().unwrap()
        );
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(LLMForgeError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("CRC32 mismatch: expected {:08x}, got {:08x}", stored_crc, computed_crc),
            )));
        }

        let mut cursor = 0;

        let num_tensors = u32::from_le_bytes(payload[cursor..cursor+4].try_into().unwrap());
        cursor += 4;

        let mut tensors = HashMap::new();

        for _ in 0..num_tensors {
            // Name Len
            if cursor + 4 > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let name_len = u32::from_le_bytes(payload[cursor..cursor+4].try_into().unwrap()) as usize;
            cursor += 4;

            // Name
            if cursor + name_len > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let name_bytes = &payload[cursor..cursor+name_len];
            let name = String::from_utf8(name_bytes.to_vec())
                .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            cursor += name_len;

            // DType (1 byte)
            if cursor + 1 > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let dtype_byte = payload[cursor];
            cursor += 1;
            let dtype = byte_to_dtype(dtype_byte)?;

            // NDim
            if cursor + 4 > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let ndim = u32::from_le_bytes(payload[cursor..cursor+4].try_into().unwrap());
            cursor += 4;

            // Shape
            let mut shape = Vec::new();
            for _ in 0..ndim {
                if cursor + 4 > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
                let dim = u32::from_le_bytes(payload[cursor..cursor+4].try_into().unwrap()) as usize;
                cursor += 4;
                shape.push(dim);
            }

            // Data Len (u64)
            if cursor + 8 > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let data_len = u64::from_le_bytes(payload[cursor..cursor+8].try_into().unwrap()) as usize;
            cursor += 8;

            // Data
            if cursor + data_len > payload.len() { return Err(LLMForgeError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))); }
            let data = payload[cursor..cursor+data_len].to_vec();
            cursor += data_len;

            let tensor = Tensor::new(data, shape, dtype);
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }
}
