/// GGUF format parser for loading quantized models from the llama.cpp ecosystem.
///
/// GGUF binary format:
/// - 4-byte magic (`GGUF`)
/// - 4-byte version (currently 2 or 3)
/// - 8-byte tensor count
/// - 8-byte metadata count
/// - Metadata KV pairs (typed: u8/i32/f32/string/array/etc.)
/// - Tensor info entries (name, dims, type, offset)
/// - Tensor data (aligned)

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use crate::config::{ModelConfig, PositionEncoding};
use crate::loader::weight_map::WeightMap;

/// GGUF magic bytes: "GGUF"
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// GGML tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl GGMLType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            _ => Err(LLMForgeError::NotImplemented(format!("Unsupported GGML type: {}", v))),
        }
    }

    fn to_dtype(&self) -> Result<DType> {
        match self {
            GGMLType::F32 => Ok(DType::F32),
            GGMLType::F16 => Ok(DType::F16),
            GGMLType::Q8_0 => Ok(DType::Q8_0),
            GGMLType::Q4_0 => Ok(DType::Q4_0),
            _ => Err(LLMForgeError::NotImplemented(
                format!("GGML type {:?} not yet supported for loading", self)
            )),
        }
    }

    /// Bytes per block for quantized types, or bytes per element for float types.
    fn block_bytes(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,   // 2-byte scale + 16 packed bytes for 32 elements
            GGMLType::Q4_1 => 20,   // 2 f16 + 16 bytes
            GGMLType::Q5_0 => 22,   // 2 f16 + 4 + 16 bytes
            GGMLType::Q5_1 => 24,   // 2*2 f16 + 4 + 16 bytes
            GGMLType::Q8_0 => 34,   // 2-byte scale + 32 bytes for 32 elements
            GGMLType::Q8_1 => 40,   // 2*2 f16 + 32 bytes + 4 pad
        }
    }

    fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 => 1,
            _ => 32,
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

impl GGUFValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFValue::U32(v) => Some(*v),
            GGUFValue::I32(v) => Some(*v as u32),
            GGUFValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GGUFValue::F32(v) => Some(*v),
            GGUFValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// GGUF tensor info entry
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub ggml_type: GGMLType,
    pub offset: u64,
}

/// Parsed GGUF file header and metadata.
#[derive(Debug)]
pub struct GGUFFile {
    pub version: u32,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensor_infos: Vec<GGUFTensorInfo>,
    pub data_offset: usize,
}

/// Helper to read little-endian integers from a byte buffer.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.remaining() < 1 {
            return Err(eof_error());
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        if self.remaining() < 2 {
            return Err(eof_error());
        }
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.remaining() < 4 {
            return Err(eof_error());
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.remaining() < 8 {
            return Err(eof_error());
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        if self.remaining() < 4 {
            return Err(eof_error());
        }
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        if self.remaining() < 8 {
            return Err(eof_error());
        }
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len {
            return Err(eof_error());
        }
        let s = String::from_utf8(self.data[self.pos..self.pos + len].to_vec())
            .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        self.pos += len;
        Ok(s)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            return Err(eof_error());
        }
        let data = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(data)
    }

    fn read_value(&mut self, type_id: u32) -> Result<GGUFValue> {
        match type_id {
            0 => Ok(GGUFValue::U8(self.read_u8()?)),
            1 => Ok(GGUFValue::I8(self.read_i8()?)),
            2 => Ok(GGUFValue::U16(self.read_u16()?)),
            3 => Ok(GGUFValue::I16(self.read_i16()?)),
            4 => Ok(GGUFValue::U32(self.read_u32()?)),
            5 => Ok(GGUFValue::I32(self.read_i32()?)),
            6 => Ok(GGUFValue::F32(self.read_f32()?)),
            7 => Ok(GGUFValue::Bool(self.read_bool()?)),
            8 => Ok(GGUFValue::String(self.read_string()?)),
            9 => {
                // Array
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_value(elem_type)?);
                }
                Ok(GGUFValue::Array(values))
            }
            10 => Ok(GGUFValue::U64(self.read_u64()?)),
            11 => Ok(GGUFValue::I64(self.read_i64()?)),
            12 => Ok(GGUFValue::F64(self.read_f64()?)),
            _ => Err(LLMForgeError::NotImplemented(format!("Unknown GGUF value type: {}", type_id))),
        }
    }
}

fn eof_error() -> LLMForgeError {
    LLMForgeError::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "Unexpected end of GGUF file",
    ))
}

impl GGUFFile {
    /// Parse the GGUF header, metadata, and tensor info from a file.
    pub fn parse_header<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        Self::parse_bytes(&data)
    }

    /// Parse GGUF from raw bytes (useful for testing).
    pub fn parse_bytes(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);

        // Magic
        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(LLMForgeError::InvalidConfig(
                format!("Invalid GGUF magic: expected {:?}, got {:?}", GGUF_MAGIC, magic)
            ));
        }

        // Version
        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(LLMForgeError::InvalidConfig(
                format!("Unsupported GGUF version: {}", version)
            ));
        }

        // Counts
        let tensor_count = cur.read_u64()? as usize;
        let metadata_count = cur.read_u64()? as usize;

        // Metadata KV pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = cur.read_string()?;
            let value_type = cur.read_u32()?;
            let value = cur.read_value(value_type)?;
            metadata.insert(key, value);
        }

        // Tensor info entries
        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(cur.read_u64()? as usize);
            }
            let ggml_type_id = cur.read_u32()?;
            let ggml_type = GGMLType::from_u32(ggml_type_id)?;
            let offset = cur.read_u64()?;

            tensor_infos.push(GGUFTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        // Data starts after header, aligned to 32 bytes
        let data_offset = (cur.pos + 31) & !31;

        Ok(GGUFFile {
            version,
            metadata,
            tensor_infos,
            data_offset,
        })
    }

    /// Extract a ModelConfig from GGUF metadata.
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        let get_u32 = |key: &str| -> Result<u32> {
            self.metadata.get(key)
                .and_then(|v| v.as_u32())
                .ok_or_else(|| LLMForgeError::InvalidConfig(format!("Missing GGUF metadata: {}", key)))
        };

        let get_f32_opt = |key: &str| -> Option<f32> {
            self.metadata.get(key).and_then(|v| v.as_f32())
        };

        let dim = get_u32("llama.embedding_length")? as usize;
        let n_heads = get_u32("llama.attention.head_count")? as usize;
        let n_layers = get_u32("llama.block_count")? as usize;
        let vocab_size = get_u32("llama.vocab_size")
            .or_else(|_| get_u32("tokenizer.ggml.tokens_count"))
            .unwrap_or(32000) as usize;

        let hidden_dim = get_u32("llama.feed_forward_length").unwrap_or((dim * 4) as u32) as usize;
        let n_kv_heads = get_u32("llama.attention.head_count_kv").ok().map(|v| v as usize);
        let max_seq_len = get_u32("llama.context_length").unwrap_or(2048) as usize;
        let norm_eps = get_f32_opt("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_theta = get_f32_opt("llama.rope.freq_base").unwrap_or(10000.0);

        let config = ModelConfig {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            norm_eps,
            max_seq_len,
            use_bias: Some(false),
            position_encoding: PositionEncoding::RoPE,
            causal: true,
            rope_theta,
        };
        config.validate()?;
        Ok(config)
    }

    /// Load tensor data from the file, converting supported GGML types to our DTypes.
    pub fn load_tensors<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, Tensor>> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut tensors = HashMap::new();

        for info in &self.tensor_infos {
            let dtype = info.ggml_type.to_dtype()?;
            let n_elements: usize = info.dimensions.iter().product();
            let n_blocks = if info.ggml_type.block_size() > 1 {
                n_elements / info.ggml_type.block_size()
            } else {
                n_elements
            };
            let byte_size = n_blocks * info.ggml_type.block_bytes();

            let abs_offset = self.data_offset + info.offset as usize;
            if abs_offset + byte_size > data.len() {
                return Err(LLMForgeError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Tensor '{}' data exceeds file bounds", info.name),
                )));
            }

            let tensor_data = data[abs_offset..abs_offset + byte_size].to_vec();

            // GGUF dimensions are stored in reverse order (innermost first)
            let shape: Vec<usize> = info.dimensions.iter().rev().copied().collect();

            let tensor = Tensor::new(tensor_data, shape, dtype);
            tensors.insert(info.name.clone(), tensor);
        }

        Ok(tensors)
    }
}

/// Build a weight mapping for GGUF Llama-style tensor names.
///
/// GGUF naming: `token_embd.weight`, `blk.{i}.attn_q.weight`, etc.
pub fn gguf_weight_map(n_layers: usize) -> WeightMap {
    let mut mapping = HashMap::new();

    // Global
    mapping.insert("token_embd.weight".to_string(), "token_embedding.weight".to_string());
    mapping.insert("output_norm.weight".to_string(), "norm.weight".to_string());
    mapping.insert("output.weight".to_string(), "output.weight".to_string());

    // Per-layer
    for i in 0..n_layers {
        mapping.insert(
            format!("blk.{}.attn_q.weight", i),
            format!("layers.{}.attention.q_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_k.weight", i),
            format!("layers.{}.attention.k_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_v.weight", i),
            format!("layers.{}.attention.v_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_output.weight", i),
            format!("layers.{}.attention.out_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_norm.weight", i),
            format!("layers.{}.attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_norm.weight", i),
            format!("layers.{}.ffn_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_up.weight", i),
            format!("layers.{}.feed_forward.up_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", i),
            format!("layers.{}.feed_forward.down_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_gate.weight", i),
            format!("layers.{}.feed_forward.gate_proj.weight", i),
        );
    }

    WeightMap::from_mapping(mapping)
}
