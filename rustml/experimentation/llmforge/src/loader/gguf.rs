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
    // K-quant types (super-block size = 256)
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
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
            10 => Ok(GGMLType::Q2K),
            11 => Ok(GGMLType::Q3K),
            12 => Ok(GGMLType::Q4K),
            13 => Ok(GGMLType::Q5K),
            14 => Ok(GGMLType::Q6K),
            15 => Ok(GGMLType::Q8K),
            _ => Err(LLMForgeError::NotImplemented(format!("Unsupported GGML type: {}", v))),
        }
    }

    /// Map to internal DType. K-quant types are dequantized to F32 during loading.
    fn to_dtype(&self) -> Result<DType> {
        match self {
            GGMLType::F32 => Ok(DType::F32),
            GGMLType::F16 => Ok(DType::F16),
            GGMLType::Q8_0 => Ok(DType::Q8_0),
            GGMLType::Q4_0 => Ok(DType::Q4_0),
            // K-quant types are dequantized to F32 during load
            GGMLType::Q6K | GGMLType::Q4K | GGMLType::Q5K |
            GGMLType::Q3K | GGMLType::Q2K | GGMLType::Q8K => Ok(DType::F32),
            _ => Err(LLMForgeError::NotImplemented(
                format!("GGML type {:?} not yet supported for loading", self)
            )),
        }
    }

    /// Whether this type needs dequantization to F32 during loading.
    fn needs_dequant(&self) -> bool {
        matches!(self, GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K |
                       GGMLType::Q5K | GGMLType::Q6K | GGMLType::Q8K)
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
            // K-quant types: super-block size = 256 (QK_K)
            GGMLType::Q2K => 256 / 16 + 256 / 4 + 2 + 2, // 84 bytes
            GGMLType::Q3K => 256 / 8 + 256 / 4 + 12 + 2, // 110 bytes
            GGMLType::Q4K => 2 + 2 + 12 + 256 / 2,       // 144 bytes
            GGMLType::Q5K => 2 + 2 + 12 + 256 / 8 + 256 / 2, // 176 bytes
            GGMLType::Q6K => 256 / 2 + 256 / 4 + 256 / 16 + 2, // 210 bytes
            GGMLType::Q8K => 4 + 256 + 16 * 2,            // 292 bytes
        }
    }

    fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 => 1,
            GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K |
            GGMLType::Q5K | GGMLType::Q6K | GGMLType::Q8K => 256,
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

        let bos_token_id = get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos_token_id = get_u32("tokenizer.ggml.eos_token_id").ok();

        let chat_template = self.metadata.get("tokenizer.chat_template")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());

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
            bos_token_id,
            eos_token_id,
            chat_template,
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

            let raw_data = &data[abs_offset..abs_offset + byte_size];

            // GGUF dimensions are stored in reverse order (innermost first)
            let shape: Vec<usize> = info.dimensions.iter().rev().copied().collect();

            // K-quant types are dequantized to F32 during loading
            let tensor = if info.ggml_type.needs_dequant() {
                let f32_data = dequantize_kquant(raw_data, n_elements, info.ggml_type)?;
                let f32_bytes = crate::core::tensor::f32_vec_to_bytes(f32_data);
                Tensor::new(f32_bytes, shape, DType::F32)
            } else {
                Tensor::new(raw_data.to_vec(), shape, dtype)
            };
            tensors.insert(info.name.clone(), tensor);
        }

        Ok(tensors)
    }
}

/// Dequantize k-quant block data to f32 values.
fn dequantize_kquant(data: &[u8], n_elements: usize, ggml_type: GGMLType) -> Result<Vec<f32>> {
    match ggml_type {
        GGMLType::Q6K => dequantize_q6k(data, n_elements),
        GGMLType::Q4K => dequantize_q4k(data, n_elements),
        GGMLType::Q5K => dequantize_q5k(data, n_elements),
        GGMLType::Q8K => dequantize_q8k(data, n_elements),
        GGMLType::Q3K => dequantize_q3k(data, n_elements),
        GGMLType::Q2K => dequantize_q2k(data, n_elements),
        _ => Err(LLMForgeError::NotImplemented(
            format!("Dequantization not implemented for {:?}", ggml_type)
        )),
    }
}

fn f16_from_bytes(lo: u8, hi: u8) -> f32 {
    half::f16::from_le_bytes([lo, hi]).to_f32()
}

/// Dequantize Q6_K blocks (matches llama.cpp dequantize_row_q6_K).
/// Layout per block (256 elements, 210 bytes):
///   ql[128]: lower 4 bits of 6-bit quants
///   qh[64]:  upper 2 bits of 6-bit quants
///   scales[16]: int8 sub-block scales
///   d[2]: f16 super-block scale
///
/// llama.cpp processes each 256-element block in two 128-element halves.
/// Within each half, l iterates 0..32 producing 4 outputs at [l, l+32, l+64, l+96].
/// Between halves, ql advances by 64, qh by 32, scales by 8.
fn dequantize_q6k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 210;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_from_bytes(block[208], block[209]);

        let out = &mut output[b * block_size..(b + 1) * block_size];

        // Two passes of 128 elements each, matching llama.cpp pointer advancement
        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let y_off = half * 128;

            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 as f32 - 32.0;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 as f32 - 32.0;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 as f32 - 32.0;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 as f32 - 32.0;

                out[y_off + l]      = d * (scales[sc_off + is] as i8 as f32) * q1;
                out[y_off + l + 32] = d * (scales[sc_off + is + 2] as i8 as f32) * q2;
                out[y_off + l + 64] = d * (scales[sc_off + is + 4] as i8 as f32) * q3;
                out[y_off + l + 96] = d * (scales[sc_off + is + 6] as i8 as f32) * q4;
            }
        }
    }

    Ok(output)
}

/// Dequantize Q4_K blocks (matches llama.cpp dequantize_row_q4_K).
/// Layout per block (256 elements, 144 bytes):
///   d[2]: f16 super-block scale
///   dmin[2]: f16 super-block min
///   scales[12]: packed 6-bit scale/min pairs for 8 sub-blocks
///   qs[128]: 4-bit quants, 2 per byte
///
/// llama.cpp processes 64 elements per chunk: first 32 = lo nibbles, next 32 = hi nibbles.
/// qs pointer advances by 32 per chunk. 4 chunks total = 256 elements.
fn dequantize_q4k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 144;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let dmin = f16_from_bytes(block[2], block[3]);
        let scales_raw = &block[4..16];
        let qs = &block[16..144];

        // Unpack 6-bit scales and mins from 12 bytes into 8 pairs
        // (matches get_scale_min_k4 in llama.cpp)
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
            sc[i + 4] = ((scales_raw[i] >> 6) & 3) | ((scales_raw[i + 8] & 0xF) << 2);
            mn[i + 4] = ((scales_raw[i + 4] >> 6) & 3) | ((scales_raw[i + 8] >> 4) << 2);
        }

        let out = &mut output[b * block_size..(b + 1) * block_size];

        // 4 chunks of 64 elements each; each chunk uses 32 qs bytes
        let mut is = 0usize;
        for chunk in 0..4usize {
            let q_off = chunk * 32;
            let y_off = chunk * 64;
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            for l in 0..32 {
                out[y_off + l] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            is += 2;
        }
    }

    Ok(output)
}

/// Dequantize Q5_K blocks (matches llama.cpp dequantize_row_q5_K).
/// Layout per block (256 elements, 176 bytes):
///   d[2], dmin[2], scales[12], qh[32], qs[128]
///
/// llama.cpp processes 64 elements per chunk: first 32 = lo nibbles + qh bit,
/// next 32 = hi nibbles + qh bit. ql advances by 32, qh bits rotate. 4 chunks = 256.
fn dequantize_q5k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 176;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let dmin = f16_from_bytes(block[2], block[3]);
        let scales_raw = &block[4..16];
        let qh = &block[16..48];  // 32 bytes, high bits
        let qs = &block[48..176]; // 128 bytes, low 4 bits

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
            sc[i + 4] = ((scales_raw[i] >> 6) & 3) | ((scales_raw[i + 8] & 0xF) << 2);
            mn[i + 4] = ((scales_raw[i + 4] >> 6) & 3) | ((scales_raw[i + 8] >> 4) << 2);
        }

        let out = &mut output[b * block_size..(b + 1) * block_size];

        // u1 and u2 are bit masks for qh, rotating left by 2 each chunk
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut is = 0usize;

        for chunk in 0..4usize {
            let ql_off = chunk * 32;
            let y_off = chunk * 64;
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            for l in 0..32 {
                let hbit = if qh[l] & u1 != 0 { 16u8 } else { 0 };
                out[y_off + l] = d1 * ((qs[ql_off + l] & 0xF) + hbit) as f32 - m1;
            }
            for l in 0..32 {
                let hbit = if qh[l] & u2 != 0 { 16u8 } else { 0 };
                out[y_off + 32 + l] = d2 * ((qs[ql_off + l] >> 4) + hbit) as f32 - m2;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(output)
}

/// Dequantize Q8_K blocks.
/// Layout per block (256 elements):
///   d[4]: f32 super-block scale
///   qs[256]: int8 quants
///   bsums[32]: int16 block sums (unused for dequant)
fn dequantize_q8k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 292;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];

        let out = &mut output[b * block_size..(b + 1) * block_size];
        for j in 0..256 {
            out[j] = d * (qs[j] as i8 as f32);
        }
    }

    Ok(output)
}

/// Dequantize Q3_K blocks (matches llama.cpp dequantize_row_q3_K).
/// Layout per block (256 elements, 110 bytes):
///   hmask[32], qs[64], scales[12], d[2]
///
/// llama.cpp processes two 128-element halves. Within each half, 4 shift values (0,2,4,6)
/// iterate over the same 32 qs bytes, producing 4×32=128 elements.
fn dequantize_q3k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 110;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let hm = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d_all = f16_from_bytes(block[108], block[109]);

        // Unpack scales using llama.cpp's aux/kmask method
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
        aux[1] = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
        aux[2] = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // Reinterpret aux as 16 i8 scales (matches llama.cpp's `const int8_t *scales = (const int8_t*)aux`)
        let scales_bytes: [u8; 16] = unsafe {
            std::mem::transmute::<[u32; 4], [u8; 16]>(aux.map(|v| v.to_le()))
        };

        let out = &mut output[b * block_size..(b + 1) * block_size];

        let mut y_pos = 0usize;
        let mut m: u8 = 1;
        let mut is = 0usize;

        for half in 0..2usize {
            let q_off = half * 32;
            let mut shift = 0u8;

            for _j in 0..4 {
                let dl = d_all * (scales_bytes[is] as i8 as f32 - 32.0);
                for l in 0..16usize {
                    let q2 = (qs[q_off + l] >> shift) & 3;
                    let hbit = if hm[l] & m != 0 { 0i8 } else { -4 };
                    out[y_pos] = dl * (q2 as i8 + hbit) as f32;
                    y_pos += 1;
                }

                let dl = d_all * (scales_bytes[is + 1] as i8 as f32 - 32.0);
                for l in 0..16usize {
                    let q2 = (qs[q_off + l + 16] >> shift) & 3;
                    let hbit = if hm[l + 16] & m != 0 { 0i8 } else { -4 };
                    out[y_pos] = dl * (q2 as i8 + hbit) as f32;
                    y_pos += 1;
                }

                is += 2;
                shift += 2;
                m <<= 1;
            }
        }
    }

    Ok(output)
}

/// Dequantize Q2_K blocks (matches llama.cpp dequantize_row_q2_K).
/// Layout per block (256 elements, 84 bytes):
///   scales[16], qs[64], d[2], dmin[2]
///
/// llama.cpp processes two 128-element halves. Within each half, 4 shift values
/// iterate over the same 32 qs bytes, producing 4×32=128 elements.
fn dequantize_q2k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 84;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_from_bytes(block[80], block[81]);
        let dmin = f16_from_bytes(block[82], block[83]);

        let out = &mut output[b * block_size..(b + 1) * block_size];

        let mut y_pos = 0usize;
        let mut is = 0usize;

        for half in 0..2usize {
            let q_off = half * 32;
            let mut shift = 0u8;

            for _j in 0..4 {
                let sc_byte = scales[is];
                is += 1;
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                for l in 0..16usize {
                    out[y_pos] = dl * ((qs[q_off + l] >> shift) & 3) as f32 - ml;
                    y_pos += 1;
                }

                let sc_byte = scales[is];
                is += 1;
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                for l in 0..16usize {
                    out[y_pos] = dl * ((qs[q_off + l + 16] >> shift) & 3) as f32 - ml;
                    y_pos += 1;
                }

                shift += 2;
            }
        }
    }

    Ok(output)
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
