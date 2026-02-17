use std::collections::HashMap;

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
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl GGMLType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2K),
            11 => Some(GGMLType::Q3K),
            12 => Some(GGMLType::Q4K),
            13 => Some(GGMLType::Q5K),
            14 => Some(GGMLType::Q6K),
            15 => Some(GGMLType::Q8K),
            _ => None,
        }
    }

    /// Whether this type needs dequantization to F32 during loading.
    pub fn needs_dequant(&self) -> bool {
        matches!(self, GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K |
                       GGMLType::Q5K | GGMLType::Q6K | GGMLType::Q8K)
    }

    /// Bytes per block for quantized types, or bytes per element for float types.
    pub fn block_bytes(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,
            GGMLType::Q4_1 => 20,
            GGMLType::Q5_0 => 22,
            GGMLType::Q5_1 => 24,
            GGMLType::Q8_0 => 34,
            GGMLType::Q8_1 => 40,
            GGMLType::Q2K => 84,
            GGMLType::Q3K => 110,
            GGMLType::Q4K => 144,
            GGMLType::Q5K => 176,
            GGMLType::Q6K => 210,
            GGMLType::Q8K => 292,
        }
    }

    pub fn block_size(&self) -> usize {
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

/// DType representation for loaded tensors (simplified, independent of rustml-core).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadedDType {
    F32,
    F16,
    Q8_0,
    Q4_0,
}

/// A loaded tensor from a GGUF file (raw bytes + metadata).
#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: LoadedDType,
}

/// Result of loading a GGUF model: config metadata + tensor map.
pub struct GgufBundle {
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: HashMap<String, LoadedTensor>,
    pub model_config: GgufModelConfig,
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct GgufModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: Option<usize>,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub chat_template: Option<String>,
}
