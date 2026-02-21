use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::api::error::{GgufError, GgufResult};
use crate::api::types::*;
use crate::core::weight_map::{gguf_bert_weight_map, gguf_gemma3_weight_map, gguf_llama_weight_map, gguf_nomic_bert_weight_map};

/// GGUF magic bytes: "GGUF"
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

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

    fn read_u8(&mut self) -> GgufResult<u8> {
        if self.remaining() < 1 {
            return Err(eof_error());
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> GgufResult<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> GgufResult<u16> {
        if self.remaining() < 2 {
            return Err(eof_error());
        }
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> GgufResult<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> GgufResult<u32> {
        if self.remaining() < 4 {
            return Err(eof_error());
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> GgufResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> GgufResult<u64> {
        if self.remaining() < 8 {
            return Err(eof_error());
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> GgufResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> GgufResult<f32> {
        if self.remaining() < 4 {
            return Err(eof_error());
        }
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> GgufResult<f64> {
        if self.remaining() < 8 {
            return Err(eof_error());
        }
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bool(&mut self) -> GgufResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> GgufResult<String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len {
            return Err(eof_error());
        }
        let s = String::from_utf8(self.data[self.pos..self.pos + len].to_vec())
            .map_err(|e| GgufError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        self.pos += len;
        Ok(s)
    }

    fn read_bytes(&mut self, len: usize) -> GgufResult<&'a [u8]> {
        if self.remaining() < len {
            return Err(eof_error());
        }
        let data = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(data)
    }

    fn read_value(&mut self, type_id: u32) -> GgufResult<GGUFValue> {
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
            _ => Err(GgufError::UnsupportedType(format!(
                "Unknown GGUF value type: {}",
                type_id
            ))),
        }
    }
}

fn eof_error() -> GgufError {
    GgufError::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "Unexpected end of GGUF file",
    ))
}

impl GGUFFile {
    /// Parse the GGUF header, metadata, and tensor info from a file.
    pub fn parse_header<P: AsRef<Path>>(path: P) -> GgufResult<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::parse_bytes(&data)
    }

    /// Parse GGUF from raw bytes.
    pub fn parse_bytes(data: &[u8]) -> GgufResult<Self> {
        let mut cur = Cursor::new(data);

        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidFormat(format!(
                "Invalid GGUF magic: expected {:?}, got {:?}",
                GGUF_MAGIC, magic
            )));
        }

        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(GgufError::InvalidFormat(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        let tensor_count = cur.read_u64()? as usize;
        let metadata_count = cur.read_u64()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = cur.read_string()?;
            let value_type = cur.read_u32()?;
            let value = cur.read_value(value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(cur.read_u64()? as usize);
            }
            let ggml_type_id = cur.read_u32()?;
            let ggml_type = GGMLType::from_u32(ggml_type_id).ok_or_else(|| {
                GgufError::UnsupportedType(format!("GGML type: {}", ggml_type_id))
            })?;
            let offset = cur.read_u64()?;

            tensor_infos.push(GGUFTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        let data_offset = (cur.pos + 31) & !31;

        Ok(GGUFFile {
            version,
            metadata,
            tensor_infos,
            data_offset,
        })
    }

    /// Extract a model config from GGUF metadata.
    ///
    /// Detects the architecture from `general.architecture` and uses the
    /// appropriate metadata key prefix (e.g. `llama.`, `gemma3.`).
    pub fn to_model_config(&self) -> GgufResult<GgufModelConfig> {
        let arch = self
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_string())
            .unwrap_or("llama")
            .to_string();

        let get_u32 = |key: &str| -> GgufResult<u32> {
            self.metadata
                .get(key)
                .and_then(|v| v.as_u32())
                .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
        };

        let get_u32_opt =
            |key: &str| -> Option<u32> { self.metadata.get(key).and_then(|v| v.as_u32()) };

        let get_f32_opt =
            |key: &str| -> Option<f32> { self.metadata.get(key).and_then(|v| v.as_f32()) };

        let dim = get_u32(&format!("{}.embedding_length", arch))? as usize;
        let n_heads = get_u32(&format!("{}.attention.head_count", arch))? as usize;
        let n_layers = get_u32(&format!("{}.block_count", arch))? as usize;
        let vocab_size = get_u32(&format!("{}.vocab_size", arch))
            .or_else(|_| get_u32("tokenizer.ggml.tokens_count"))
            .or_else(|_| -> GgufResult<u32> {
                // Infer vocab size from tokenizer tokens array length
                if let Some(GGUFValue::Array(tokens)) = self.metadata.get("tokenizer.ggml.tokens") {
                    Ok(tokens.len() as u32)
                } else {
                    Ok(32000)
                }
            })
            .unwrap_or(32000) as usize;

        let hidden_dim =
            get_u32(&format!("{}.feed_forward_length", arch)).unwrap_or((dim * 4) as u32) as usize;
        let n_kv_heads = get_u32(&format!("{}.attention.head_count_kv", arch))
            .ok()
            .map(|v| v as usize);
        let max_seq_len = get_u32(&format!("{}.context_length", arch)).unwrap_or(2048) as usize;
        let norm_eps = get_f32_opt(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            .or_else(|| get_f32_opt(&format!("{}.attention.layer_norm_epsilon", arch)))
            .unwrap_or(1e-5);
        let rope_theta = get_f32_opt(&format!("{}.rope.freq_base", arch)).unwrap_or(10000.0);

        let head_dim = get_u32_opt(&format!("{}.attention.key_length", arch)).map(|v| v as usize);
        let sliding_window =
            get_u32_opt(&format!("{}.attention.sliding_window", arch)).map(|v| v as usize);

        let bos_token_id = get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos_token_id = get_u32("tokenizer.ggml.eos_token_id").ok();

        let chat_template = self
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());

        Ok(GgufModelConfig {
            architecture: arch,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            norm_eps,
            max_seq_len,
            rope_theta,
            bos_token_id,
            eos_token_id,
            chat_template,
            head_dim,
            sliding_window,
        })
    }

    /// Load tensor data from the file, returning raw bytes per tensor.
    pub fn load_tensors<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> GgufResult<HashMap<String, LoadedTensor>> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut tensors = HashMap::new();

        for info in &self.tensor_infos {
            let n_elements: usize = info.dimensions.iter().product();
            let n_blocks = if info.ggml_type.block_size() > 1 {
                n_elements / info.ggml_type.block_size()
            } else {
                n_elements
            };
            let byte_size = n_blocks * info.ggml_type.block_bytes();

            let abs_offset = self.data_offset + info.offset as usize;
            if abs_offset + byte_size > data.len() {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Tensor '{}' data exceeds file bounds", info.name),
                )));
            }

            let raw_data = &data[abs_offset..abs_offset + byte_size];

            // GGUF dimensions are stored in reverse order (innermost first)
            let shape: Vec<usize> = info.dimensions.iter().rev().copied().collect();

            // K-quant types are dequantized to F32 during loading
            let loaded = if info.ggml_type.needs_dequant() {
                let f32_data = dequantize_kquant(raw_data, n_elements, info.ggml_type)?;
                let f32_bytes = rustml_quant::f32_vec_to_bytes(f32_data);
                LoadedTensor {
                    data: f32_bytes,
                    shape,
                    dtype: LoadedDType::F32,
                }
            } else {
                let dtype = match info.ggml_type {
                    GGMLType::F32 => LoadedDType::F32,
                    GGMLType::F16 => LoadedDType::F16,
                    GGMLType::Q8_0 => LoadedDType::Q8_0,
                    GGMLType::Q4_0 => LoadedDType::Q4_0,
                    GGMLType::Q4_1 => LoadedDType::Q4_1,
                    GGMLType::Q5_0 | GGMLType::Q5_1 | GGMLType::Q8_1 => {
                        // Legacy quant types: dequantize to F32 at load time
                        let f32_data = dequantize_legacy(raw_data, n_elements, info.ggml_type)?;
                        let f32_bytes = rustml_quant::f32_vec_to_bytes(f32_data);
                        tensors.insert(
                            info.name.clone(),
                            LoadedTensor {
                                data: f32_bytes,
                                shape,
                                dtype: LoadedDType::F32,
                            },
                        );
                        continue;
                    }
                    _ => {
                        return Err(GgufError::UnsupportedType(format!(
                            "Unsupported GGML type {:?} for tensor '{}'",
                            info.ggml_type, info.name
                        )));
                    }
                };
                LoadedTensor {
                    data: raw_data.to_vec(),
                    shape,
                    dtype,
                }
            };
            tensors.insert(info.name.clone(), loaded);
        }

        Ok(tensors)
    }

    /// Load and remap tensors using the standard Llama weight map.
    pub fn load_and_remap<P: AsRef<Path>>(
        &self,
        path: P,
        n_layers: usize,
    ) -> GgufResult<HashMap<String, LoadedTensor>> {
        let tensors = self.load_tensors(path)?;
        let weight_map = gguf_llama_weight_map(n_layers);
        Ok(weight_map.remap(tensors))
    }

    /// Load tensors and remap using Gemma 3 GGUF naming conventions.
    ///
    /// Handles Gemma 3 specifics: QK norms, 4 sandwich norms, GeGLU MLP.
    pub fn load_and_remap_gemma3<P: AsRef<Path>>(
        &self,
        path: P,
        n_layers: usize,
    ) -> GgufResult<HashMap<String, LoadedTensor>> {
        let tensors = self.load_tensors(path)?;
        let weight_map = gguf_gemma3_weight_map(n_layers);
        Ok(weight_map.remap(tensors))
    }

    /// Load tensors and remap using BERT GGUF naming conventions.
    ///
    /// Handles BERT specifics: bias on all projections, `attn_output_norm`/`layer_output_norm`,
    /// position embeddings, embedding norm, no gate_proj.
    pub fn load_and_remap_bert<P: AsRef<Path>>(
        &self,
        path: P,
        n_layers: usize,
    ) -> GgufResult<HashMap<String, LoadedTensor>> {
        let tensors = self.load_tensors(path)?;
        let weight_map = gguf_bert_weight_map(n_layers);
        Ok(weight_map.remap(tensors))
    }

    /// Load tensors and remap using Nomic-BERT GGUF naming conventions.
    ///
    /// Handles Nomic-BERT specifics: fused QKV (no bias), SwiGLU FFN,
    /// RoPE (no position embeddings), embedding norm.
    pub fn load_and_remap_nomic_bert<P: AsRef<Path>>(
        &self,
        path: P,
        n_layers: usize,
    ) -> GgufResult<HashMap<String, LoadedTensor>> {
        let tensors = self.load_tensors(path)?;
        let weight_map = gguf_nomic_bert_weight_map(n_layers);
        Ok(weight_map.remap(tensors))
    }

    /// Read and dequantize a single tensor to f32 values.
    ///
    /// Uses seek+read_exact to read only the bytes for this tensor,
    /// avoiding loading the entire file into memory.
    pub fn read_tensor_f32<P: AsRef<Path>>(
        path: P,
        info: &GGUFTensorInfo,
        data_offset: usize,
    ) -> GgufResult<Vec<f32>> {
        let n_elements: usize = info.dimensions.iter().product();
        let n_blocks = if info.ggml_type.block_size() > 1 {
            n_elements / info.ggml_type.block_size()
        } else {
            n_elements
        };
        let byte_size = n_blocks * info.ggml_type.block_bytes();

        let abs_offset = data_offset + info.offset as usize;
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(abs_offset as u64))?;
        let mut raw = vec![0u8; byte_size];
        file.read_exact(&mut raw)?;

        match info.ggml_type {
            GGMLType::F32 => {
                let mut out = vec![0.0f32; n_elements];
                for i in 0..n_elements {
                    let off = i * 4;
                    out[i] =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                }
                Ok(out)
            }
            GGMLType::F16 => {
                let mut out = vec![0.0f32; n_elements];
                for i in 0..n_elements {
                    let off = i * 2;
                    out[i] = f16_from_bytes(raw[off], raw[off + 1]);
                }
                Ok(out)
            }
            GGMLType::Q4_0 => rustml_quant::dequantize_q4_0(&raw, n_elements)
                .map_err(|e| GgufError::UnsupportedType(e.to_string())),
            GGMLType::Q8_0 => rustml_quant::dequantize_q8_0(&raw, n_elements)
                .map_err(|e| GgufError::UnsupportedType(e.to_string())),
            GGMLType::Q4_1 => rustml_quant::dequantize_q4_1(&raw, n_elements)
                .map_err(|e| GgufError::UnsupportedType(e.to_string())),
            GGMLType::Q5_0 | GGMLType::Q5_1 | GGMLType::Q8_1 => {
                dequantize_legacy(&raw, n_elements, info.ggml_type)
            }
            t if t.needs_dequant() => dequantize_kquant(&raw, n_elements, info.ggml_type),
            _ => Err(GgufError::UnsupportedType(format!(
                "Unsupported GGML type {:?} for read_tensor_f32",
                info.ggml_type
            ))),
        }
    }
}

/// Dequantize legacy quantization types (Q4_1, Q5_0, Q5_1, Q8_1) to f32.
fn dequantize_legacy(data: &[u8], n_elements: usize, ggml_type: GGMLType) -> GgufResult<Vec<f32>> {
    match ggml_type {
        GGMLType::Q4_1 => dequantize_q4_1(data, n_elements),
        GGMLType::Q5_0 => dequantize_q5_0(data, n_elements),
        GGMLType::Q5_1 => dequantize_q5_1(data, n_elements),
        GGMLType::Q8_1 => dequantize_q8_1(data, n_elements),
        _ => Err(GgufError::UnsupportedType(format!(
            "Legacy dequantization not implemented for {:?}",
            ggml_type
        ))),
    }
}

/// Q4_1: 32 elements/block, 20 bytes/block = [f16 d][f16 m][16 bytes 4-bit]
/// v[i] = d * nibble[i] + m
fn dequantize_q4_1(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 20;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let m = f16_from_bytes(block[2], block[3]);
        let qs = &block[4..20];

        let out = &mut output[b * block_size..(b + 1) * block_size];
        for j in 0..16 {
            let lo = (qs[j] & 0xF) as f32;
            let hi = (qs[j] >> 4) as f32;
            out[j] = d * lo + m;
            out[j + 16] = d * hi + m;
        }
    }

    Ok(output)
}

/// Q5_0: 32 elements/block, 22 bytes/block = [f16 d][4 bytes high-bits][16 bytes 4-bit low]
fn dequantize_q5_0(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 22;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let qh = &block[2..6]; // 4 bytes = 32 high-bits
        let qs = &block[6..22]; // 16 bytes = 32 x 4-bit low nibbles

        let out = &mut output[b * block_size..(b + 1) * block_size];
        let qh_u32 = u32::from_le_bytes([qh[0], qh[1], qh[2], qh[3]]);

        for j in 0..16 {
            let lo_nib = (qs[j] & 0xF) as i32;
            let hi_nib = (qs[j] >> 4) as i32;
            let hbit_lo = ((qh_u32 >> j) & 1) as i32;
            let hbit_hi = ((qh_u32 >> (j + 16)) & 1) as i32;
            out[j] = d * ((lo_nib | (hbit_lo << 4)) as f32 - 16.0);
            out[j + 16] = d * ((hi_nib | (hbit_hi << 4)) as f32 - 16.0);
        }
    }

    Ok(output)
}

/// Q5_1: 32 elements/block, 24 bytes/block = [f16 d][f16 m][4 bytes high-bits][16 bytes 4-bit low]
fn dequantize_q5_1(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 24;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let m = f16_from_bytes(block[2], block[3]);
        let qh = &block[4..8];
        let qs = &block[8..24];

        let out = &mut output[b * block_size..(b + 1) * block_size];
        let qh_u32 = u32::from_le_bytes([qh[0], qh[1], qh[2], qh[3]]);

        for j in 0..16 {
            let lo_nib = (qs[j] & 0xF) as u32;
            let hi_nib = (qs[j] >> 4) as u32;
            let hbit_lo = (qh_u32 >> j) & 1;
            let hbit_hi = (qh_u32 >> (j + 16)) & 1;
            out[j] = d * (lo_nib | (hbit_lo << 4)) as f32 + m;
            out[j + 16] = d * (hi_nib | (hbit_hi << 4)) as f32 + m;
        }
    }

    Ok(output)
}

/// Q8_1: 32 elements/block, 40 bytes/block = [f32 d][f32 s][32 bytes int8]
fn dequantize_q8_1(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 40;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        // block[4..8] is the sum (unused for dequantization)
        let qs = &block[8..40];

        let out = &mut output[b * block_size..(b + 1) * block_size];
        for j in 0..32 {
            out[j] = d * (qs[j] as i8 as f32);
        }
    }

    Ok(output)
}

/// Dequantize k-quant block data to f32 values.
fn dequantize_kquant(data: &[u8], n_elements: usize, ggml_type: GGMLType) -> GgufResult<Vec<f32>> {
    match ggml_type {
        GGMLType::Q6K => dequantize_q6k(data, n_elements),
        GGMLType::Q4K => dequantize_q4k(data, n_elements),
        GGMLType::Q5K => dequantize_q5k(data, n_elements),
        GGMLType::Q8K => dequantize_q8k(data, n_elements),
        GGMLType::Q3K => dequantize_q3k(data, n_elements),
        GGMLType::Q2K => dequantize_q2k(data, n_elements),
        _ => Err(GgufError::UnsupportedType(format!(
            "Dequantization not implemented for {:?}",
            ggml_type
        ))),
    }
}

fn f16_from_bytes(lo: u8, hi: u8) -> f32 {
    half::f16::from_le_bytes([lo, hi]).to_f32()
}

fn dequantize_q6k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
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

        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let y_off = half * 128;

            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 as f32
                    - 32.0;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8
                    as f32
                    - 32.0;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 as f32
                    - 32.0;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8
                    as f32
                    - 32.0;

                out[y_off + l] = d * (scales[sc_off + is] as i8 as f32) * q1;
                out[y_off + l + 32] = d * (scales[sc_off + is + 2] as i8 as f32) * q2;
                out[y_off + l + 64] = d * (scales[sc_off + is + 4] as i8 as f32) * q3;
                out[y_off + l + 96] = d * (scales[sc_off + is + 6] as i8 as f32) * q4;
            }
        }
    }

    Ok(output)
}

fn dequantize_q4k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
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

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
            sc[i + 4] = ((scales_raw[i] >> 6) & 3) | ((scales_raw[i + 8] & 0xF) << 2);
            mn[i + 4] = ((scales_raw[i + 4] >> 6) & 3) | ((scales_raw[i + 8] >> 4) << 2);
        }

        let out = &mut output[b * block_size..(b + 1) * block_size];

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

fn dequantize_q5k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
    let block_size = 256;
    let block_bytes = 176;
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let d = f16_from_bytes(block[0], block[1]);
        let dmin = f16_from_bytes(block[2], block[3]);
        let scales_raw = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
            sc[i + 4] = ((scales_raw[i] >> 6) & 3) | ((scales_raw[i + 8] & 0xF) << 2);
            mn[i + 4] = ((scales_raw[i + 4] >> 6) & 3) | ((scales_raw[i + 8] >> 4) << 2);
        }

        let out = &mut output[b * block_size..(b + 1) * block_size];

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

fn dequantize_q8k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
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

fn dequantize_q3k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
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

        let scales_bytes: [u8; 16] =
            unsafe { std::mem::transmute::<[u32; 4], [u8; 16]>(aux.map(|v| v.to_le())) };

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

fn dequantize_q2k(data: &[u8], n_elements: usize) -> GgufResult<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_check() {
        let bad_data = [0x00u8; 20];
        assert!(GGUFFile::parse_bytes(&bad_data).is_err());
    }

    #[test]
    fn test_gguf_valid_minimal() {
        // Construct a minimal valid GGUF v3 file with 0 tensors and 0 metadata
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

        let gguf = GGUFFile::parse_bytes(&data).unwrap();
        assert_eq!(gguf.version, 3);
        assert!(gguf.metadata.is_empty());
        assert!(gguf.tensor_infos.is_empty());
    }
}
