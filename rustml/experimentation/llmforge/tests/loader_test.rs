use llmforge::loader::ModelLoader;
use llmforge::core::tensor::{Tensor, DType};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_custom_bin_loader() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("model.bin");

    // Create a dummy binary file with CRC32 trailer
    {
        let mut payload = Vec::new();
        // 1 tensor
        payload.extend_from_slice(&1u32.to_le_bytes());

        // Tensor 1: "test_tensor"
        let name = "test_tensor";
        payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
        payload.extend_from_slice(name.as_bytes());

        // DType: F32 (0)
        payload.push(0u8);

        // NDim: 2
        payload.extend_from_slice(&2u32.to_le_bytes());

        // Shape: [2, 2]
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&2u32.to_le_bytes());

        // Data: [1.0, 2.0, 3.0, 4.0] (F32)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        for f in data {
            bytes.extend_from_slice(&f.to_ne_bytes());
        }

        // Data Len
        payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());

        // Data
        payload.extend_from_slice(&bytes);

        // CRC32 trailer
        let crc = crc32fast::hash(&payload);
        payload.extend_from_slice(&crc.to_le_bytes());

        let mut file = File::create(&file_path).unwrap();
        file.write_all(&payload).unwrap();
    }

    // Load it back
    let tensors = ModelLoader::load_custom_bin(&file_path).expect("Failed to load");

    assert!(tensors.contains_key("test_tensor"));
    let t = tensors.get("test_tensor").unwrap();
    assert_eq!(t.shape(), &[2, 2]);

    // Verify content
    let slice = t.as_slice_f32().unwrap();
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[3], 4.0);
}

#[test]
fn test_custom_bin_roundtrip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("roundtrip.bin");

    // Create tensors
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes_a = bytemuck::cast_slice::<f32, u8>(&data_a).to_vec();
    let tensor_a = Tensor::new(bytes_a, vec![2, 3], DType::F32);

    let data_b: Vec<f32> = vec![10.0, 20.0];
    let bytes_b = bytemuck::cast_slice::<f32, u8>(&data_b).to_vec();
    let tensor_b = Tensor::new(bytes_b, vec![2], DType::F32);

    let mut tensors = HashMap::new();
    tensors.insert("weight_a".to_string(), tensor_a);
    tensors.insert("weight_b".to_string(), tensor_b);

    // Save
    ModelLoader::save_custom_bin(&file_path, &tensors).expect("save failed");

    // Load
    let loaded = ModelLoader::load_custom_bin(&file_path).expect("load failed");

    assert_eq!(loaded.len(), 2);

    let la = loaded.get("weight_a").unwrap();
    assert_eq!(la.shape(), &[2, 3]);
    let la_data = la.as_slice_f32().unwrap();
    assert_eq!(la_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let lb = loaded.get("weight_b").unwrap();
    assert_eq!(lb.shape(), &[2]);
    let lb_data = lb.as_slice_f32().unwrap();
    assert_eq!(lb_data, &[10.0, 20.0]);
}

#[test]
fn test_custom_bin_corrupt_crc() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("corrupt.bin");

    // Create a valid file first
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&data).to_vec();
    let tensor = Tensor::new(bytes, vec![2, 2], DType::F32);

    let mut tensors = HashMap::new();
    tensors.insert("test".to_string(), tensor);

    ModelLoader::save_custom_bin(&file_path, &tensors).expect("save failed");

    // Corrupt a byte in the middle of the file
    let mut content = std::fs::read(&file_path).unwrap();
    if content.len() > 10 {
        content[10] ^= 0xFF; // flip bits
    }
    std::fs::write(&file_path, &content).unwrap();

    // Load should fail with CRC mismatch
    let result = ModelLoader::load_custom_bin(&file_path);
    assert!(result.is_err(), "Expected CRC32 mismatch error");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("CRC32"), "Error should mention CRC32: {}", err_msg);
}

// --- New tests below ---

#[test]
fn safetensors_bf16_loading() {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bf16.safetensors");

    let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bf16_bytes: Vec<u8> = f32_vals
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();

    let view = TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).unwrap();
    let tensor_list: Vec<(&str, &TensorView)> = vec![("w", &view)];
    let serialized = safetensors::serialize(tensor_list, &None).unwrap();

    let mut file = File::create(&file_path).unwrap();
    file.write_all(&serialized).unwrap();

    let loaded = ModelLoader::load_safetensors(&file_path).unwrap();
    let t = loaded.get("w").unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.shape(), &[2, 2]);

    // Convert to f32 and verify
    let f32_t = t.to_f32().unwrap();
    let data = f32_t.as_slice_f32().unwrap();
    for (i, &expected) in f32_vals.iter().enumerate() {
        assert!(
            (data[i] - expected).abs() < 0.05,
            "BF16 roundtrip: expected {}, got {}",
            expected,
            data[i]
        );
    }
}

#[test]
fn safetensors_f32_loading() {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("f32.safetensors");

    let f32_vals: Vec<f32> = vec![3.14, -2.71, 0.0, 42.0, 1.0, -1.0];
    let bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let view = TensorView::new(Dtype::F32, vec![2, 3], &bytes).unwrap();
    let tensor_list: Vec<(&str, &TensorView)> = vec![("weight", &view)];
    let serialized = safetensors::serialize(tensor_list, &None).unwrap();

    let mut file = File::create(&file_path).unwrap();
    file.write_all(&serialized).unwrap();

    let loaded = ModelLoader::load_safetensors(&file_path).unwrap();
    let t = loaded.get("weight").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.shape(), &[2, 3]);

    let data = t.as_slice_f32().unwrap();
    for (i, &expected) in f32_vals.iter().enumerate() {
        assert_eq!(data[i], expected, "F32 value mismatch at index {}", i);
    }
}

#[test]
fn safetensors_multiple_tensors() {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("multi.safetensors");

    let d1: Vec<f32> = vec![1.0, 2.0];
    let b1: Vec<u8> = d1.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let v1 = TensorView::new(Dtype::F32, vec![2], &b1).unwrap();

    let d2: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0];
    let b2: Vec<u8> = d2.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let v2 = TensorView::new(Dtype::F32, vec![2, 2], &b2).unwrap();

    let d3: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b3: Vec<u8> = d3.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let v3 = TensorView::new(Dtype::F32, vec![3, 2], &b3).unwrap();

    let tensor_list: Vec<(&str, &TensorView)> = vec![("a", &v1), ("b", &v2), ("c", &v3)];
    let serialized = safetensors::serialize(tensor_list, &None).unwrap();

    let mut file = File::create(&file_path).unwrap();
    file.write_all(&serialized).unwrap();

    let loaded = ModelLoader::load_safetensors(&file_path).unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded["a"].shape(), &[2]);
    assert_eq!(loaded["b"].shape(), &[2, 2]);
    assert_eq!(loaded["c"].shape(), &[3, 2]);
}

#[test]
fn corrupted_safetensors_truncated() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("truncated.safetensors");

    // Write a valid safetensors file then truncate it
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let view = TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap();
    let tensor_list: Vec<(&str, &TensorView)> = vec![("t", &view)];
    let serialized = safetensors::serialize(tensor_list, &None).unwrap();

    // Write only first 10 bytes (truncated)
    let mut file = File::create(&file_path).unwrap();
    file.write_all(&serialized[..10.min(serialized.len())]).unwrap();

    let result = ModelLoader::load_safetensors(&file_path);
    assert!(result.is_err(), "Expected error loading truncated safetensors");
}

#[test]
fn corrupted_safetensors_invalid_header() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("garbage.safetensors");

    // Write random bytes
    let mut file = File::create(&file_path).unwrap();
    file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]).unwrap();

    let result = ModelLoader::load_safetensors(&file_path);
    assert!(result.is_err(), "Expected error loading invalid safetensors");
}

#[test]
fn custom_bin_roundtrip_multiple_dtypes() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("multi_dtype.bin");

    // F32 tensor
    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let f32_bytes = bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec();
    let f32_tensor = Tensor::new(f32_bytes, vec![3], DType::F32);

    // BF16 tensor
    let bf16_vals: Vec<half::bf16> = vec![
        half::bf16::from_f32(10.0),
        half::bf16::from_f32(20.0),
    ];
    let bf16_bytes = bytemuck::cast_slice::<half::bf16, u8>(&bf16_vals).to_vec();
    let bf16_tensor = Tensor::new(bf16_bytes, vec![2], DType::BF16);

    let mut tensors = HashMap::new();
    tensors.insert("f32_w".to_string(), f32_tensor);
    tensors.insert("bf16_w".to_string(), bf16_tensor);

    ModelLoader::save_custom_bin(&file_path, &tensors).unwrap();
    let loaded = ModelLoader::load_custom_bin(&file_path).unwrap();

    assert_eq!(loaded.len(), 2);

    let lf = loaded.get("f32_w").unwrap();
    assert_eq!(lf.dtype(), DType::F32);
    assert_eq!(lf.shape(), &[3]);
    let lf_data = lf.as_slice_f32().unwrap();
    assert_eq!(lf_data, &[1.0, 2.0, 3.0]);

    let lb = loaded.get("bf16_w").unwrap();
    assert_eq!(lb.dtype(), DType::BF16);
    assert_eq!(lb.shape(), &[2]);
    // Convert to f32 and check
    let lb_f32 = lb.to_f32().unwrap();
    let lb_data = lb_f32.as_slice_f32().unwrap();
    assert!((lb_data[0] - 10.0).abs() < 0.1, "bf16 roundtrip value 0");
    assert!((lb_data[1] - 20.0).abs() < 0.1, "bf16 roundtrip value 1");
}

#[test]
fn custom_bin_empty_file_errors() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("empty.bin");

    // Write an empty file
    File::create(&file_path).unwrap();

    let result = ModelLoader::load_custom_bin(&file_path);
    assert!(result.is_err(), "Expected error loading empty custom bin file");
}
