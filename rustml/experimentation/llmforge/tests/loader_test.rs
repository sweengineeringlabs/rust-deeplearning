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
