use llmforge::loader::ModelLoader;
use llmforge::core::tensor::{Tensor, DType};
use safetensors::tensor::{TensorView, Dtype};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use std::collections::HashMap;

#[test]
fn test_safetensors_loader() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");
    
    // Create dummy safetensors file
    {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|val| val.to_ne_bytes()).collect();
        
        // TensorView doesn't own data, just points to it
        let view = TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("Invalid view");
        let tensors = HashMap::from([("test_tensor".to_string(), view)]);
        
        // Serialize
        // SafeTensors::serialize takes &[(&str, &TensorView)]
        let tensor_list: Vec<(&str, &TensorView)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let serialized = safetensors::serialize(tensor_list, &None).expect("Serialization failed");
        
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&serialized).unwrap();
    }
    
    // Load back
    let loaded_tensors = ModelLoader::load_safetensors(&file_path).expect("Failed to load safetensors");
    
    assert!(loaded_tensors.contains_key("test_tensor"));
    let tensor = loaded_tensors.get("test_tensor").unwrap();
    
    assert_eq!(tensor.shape(), &[2, 2]);
    let slice = tensor.as_slice_f32().expect("Failed to access as f32");
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[3], 4.0);
}
