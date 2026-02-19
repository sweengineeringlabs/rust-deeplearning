use crate::api::error::{SwetsError, SwetsResult};
use crate::api::layer::Layer;
use std::fs;
use std::io::{Read, Cursor};
use std::path::Path;

/// A saved model checkpoint.
pub struct Checkpoint {
    pub param_data: Vec<SavedParam>,
    pub epoch: usize,
    pub best_val_loss: f32,
}

pub struct SavedParam {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Checkpoint {
    /// Create a checkpoint from a model's current parameters.
    pub fn from_model(model: &dyn Layer, epoch: usize, best_val_loss: f32) -> Self {
        let params = model.parameters();
        let param_data = params.iter().map(|p| SavedParam {
            data: p.to_vec(),
            shape: p.shape().to_vec(),
        }).collect();
        Checkpoint { param_data, epoch, best_val_loss }
    }

    /// Save checkpoint to a file using a simple binary format.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> SwetsResult<()> {
        // Format: [epoch: u64][best_val_loss: f32][num_params: u64]
        // For each param: [num_dims: u64][dim0..dimN: u64 each][num_floats: u64][f32 data...]
        let mut buf = Vec::new();
        // Write epoch
        buf.extend_from_slice(&(self.epoch as u64).to_le_bytes());
        // Write best_val_loss
        buf.extend_from_slice(&self.best_val_loss.to_le_bytes());
        // Write num params
        buf.extend_from_slice(&(self.param_data.len() as u64).to_le_bytes());
        for param in &self.param_data {
            // Write shape
            buf.extend_from_slice(&(param.shape.len() as u64).to_le_bytes());
            for &dim in &param.shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            // Write data
            buf.extend_from_slice(&(param.data.len() as u64).to_le_bytes());
            for &val in &param.data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        fs::write(path, &buf).map_err(|e| SwetsError::TrainingError(format!("save checkpoint: {e}")))?;
        Ok(())
    }

    /// Load checkpoint from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> SwetsResult<Self> {
        // Read and parse the binary format
        let data = fs::read(path).map_err(|e| SwetsError::TrainingError(format!("load checkpoint: {e}")))?;
        let mut cursor = Cursor::new(&data);

        let read_u64 = |cursor: &mut Cursor<&Vec<u8>>| -> SwetsResult<u64> {
            let mut buf = [0u8; 8];
            cursor.read_exact(&mut buf).map_err(|e| SwetsError::TrainingError(format!("read u64: {e}")))?;
            Ok(u64::from_le_bytes(buf))
        };
        let read_f32_val = |cursor: &mut Cursor<&Vec<u8>>| -> SwetsResult<f32> {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf).map_err(|e| SwetsError::TrainingError(format!("read f32: {e}")))?;
            Ok(f32::from_le_bytes(buf))
        };

        let epoch = read_u64(&mut cursor)? as usize;
        let best_val_loss = read_f32_val(&mut cursor)?;
        let num_params = read_u64(&mut cursor)? as usize;

        let mut param_data = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            let num_dims = read_u64(&mut cursor)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut cursor)? as usize);
            }
            let num_floats = read_u64(&mut cursor)? as usize;
            let mut data = Vec::with_capacity(num_floats);
            for _ in 0..num_floats {
                data.push(read_f32_val(&mut cursor)?);
            }
            param_data.push(SavedParam { data, shape });
        }

        Ok(Checkpoint { param_data, epoch, best_val_loss })
    }

    /// Load checkpoint parameters into a model.
    pub fn load_into_model(&self, model: &mut dyn Layer) -> SwetsResult<()> {
        let mut params = model.parameters_mut();
        if params.len() != self.param_data.len() {
            return Err(SwetsError::InvalidConfig(format!(
                "checkpoint has {} params, model has {}", self.param_data.len(), params.len()
            )));
        }
        for (param, saved) in params.iter_mut().zip(self.param_data.iter()) {
            let tensor = crate::api::tensor::Tensor::from_vec(saved.data.clone(), saved.shape.clone())
                .map_err(|e| SwetsError::TensorError(e))?;
            param.update_data_from(&tensor);
        }
        Ok(())
    }
}

/// Convenience: save model checkpoint to path.
pub fn save_checkpoint<P: AsRef<Path>>(model: &dyn Layer, path: P, epoch: usize, best_val_loss: f32) -> SwetsResult<()> {
    let checkpoint = Checkpoint::from_model(model, epoch, best_val_loss);
    checkpoint.save(path)
}

/// Convenience: load checkpoint from path into model.
pub fn load_checkpoint<P: AsRef<Path>>(model: &mut dyn Layer, path: P) -> SwetsResult<Checkpoint> {
    let checkpoint = Checkpoint::load(path)?;
    checkpoint.load_into_model(model)?;
    Ok(checkpoint)
}
