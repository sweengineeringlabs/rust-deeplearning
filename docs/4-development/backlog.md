# Backlog

## Model Format Support

- [ ] **ONNX runtime/loading is not implemented** — No `.onnx` model loading exists in the production crates. SafeTensors and GGUF are supported; ONNX is not. Requires adding an ONNX parser or integrating an ONNX runtime (e.g., `ort` crate) to load and execute ONNX graphs.
