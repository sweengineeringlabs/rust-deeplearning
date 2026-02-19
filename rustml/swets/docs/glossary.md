# Glossary

**Audience**: Developers, contributors

**Autodiff** - Automatic differentiation; computing gradients of functions programmatically

**Backcast** - N-BEATS term for the model's reconstruction of its input signal

**BackwardOp** - Trait implementing the gradient computation for a single operation

**Causal Masking** - Attention mask preventing information flow from future timesteps

**DType** - Data type enum (F32, F16, BF16, Q8_0, etc.) shared with rustml-core

**Forecast** - N-BEATS term for the model's prediction of future values

**GradientTape** - Central autodiff structure recording forward operations for reverse-mode differentiation

**Layer** - Trait for neural network modules with forward pass, parameters, and train/eval modes

**N-BEATS** - Neural Basis Expansion Analysis for Time Series; a deep learning architecture for forecasting

**OHLCV** - Open, High, Low, Close, Volume; standard financial candlestick data format

**RSI** - Relative Strength Index; a momentum indicator using Wilder's EMA

**SEA** - Stratified Encapsulation Architecture; the layering convention used in rustml crates

**SIMD** - Single Instruction Multiple Data; CPU vector instructions for parallel arithmetic

**SiLU** - Sigmoid Linear Unit; activation function x * sigmoid(x)

**TCN** - Temporal Convolutional Network; causal dilated convolutions for sequential data

**TensorId** - Unique identifier linking tensors to their gradients on the tape

**TensorPool** - Thread-local buffer recycler reducing allocation pressure during training
