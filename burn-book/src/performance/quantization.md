# Quantization

Quantization techniques perform computations and store tensors in lower precision data types like
8-bit integer instead of floating point precision. There are multiple approaches to quantize a deep
learning model categorized as:

- Post-training quantization (PTQ)
- Quantization aware training (QAT)

In post-training quantization, the model is trained in floating point precision and later converted
to the lower precision data type. There are two types of post-training quantization:

1. Static quantization: quantizes the weights and activations of the model. Quantizing the
   activations statically requires data to be calibrated (i.e., recording the activation values to
   compute the optimal quantization parameters with representative data).
1. Dynamic quantization: quantized the weights ahead of time (like static quantization) but the
   activations are dynamically at runtime.

Sometimes post-training quantization is not able to achieve acceptable task accuracy. In general,
this is where quantization-aware training (QAT) can be used: during training, fake-quantization
modules are inserted in the forward and backward passes to simulate quantization effects, allowing
the model to learn representations that are more robust to reduced precision.

Burn does not currently support QAT. Only post-training quantization (PTQ) is implemented at this
time.

<div class="warning">

Quantization support in Burn is currently in active development.

It supports the following PTQ modes on some backends:

- Per-tensor and per-block quantization to 8-bit, 4-bit and 2-bit representations

No integer operations are currently supported, which means tensors are dequantized to perform the
operations in floating point precision.

</div>

## Module Quantization

Quantizing the weights of your model after training is quite simple. We have access to the weight
tensors and can collect their statistics, such as the min and max value when using
`MinMaxCalibration`, to compute the quantization parameters.

```rust , ignore
# use burn::module::Quantizer;
# use burn::tensor::quantization::{Calibration, QuantLevel, QuantParam, QuantScheme, QuantValue};
#
// Quantization config
let scheme = QuantScheme::default()
    .with_level(QuantLevel::Block(32))
    .with_value(QuantValue::Q4F)
    .with_param(QuantParam::F16);
let mut quantizer = Quantizer {
    calibration: Calibration::MinMax,
    scheme,
};

// Quantize the weights
let model = model.quantize_weights(&mut quantizer);
```

### Calibration

Calibration is the step during quantization where the range of all floating-point tensors is
computed. This is pretty straightforward for weights since the actual range is known at
_quantization-time_ (weights are static), but activations require more attention.

To compute the quantization parameters, Burn supports the following `Calibration` methods.

| Method   | Description                                                                      |
| :------- | :------------------------------------------------------------------------------- |
| `MinMax` | Computes the quantization range mapping based on the running min and max values. |

### Quantization Scheme

A quantization scheme defines how an input is quantized, including the representation of quantized
values, storage format, granularity, and how the values are scaled.

```rust
let scheme = QuantScheme::default()
    .with_mode(QuantMode::Symmetric)         // Quantization mode
    .with_level(QuantLevel::block([2, 16]))  // Granularity (per-tensor or per-block)
    .with_value(QuantValue::Q8S)             // Data type of quantized values, independent of how they're stored
    .with_store(QuantStore::Native)          // Storage format for quantized values
    .with_param(QuantParam::F16);            // Precision for quantization parameters
```

#### Quantization Mode

| Mode        | Description                                  |
| :---------- | :------------------------------------------- |
| `Symmetric` | Values are scaled symmetrically around zero. |

#### Quantization Level

| Level                          | Description                                                                                                  |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------- |
| `Tensor`                       | A single quantization parameter set for the entire tensor.                                                   |
| `Block(block_size: BlockSize)` | Tensor divided into blocks (1D, 2D, or higher) defined by block_size, each with its own quantization params. |

#### Quantization Value

| Value  | Bits | Description                                   |
| :----- | :--: | :-------------------------------------------- |
| `Q8F`  |  8   | 8-bit full-range quantization                 |
| `Q4F`  |  4   | 4-bit full-range quantization                 |
| `Q2F`  |  2   | 2-bit full-range quantization                 |
| `Q8S`  |  8   | 8-bit symmetric quantization                  |
| `Q4S`  |  4   | 4-bit symmetric quantization                  |
| `Q2S`  |  2   | 2-bit symmetric quantization                  |
| `E5M2` |  8   | 8-bit floating-point (5 exponent, 2 mantissa) |
| `E4M3` |  8   | 8-bit floating-point (4 exponent, 3 mantissa) |
| `E2M1` |  4   | 4-bit floating-point (2 exponent, 1 mantissa) |

#### Quantization Store

| Store    | Description                                             |
| :------- | :------------------------------------------------------ |
| `Native` | Each quantized value stored directly in memory.         |
| `U32`    | Multiple quantized values packed into a 32-bit integer. |

Native storage is not supported for sub-byte quantization values.

#### Quantization Parameters Precision

| Param  | Description                    |
| :----- | :----------------------------- |
| `F32`  | Full floating-point precision. |
| `F16`  | Half-precision floating point. |
| `BF16` | Brain float 16-bit precision.  |
