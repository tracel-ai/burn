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
# use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue, ScaleDtype};
#
// Quantization config
let scheme = QuantScheme::default()
    .per_block([32], ScaleDtype::F16)
    .with_value(QuantValue::Q4F);
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
    .with_mode(QuantMode::Symmetric)          // Quantization mode
    .per_block([2, 16], ScaleDtype::F16)      // One scale per block of values, stored as f16
    .with_value(QuantValue::Q8S)              // Data type of quantized values, independent of how they're stored
    .with_store(QuantStore::Native);          // Storage format for quantized values
```

A scheme carries up to two scale levels, each set in any order and each owning the type its scales
are stored in. `per_tensor` is one scale for the whole tensor, which is also what a scheme with no
level set resolves to; `per_block` is one scale per block of values.

```rust
// Block scales in ue4m3, normalized by a single per-tensor f32 scale.
let scheme = QuantScheme::default()
    .per_block([16], ScaleDtype::UE4M3)
    .per_tensor(ScaleDtype::F32);
```

Two levels exist so the block scales can live in a narrow type: the per-tensor scale absorbs the
tensor's dynamic range, and the block type only has to cover the spread between blocks.

#### Quantization Mode

| Mode        | Description                                  |
| :---------- | :------------------------------------------- |
| `Symmetric` | Values are scaled symmetrically around zero. |

#### Scale Levels

| Level                        | Description                                                                                     |
| :--------------------------- | :---------------------------------------------------------------------------------------------- |
| `per_tensor(dtype)`          | A single scale for the entire tensor.                                                           |
| `per_block(block, dtype)`    | Tensor divided into blocks (1D, 2D, or higher) defined by `block`, each with its own scale.     |

Setting both nests the blocks inside the tensor: the per-tensor scale is the factor the block
scales are relative to, and it has to be `F32`.

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

| Store               | Description                                                                                                                                       |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Native`            | Each quantized value is stored directly in a native format, which doesn't require packing and unpacking.                                          |
| `PackedNative(dim)` | Multiple quantized values packed into a 32-bit integer. Argument is the dimension the tensor is packed on, starting from the innermost dimension. |
| `PackedU32(dim)`    | Multiple quantized values packed into a 32-bit integer. Argument is the dimension the tensor is packed on, starting from the innermost dimension. |

Native storage is not supported for sub-byte quantization values.

#### Scale Data Type

| Dtype   | Description                                                                            |
| :------ | :------------------------------------------------------------------------------------- |
| `F32`   | Full floating-point precision.                                                         |
| `F16`   | Half-precision floating point.                                                         |
| `BF16`  | Brain float 16-bit precision.                                                          |
| `UE4M3` | 8-bit floating point (4 exponent, 3 mantissa). Currently supported on CPU backends only. |

A narrower scale type stores less per block, but it also has a much smaller range. `UE4M3`
cannot represent a value below `2^-9`, so a scale smaller than that rounds to zero and the block
is lost. Scales stay in range when the quantized values are large enough, which in practice means
it is not a drop-in replacement for `F32` on small-magnitude weights.
