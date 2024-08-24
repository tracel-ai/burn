# Quantization (Beta)

Quantization techniques perform computations and store tensors in lower precision data types like
8-bit integer instead of floating point precision. There are multiple approaches to quantize a deep
learning model categorized as:

- Post-training quantization (PTQ)
- Quantization aware training (QAT)

In post-training quantization, the model is trained in floating point precision and later converted
to the lower precision data type.

There are two types of post-training quantization:

1. Static quantization: quantizes the weights and activations of the model. Quantizing the
   activations statically requires data to be calibrated (i.e., recording the activation values to
   compute the optimal quantization parameters with representative data).
1. Dynamic quantization: quantized the weights ahead of time (like static quantization) but the
   activations are dynamically at runtime.

Sometimes post-training quantization is not able to achieve acceptable task accuracy. This is where
quantization aware training comes into play, as it models the effects of quantization during
training. Quantization errors are thus modeled in the forward and backward passes using fake
quantization modules, which helps the model learn representations that are more robust to the
reduction in precision.

<div class="warning">

Quantization support in Burn is currently in active development.

It supports the following modes on some backends:

- Static per-tensor quantization to signed 8-bit integer (`i8`)

No integer operations are currently supported, which means tensors are dequantized to perform the
operations in floating point precision.

</div>

## Module Quantization

Quantizing the weights of your model after training is quite simple. We have access to the weight
tensors and can collect their statistics, such as the min and max value when using
`MinMaxCalibration`, to compute the quantization parameters.

```rust , ignore
# use burn::module::Quantizer;
# use burn::tensor::quantization::{MinMaxCalibration, QuantizationScheme, QuantizationType};
#
// Quantization config
let mut quantizer = Quantizer {
    calibration: MinMaxCalibration {},
    scheme: QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8),
};

// Quantize the weights
let model = model.quantize_weights(&mut quantizer);
```

> Given that all operations are currently performed in floating point precision, it might be wise to
> dequantize the module parameters before inference. This allows us to save disk space by storing
> the model in reduced precision while preserving the inference speed.
>
> This can easily be implemented with a `ModuleMapper`.
>
> ```rust, ignore
> # use burn::module::{ModuleMapper, ParamId};
> # use burn::tensor::{backend::Backend, Tensor};
> #
> /// Module mapper used to dequantize the model params being loaded.
> pub struct Dequantize {}
>
> impl<B: Backend> ModuleMapper<B> for Dequantize {
>     fn map_float<const D: usize>(
>         &mut self,
>         _id: &ParamId,
>         tensor: Tensor<B, D>,
>     ) -> Tensor<B, D> {
>         tensor.dequantize()
>     }
> }
>
> // Load saved quantized model in floating point precision
> model = model
>     .load_file(file_path, recorder, &device)
>     .expect("Should be able to load the quantized model weights")
>     .map(&mut Dequantize {});
> ```

### Calibration

Calibration is the step during quantization where the range of all floating-point tensors is
computed. This is pretty straightforward for weights since the actual range is known at
_quantization-time_ (weights are static), but activations require more attention.

To compute the quantization parameters, Burn supports the following `Calibration` methods.

| Method              | Description                                                                      |
| :------------------ | :------------------------------------------------------------------------------- |
| `MinMaxCalibration` | Computes the quantization range mapping based on the running min and max values. |

### Quantization Scheme

A quantization scheme defines the quantized type, quantization granularity and range mapping
technique.

Burn currently supports the following `QuantizationType` variants.

| Type    | Description                        |
| :------ | :--------------------------------- |
| `QInt8` | 8-bit signed integer quantization. |

Quantization parameters are defined based on the range of values to represent and can typically be
calculated for the layer's entire weight tensor with per-tensor quantization or separately for each
channel with per-channel quantization (commonly used with CNNs).

Burn currently supports the following `QuantizationScheme` variants.

| Variant              | Description                                                                                                    |
| :------------------- | :------------------------------------------------------------------------------------------------------------- |
| `PerTensorAffine`    | Computes the quantization parameters for the whole tensor and applies an affine range mapping with zero point. |
| `PerTensorSymmetric` | Computes the quantization parameters for the whole tensor and applies a scale range mapping centered around 0. |
