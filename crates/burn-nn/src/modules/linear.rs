use burn_core as burn;

use burn::config::Config;
use burn::module::Param;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, ParamId};
use burn::tensor::module::linear;
use burn::tensor::{Device, Tensor};

/// Configuration to create a [`Linear`] layer using the [init function](LinearConfig::init).
#[derive(Config, Debug)]
pub struct LinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
    /// The layout in which the linear parameters are stored.
    #[config(default = "LinearLayout::Row")]
    pub layout: LinearLayout,
}

#[derive(Config, Debug, Copy)]
/// The layout in which the linear parameters are stored.
///
/// This can have performance impacts.
pub enum LinearLayout {
    /// Parameters are stored in Row major.
    Row,
    /// Parameters are stored in Col major.
    Col,
}

/// Applies a linear transformation to the input tensor.
///
/// Should be created with [LinearConfig]
///
/// `O = IW + b`
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Linear {
    /// Matrix of shape `[d_input, d_output]` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub weight: Param<Tensor<2>>,
    /// Vector of size `d_output` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub bias: Option<Param<Tensor<1>>>,
}

impl LinearConfig {
    /// Initialize a new [`Linear`] module.
    pub fn init(&self, device: &Device) -> Linear {
        let weight = match self.layout {
            LinearLayout::Row => {
                let shape = [self.d_input, self.d_output];
                self.initializer
                    .init_with(shape, Some(self.d_input), Some(self.d_output), device)
            }
            LinearLayout::Col => {
                let shape = [self.d_output, self.d_input];

                self.initializer
                    .init_with(shape, Some(self.d_output), Some(self.d_input), device)
                    // The param is already transposed when init. We re-transpose to have
                    // [d_output, d_input] while saving.
                    .save_mapper(move |tensor| {
                        let device = tensor.device();
                        device.sync().unwrap();
                        let tensor = tensor.transpose();
                        device.sync().unwrap();
                        tensor
                    })
                    // When loading from record we have to transpose.
                    .load_mapper(move |tensor| {
                        let device = tensor.device();
                        device.sync().unwrap();
                        let tensor = tensor.transpose();
                        device.sync().unwrap();

                        tensor
                    })
                    // When loading from initialization, we have to transpose.
                    .init_mapper(|tensor| {
                        let device = tensor.device();
                        device.sync().unwrap();
                        let tensor = tensor.transpose();
                        device.sync().unwrap();
                        tensor
                    })
            }
        };
        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Linear {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Arguments
    ///
    /// - `input` - The input tensor of shape `[..., d_input]`.
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    ///
    /// # Returns
    ///
    /// The transformed tensor of shape `[..., d_output]`.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        linear(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        )
    }
}

impl ModuleDisplay for Linear {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.weight.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("bias", &self.bias.is_some())
            .optional()
    }
}

/// Configuration to create a [`LinearTernary`] layer using the [init function](LinearTernaryConfig::init).
#[derive(Config, Debug)]
pub struct LinearTernaryConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = true)]
    pub bias: bool,
    /// Weight quantization threshold. `None` uses the mean absolute weight value
    /// (BitNet b1.58 rule: <https://arxiv.org/abs/2402.17764> §3.1).
    /// `Some(0.0)` maps all weights to ±1 (no zeros).
    #[config(default = "None")]
    pub threshold: Option<f32>,
    /// The type of function used to initialize neural network parameters before quantization.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a linear transformation using weights quantized to {-1, 0, +1}.
///
/// A drop-in replacement for [`Linear`] for ternary-weight models (BitNet b1.58, et al.).
/// Weights are quantized at init time and remain frozen as {-1, 0, +1}. The forward pass is
/// numerically identical to [`Linear`] — the sparsity benefit is realized by backends that
/// specialize `linear()` on ternary weight tensors.
///
/// Quantization rule:
/// ```text
/// w_ternary = sign(w)  if |w| > threshold
///             0        otherwise
/// ```
/// where `threshold` defaults to `mean(|w|)` across the weight matrix (BitNet b1.58 §3.1).
///
/// Should be created with [`LinearTernaryConfig`].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LinearTernary {
    /// Ternary weight matrix of shape `[d_input, d_output]`. All values are in {-1, 0, +1}.
    pub weight: Param<Tensor<2>>,
    /// Optional bias vector of size `d_output`.
    pub bias: Option<Param<Tensor<1>>>,
}

impl LinearTernaryConfig {
    /// Initialize a new [`LinearTernary`] module.
    pub fn init(&self, device: &Device) -> LinearTernary {
        let shape = [self.d_input, self.d_output];
        let raw: Param<Tensor<2>> = self
            .initializer
            .init_with(shape, Some(self.d_input), Some(self.d_output), device);

        let w = raw.val();
        let abs_w = w.clone().abs();
        let threshold: f32 = match self.threshold {
            Some(t) => t,
            None => abs_w.clone().mean().into_scalar::<f32>(),
        };
        let zero_mask = abs_w.lower_equal_elem(threshold);
        let ternary = w.sign().mask_fill(zero_mask, 0.0_f32);
        let weight = Param::initialized(ParamId::new(), ternary);

        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        LinearTernary { weight, bias }
    }
}

impl LinearTernary {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        linear(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        )
    }
}

impl ModuleDisplay for LinearTernary {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.weight.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("bias", &self.bias.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::ParamId;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::ElementConversion;
    use burn::tensor::Tolerance;
    use burn::tensor::{Shape, TensorData};
    type FT = f32;

    #[test]
    fn initializer_default() {
        let device = Device::default();
        device.seed(0);

        let config = LinearConfig::new(5, 5);
        let k = (1.0 / config.d_input as f64).sqrt().elem::<FT>();
        let linear = config.init(&device);

        assert_eq!(
            config.initializer,
            Initializer::KaimingUniform {
                gain: 1.0 / 3.0f64.sqrt(),
                fan_out_only: false
            }
        );
        linear.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        let device = Device::default();
        device.seed(0);

        let config = LinearConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let linear = config.init(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        linear.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<f32, _>(linear.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let device = Device::default();
        device.seed(0);

        let value = 2.;
        let config = LinearConfig::new(2, 3)
            .with_initializer(Initializer::Constant { value })
            .with_bias(false);
        let linear = config.init(&device);

        let input = Tensor::<2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<2>::from_data([[4., 4., 4.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let device = Device::default();
        device.seed(0);

        let device = Device::default();

        let value = 2.;
        let config = LinearConfig::new(2, 3).with_initializer(Initializer::Constant { value });
        let linear = config.init(&device);

        let input = Tensor::<2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<2>::from_data([[6., 6., 6.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }

    #[test]
    fn test_linear_1d() {
        let device = Device::default();
        device.seed(0);

        let value = 2.;
        let config = LinearConfig::new(2, 3).with_initializer(Initializer::Constant { value });
        let linear = config.init(&device);

        let input_1d = Tensor::<1>::ones(Shape::new([2]), &device);
        let input_2d = Tensor::<2>::ones(Shape::new([1, 2]), &device);

        let result_1d = linear.forward(input_1d).unsqueeze::<2>();
        let result_2d = linear.forward(input_2d);

        assert_eq!(result_1d.into_data(), result_2d.into_data());
    }

    #[test]
    fn display() {
        let config = LinearConfig::new(3, 5);
        let linear = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{linear}"),
            "Linear {d_input: 3, d_output: 5, bias: true, params: 20}"
        );
    }

    #[test]
    fn layout() {
        let device = Default::default();
        let config = LinearConfig::new(6, 12).with_layout(LinearLayout::Col);
        let linear = config.init(&device);

        assert_eq!(linear.weight.dims(), [6, 12], "Shape is as configured");

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();

        // We go through serialization to trigger the mappers..
        let record = linear.into_record();
        let data = recorder.record(record, ()).unwrap();
        let record = recorder.load(data.clone(), &device).unwrap();

        let config = LinearConfig::new(12, 6).with_layout(LinearLayout::Row);
        let linear_row = config.init(&device).load_record(record);

        assert_eq!(
            linear_row.weight.dims(),
            [12, 6],
            "Shape should be transposed"
        );

        let record = recorder.load(data.clone(), &device).unwrap();
        let config = LinearConfig::new(6, 12).with_layout(LinearLayout::Col);
        let linear_col = config.init(&device).load_record(record);

        assert_eq!(
            linear_col.weight.dims(),
            [6, 12],
            "Shape should be as configured"
        );

        // We go through serialization to trigger the mappers.
        //
        // The test will fail if the mapper is not correctly given to the module after loading a
        // record.
        let record = linear_col.into_record();
        let data = recorder.record(record, ()).unwrap();

        let record = recorder.load(data, &device).unwrap();
        let config = LinearConfig::new(6, 12).with_layout(LinearLayout::Col);
        let linear_col = config.init(&device).load_record(record);

        assert_eq!(
            linear_col.weight.dims(),
            [6, 12],
            "Shape should be as configured"
        );
    }

    #[test]
    fn col_row_same_result() {
        let device = Default::default();
        let config_col = LinearConfig::new(6, 12).with_layout(LinearLayout::Col);
        let linear_col = config_col.init(&device);
        let signal = Tensor::<2>::random([8, 6], burn::tensor::Distribution::Default, &device);
        let value = linear_col.forward(signal.clone());

        let data_1 = value.into_data();

        let weights = linear_col.weight.val().into_data();
        let weights = Tensor::from_data(weights, &device);

        let linear = Linear {
            weight: Param::initialized(ParamId::new(), weights),
            bias: linear_col
                .bias
                .map(|b| Param::initialized(ParamId::new(), b.val())),
        };

        let value = linear.forward(signal);
        let data_2 = value.into_data();

        data_1.assert_approx_eq::<f32>(&data_2, Default::default());
    }

    // --- LinearTernary tests ---

    #[test]
    fn ternary_weights_are_ternary() {
        let device = Device::default();
        device.seed(0);
        let layer = LinearTernaryConfig::new(8, 16).init(&device);
        let values: Vec<f32> = layer.weight.to_data().to_vec().unwrap();
        for v in &values {
            assert!(
                *v == -1.0 || *v == 0.0 || *v == 1.0,
                "weight {v} is not in {{-1, 0, 1}}"
            );
        }
    }

    #[test]
    fn ternary_output_shape() {
        let device = Device::default();
        device.seed(0);
        let layer = LinearTernaryConfig::new(4, 8).init(&device);
        let input = Tensor::<2>::ones(Shape::new([2, 4]), &device);
        let out = layer.forward(input);
        assert_eq!(out.dims(), [2, 8]);
    }

    #[test]
    fn ternary_threshold_zero_no_zeros() {
        // threshold=0.0 → |w| > 0 for all Kaiming-initialised weights → all ±1
        let device = Device::default();
        device.seed(0);
        let layer = LinearTernaryConfig::new(8, 16)
            .with_threshold(Some(0.0))
            .init(&device);
        let values: Vec<f32> = layer.weight.to_data().to_vec().unwrap();
        assert!(
            values.iter().all(|v| *v == 1.0 || *v == -1.0),
            "expected no zero weights with threshold=0.0"
        );
    }

    #[test]
    fn ternary_threshold_large_all_zeros() {
        let device = Device::default();
        device.seed(0);
        let layer = LinearTernaryConfig::new(4, 4)
            .with_threshold(Some(1e9))
            .init(&device);
        let values: Vec<f32> = layer.weight.to_data().to_vec().unwrap();
        assert!(
            values.iter().all(|v| *v == 0.0),
            "expected all-zero weights with threshold=1e9"
        );
    }

    #[test]
    fn ternary_forward_matches_manual() {
        // 2-input, 2-output layer with hand-crafted weights and no bias.
        // weight = [[1, -1], [0, 1]]  (d_input=2, d_output=2)
        // input  = [[2, 3]]
        // expected output = [[2*1 + 3*0, 2*(-1) + 3*1]] = [[2, 1]]
        let device = Device::default();
        let w_data: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0];
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(w_data, Shape::new([2, 2])),
            &device,
        );
        let layer = LinearTernary {
            weight: Param::initialized(ParamId::new(), w),
            bias: None,
        };
        let input = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![2.0_f32, 3.0], Shape::new([1, 2])),
            &device,
        );
        let expected = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![2.0_f32, 1.0], Shape::new([1, 2])),
            &device,
        );
        let out = layer.forward(input);
        out.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn ternary_no_bias() {
        let device = Device::default();
        let layer = LinearTernaryConfig::new(4, 4)
            .with_bias(false)
            .init(&device);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn ternary_display() {
        let config = LinearTernaryConfig::new(3, 5);
        let layer = config.init(&Default::default());
        assert_eq!(
            alloc::format!("{layer}"),
            "LinearTernary {d_input: 3, d_output: 5, bias: true, params: 20}"
        );
    }
}
