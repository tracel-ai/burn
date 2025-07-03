use alloc::format;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param};
use crate::nn::Initializer;
use crate::nn::PaddingConfig2d;
use crate::tensor::Tensor;
use crate::tensor::backend::Backend;
use crate::tensor::module::conv2d;
use crate::tensor::ops::ConvOptions;

use crate::nn::conv::checks;

/// Configuration to create a [2D convolution](Conv2d) layer, using the [init function](Conv2dConfig::init).
#[derive(Config, Debug)]
pub struct Conv2dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// Controls the connections between input and output channels.
    #[config(default = "1")]
    pub groups: usize,
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a 2D convolution over input tensors.
///
/// Should be created with [Conv2dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Conv2d<B: Backend> {
    /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
    pub weight: Param<Tensor<B, 4>>,
    /// Tensor of shape `[channels_out]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Stride of the convolution.
    pub stride: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 2],
    /// Spacing between kernel elements.
    pub dilation: [usize; 2],
    /// Controls the connections between input and output channels.
    pub groups: usize,
    /// The padding configuration.
    pub padding: Ignored<PaddingConfig2d>,
}

impl Conv2dConfig {
    /// Initialize a new [conv2d](Conv2d) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2d<B> {
        checks::checks_channels_div_groups(self.channels[0], self.channels[1], self.groups);
        if self.padding == PaddingConfig2d::Same {
            checks::check_same_padding_support(&self.kernel_size);
        }

        let shape = [
            self.channels[1],
            self.channels[0] / self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        ];

        let k = self.kernel_size.iter().product::<usize>();
        let fan_in = self.channels[0] / self.groups * k;
        let fan_out = self.channels[1] / self.groups * k;

        let weight = self
            .initializer
            .init_with(shape, Some(fan_in), Some(fan_out), device);
        let mut bias = None;

        if self.bias {
            bias = Some(self.initializer.init_with(
                [self.channels[1]],
                Some(fan_in),
                Some(fan_out),
                device,
            ));
        }

        Conv2d {
            weight,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            padding: Ignored(self.padding.clone()),
            groups: self.groups,
        }
    }
}

impl<B: Backend> ModuleDisplay for Conv2d<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        // Since padding does not implement ModuleDisplay, we need to format it manually.
        let padding_formatted = format!("{}", &self.padding);

        // Format the stride, kernel_size and dilation as strings, formatted as arrays instead of indexed.
        let stride = format!("{:?}", self.stride);
        let kernel_size = format!("{:?}", self.kernel_size);
        let dilation = format!("{:?}", self.dilation);

        content
            .add("stride", &stride)
            .add("kernel_size", &kernel_size)
            .add("dilation", &dilation)
            .add("groups", &self.groups)
            .add("padding", &padding_formatted)
            .optional()
    }
}

impl<B: Backend> Conv2d<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [conv2d](crate::tensor::module::conv2d) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels_in, height_in, width_in]`
    /// - output: `[batch_size, channels_out, height_out, width_out]`
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let padding =
            self.padding
                .calculate_padding_2d(height_in, width_in, &self.kernel_size, &self.stride);
        conv2d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new(self.stride, padding, self.dilation, self.groups),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{ElementConversion, Tolerance};

    use super::*;
    use crate::TestBackend;
    use crate::tensor::TensorData;
    type FT = FloatElem<TestBackend>; // Float test

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let k = (config.channels[0] * config.kernel_size[0] * config.kernel_size[1]) as f64;
        let k = (config.groups as f64 / k).sqrt().elem::<FT>();
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        conv.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = Conv2dConfig::new([5, 2], [5, 5]).with_initializer(Initializer::Zeros);
        let device = Default::default();
        let conv = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        conv.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<FT, _>(conv.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn initializer_fan_out() {
        TestBackend::seed(0);

        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: true, // test that fan_out is passed to `init_with()`
        };
        let device = Default::default();
        let config = Conv2dConfig::new([5, 1], [5, 5]).with_initializer(init.clone());
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    fn initializer_fan_with_groups_is_valid() {
        TestBackend::seed(0);

        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: true,
        };
        let device = Default::default();
        let config = Conv2dConfig::new([4, 4], [1, 1])
            .with_initializer(init.clone())
            .with_groups(4);
        let _ = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, init);
    }

    #[test]
    #[should_panic = "Both channels must be divisible by the number of groups."]
    fn channels_with_groups_is_invalid() {
        let device = Default::default();
        let config = Conv2dConfig::new([1, 4], [1, 1]).with_groups(4);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    #[should_panic = "Same padding with an even kernel size is not supported"]
    fn same_with_even_kernel_is_invalid() {
        let device = Default::default();
        let config = Conv2dConfig::new([4, 4], [2, 2]).with_padding(PaddingConfig2d::Same);
        let _ = config.init::<TestBackend>(&device);
    }

    #[test]
    fn display() {
        let config = Conv2dConfig::new([5, 1], [5, 5]);
        let conv = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", conv),
            "Conv2d {stride: [1, 1], kernel_size: [5, 5], dilation: [1, 1], groups: 1, padding: Valid, params: 126}"
        );
    }

    #[test]
    #[should_panic = "Number of channels in input tensor and input channels of convolution must be equal. got: 4, expected: 5"]
    fn input_channels_mismatch() {
        let config = Conv2dConfig::new([5, 3], [3, 3]);
        let conv = config.init::<TestBackend>(&Default::default());

        let input = Tensor::<TestBackend, 4>::zeros([1, 4, 10, 10], &Default::default());
        let _ = conv.forward(input);
    }

    #[rustfmt::skip] // param values are too long
    fn conv2d() -> Conv2d<TestBackend> {
        let device = Default::default();
        let weight = TensorData::new(
            vec![0.048065186, -0.3059082, -0.10345459, -0.34643555, -0.20788574, -0.021072388, 0.13745117, -0.05102539, 0.024536133, -0.16479492, -0.19519043, 0.27270508, 0.17700195, -0.33764648, -0.08239746, -0.27929688, 0.17321777, -0.1315918, 0.04574585, -0.17980957, -0.33569336, 0.27612305, 0.30004883, -0.28979492, -0.17297363, -0.021759033, -0.27148438, 0.005657196, 0.29956055, -0.06958008, -0.29345703, -0.14440918, 0.10827637, -0.13305664, -0.20239258, 0.24890137, -0.1541748, -0.20019531, -0.2854004, 0.17016602, 0.07861328, -0.09075928, 0.30908203, -0.00013422966, 0.29589844, 0.15258789, -0.25708008, 0.20422363, -0.2529297, 0.07891846, -0.19506836, 0.23571777, 0.27124023, 0.17370605, -0.16992188, -0.23522949, 0.14648438, -0.09576416, -0.18310547, 0.21044922, -0.08911133, -0.2541504, -0.2775879, -0.2064209, -0.16271973, -0.048919678, -0.03555298, -0.11639404, 0.09661865, -0.10241699, 0.08929443, 0.2866211],
            [8, 1, 3, 3],
        );
        let bias = TensorData::from([0.082336426, -0.049591064, 0.0031795502, 0.00095653534, 0.02357483, 0.005569458, 0.07525635, 0.056396484]);

        let cfg = Conv2dConfig::new([1, 8], [3, 3]).with_padding(PaddingConfig2d::Valid);
        Conv2d {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
            stride: cfg.stride,
            kernel_size: cfg.kernel_size,
            dilation: cfg.dilation,
            padding: Ignored(cfg.padding.clone()),
            groups: cfg.groups,
        }
    }

    #[rustfmt::skip] // param values are too long
    fn batchnorm_2d() -> crate::nn::BatchNorm<TestBackend, 2> {
        let device = Default::default();
        let gamma = Param::from_data(TensorData::from([1.0048828, 0.9902344, 1.0185547, 0.97558594, 1.0097656, 0.97802734, 1.0009766, 1.0146484]), &device);
        let beta = Param::from_data(TensorData::from([0.026290894, 0.0007505417, 0.006134033, 0.02418518, 0.07373047, 0.020507813, 0.01902771, 0.02003479]), &device);
        let running_mean = crate::module::RunningState::new(Tensor::from_floats([0.029159546, -0.08673096, -0.03894043, -0.01108551, 0.032440186, 0.03237915, 0.013839722, 0.04397583], &device));
        let running_var = crate::module::RunningState::new(Tensor::from_floats([0.67089844, 0.29956055, 0.5209961, 0.1862793, 0.30419922, 0.21313477, 0.7504883, 0.26342773], &device));
        crate::nn::BatchNorm::<_, 2> {
            gamma,
            beta,
            running_mean,
            running_var,
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    #[test]
    fn conv2d_block_regression() {
        let device = Default::default();

        let conv = conv2d();
        let norm = batchnorm_2d();
        let activation = crate::nn::Gelu::new();

        let x = Tensor::<TestBackend, 4>::full([1, 1, 28, 28], -0.42421296, &device);

        // ConvBlock
        let x = conv.forward(x);
        let x = norm.forward(x);
        let x = activation.forward(x);

        let expected: Vec<f32> = [
            0.36432067f32,
            0.34909567,
            0.30684796,
            0.13217466,
            -0.018471397,
            -0.1389876,
            0.39402074,
            0.12394252,
        ]
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, 676))
        .collect();
        let expected = TensorData::new(expected, [1, 8, 26, 26]);

        x.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
