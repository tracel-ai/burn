use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::PadMode;

/// Configuration to create a [1D Lp pooling](LpPool1d) layer.
#[derive(Config, Debug)]
pub struct LpPool1dConfig {
    /// The size of the kernel.
    pub kernel_size: usize,
    /// The stride. Defaults to the kernel size.
    #[config(default = "kernel_size")]
    pub stride: usize,
    /// The exponent used by the Lp norm.
    #[config(default = "2.0")]
    pub p: f64,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Applies one-dimensional Lp pooling to `[batch, channels, length]` tensors.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LpPool1d {
    /// The size of the pooling window.
    pub kernel_size: usize,
    /// The step between pooling windows.
    pub stride: usize,
    /// The exponent used by the Lp norm.
    pub p: f64,
    /// Whether to round the output size up and include a partial final window.
    pub ceil_mode: bool,
}

impl ModuleDisplay for LpPool1d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("kernel_size", &self.kernel_size)
            .add("stride", &self.stride)
            .add("p", &self.p)
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl LpPool1dConfig {
    /// Initialize a new 1D Lp pooling module.
    pub fn init(&self) -> LpPool1d {
        LpPool1d {
            kernel_size: self.kernel_size,
            stride: self.stride,
            p: self.p,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl LpPool1d {
    /// Applies Lp pooling over the last input dimension.
    ///
    /// The input shape is `[batch_size, channels, length_in]`; the output shape is
    /// `[batch_size, channels, length_out]`. For partial windows in ceil mode,
    /// only input elements are included in the norm.
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        assert!(
            self.p.is_finite() && self.p > 0.0,
            "Lp pooling requires a finite p > 0"
        );
        assert!(
            self.kernel_size > 0 && self.stride > 0,
            "kernel size and stride must be positive"
        );

        let length = input.dims()[2];
        let output_length = output_size(length, self.kernel_size, self.stride, self.ceil_mode);
        let padded_length = (output_length - 1) * self.stride + self.kernel_size;
        let padded = if padded_length > length {
            input.pad([(0, padded_length - length)], PadMode::Constant(0.0))
        } else {
            input
        };

        padded
            .unfold::<4, _>(2, self.kernel_size, self.stride)
            .abs()
            .powf_scalar(self.p)
            .sum_dim(3)
            .powf_scalar(1.0 / self.p)
            .squeeze_dim(3)
    }
}

fn output_size(input: usize, kernel: usize, stride: usize, ceil_mode: bool) -> usize {
    assert!(
        input > 0,
        "Lp pooling does not support an empty input dimension"
    );
    if ceil_mode {
        if input <= kernel {
            1
        } else {
            let output = (input - kernel).div_ceil(stride) + 1;
            if output > 1 && (output - 1) * stride >= input {
                output - 1
            } else {
                output
            }
        }
    } else {
        assert!(
            input >= kernel,
            "kernel size must not exceed the input size"
        );
        (input - kernel) / stride + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData, Tolerance};

    #[test]
    fn computes_l2_norms() {
        let device = Default::default();
        let input = Tensor::<3>::from_data([[[3.0f32, -4.0, 5.0, 12.0]]], &device);
        let output = LpPool1dConfig::new(2).init().forward(input);

        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[5.0, 13.0]]]), Tolerance::default());
    }

    #[test]
    fn computes_l1_norms() {
        let device = Default::default();
        let input = Tensor::<3>::from_data([[[3.0f32, -4.0, 5.0, -12.0]]], &device);
        let output = LpPool1dConfig::new(2).with_p(1.0).init().forward(input);

        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[7.0, 17.0]]]), Tolerance::default());
    }

    #[test]
    fn preserves_autodiff_gradients() {
        let device = Device::default().autodiff();
        let input = Tensor::<3>::from_data([[[3.0f32, 4.0]]], &device).require_grad();
        let output = LpPool1dConfig::new(2).init().forward(input.clone());
        let gradients = output.sum().backward();

        input
            .grad(&gradients)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[0.6, 0.8]]]), Tolerance::default());
    }

    #[test]
    fn ceil_mode_includes_partial_window_without_counting_padding() {
        let device = Default::default();
        let input = Tensor::<3>::from_data([[[3.0f32, 4.0, 12.0]]], &device);
        let output = LpPool1dConfig::new(2)
            .with_stride(2)
            .with_ceil_mode(true)
            .init()
            .forward(input);

        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[5.0, 12.0]]]), Tolerance::default());
    }

    #[test]
    fn ceil_mode_does_not_create_windows_beyond_the_input() {
        let device = Default::default();
        let input = Tensor::<3>::from_data([[[3.0f32, 4.0, 12.0]]], &device);
        let output = LpPool1dConfig::new(2)
            .with_stride(10)
            .with_ceil_mode(true)
            .init()
            .forward(input);

        assert_eq!(output.dims(), [1, 1, 1]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[5.0]]]), Tolerance::default());
    }
}
