use crate::modules::bounds::{self, Windows};
use burn::instrument::{Bounds, compute};
use burn_core as burn;

use super::output_size;
use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::PadMode;

/// Configuration to create a [2D Lp pooling](LpPool2d) layer.
#[derive(Config, Debug)]
pub struct LpPool2dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The strides. Defaults to the kernel size.
    #[config(default = "kernel_size")]
    pub strides: [usize; 2],
    /// The exponent used by the Lp norm.
    #[config(default = "2.0")]
    pub p: f64,
    /// If true, use ceiling instead of floor for output size calculation.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Applies two-dimensional Lp pooling to `[batch, channels, height, width]` tensors.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LpPool2d {
    /// The height and width of the pooling window.
    pub kernel_size: [usize; 2],
    /// The vertical and horizontal steps between pooling windows.
    pub stride: [usize; 2],
    /// The exponent used by the Lp norm.
    pub p: f64,
    /// Whether to round output sizes up and include partial final windows.
    pub ceil_mode: bool,
}

impl ModuleDisplay for LpPool2d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("kernel_size", &alloc::format!("{:?}", self.kernel_size))
            .add("stride", &alloc::format!("{:?}", self.stride))
            .add("p", &self.p)
            .add("ceil_mode", &self.ceil_mode)
            .optional()
    }
}

impl LpPool2dConfig {
    /// Initialize a new 2D Lp pooling module.
    pub fn init(&self) -> LpPool2d {
        LpPool2d {
            kernel_size: self.kernel_size,
            stride: self.strides,
            p: self.p,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl LpPool2d {
    /// The floor for an Lp pool alone: a resampling to the size its windows cover, reading the
    /// positions they read. None where the pool rejects its input or its windows.
    fn bounds(&self, input: &Tensor<4>) -> Option<Bounds> {
        let input_size: [usize; 2] = bounds::spatial(input);
        let empty = input_size.contains(&0);
        let overhangs = !self.ceil_mode && (0..2).any(|i| self.kernel_size[i] > input_size[i]);
        if empty || self.kernel_size.contains(&0) || self.stride.contains(&0) || overhangs {
            return None;
        }
        let output_size: [usize; 2] = core::array::from_fn(|i| {
            output_size(
                input_size[i],
                self.kernel_size[i],
                self.stride[i],
                self.ceil_mode,
            )
        });
        let windows = Windows {
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding_begin: [0; 2],
            dilation: [1; 2],
        };
        Some(windows.over(input, output_size).resampling(input))
    }

    /// Applies Lp pooling over the last two input dimensions.
    ///
    /// The input shape is `[batch_size, channels, height_in, width_in]`; the
    /// output shape is `[batch_size, channels, height_out, width_out]`. For
    /// partial windows in ceil mode, only input elements are included in the norm.
    #[compute(bounds = Self::bounds)]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        assert!(
            self.p.is_finite() && self.p > 0.0,
            "Lp pooling requires a finite p > 0"
        );
        assert!(
            self.kernel_size.iter().all(|&size| size > 0),
            "kernel sizes must be positive"
        );
        assert!(
            self.stride.iter().all(|&step| step > 0),
            "strides must be positive"
        );

        let [height, width] = [input.dims()[2], input.dims()[3]];
        let output_height =
            output_size(height, self.kernel_size[0], self.stride[0], self.ceil_mode);
        let output_width = output_size(width, self.kernel_size[1], self.stride[1], self.ceil_mode);
        let padded_height = (output_height - 1) * self.stride[0] + self.kernel_size[0];
        let padded_width = (output_width - 1) * self.stride[1] + self.kernel_size[1];
        let pad = [
            (0, 0),
            (0, 0),
            (0, padded_height.saturating_sub(height)),
            (0, padded_width.saturating_sub(width)),
        ];
        let padded = if padded_height > height || padded_width > width {
            input.pad(pad, PadMode::Constant(0.0))
        } else {
            input
        };

        padded
            .abs()
            .powf_scalar(self.p)
            .unfold::<5, _>(2, self.kernel_size[0], self.stride[0])
            .unfold::<6, _>(3, self.kernel_size[1], self.stride[1])
            .sum_dims_squeeze::<4, _>(&[4, 5])
            .powf_scalar(1.0 / self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData, Tolerance};

    #[test]
    fn computes_l2_norms() {
        let device = Default::default();
        let input = Tensor::<4>::from_data([[[[3.0f32, -4.0], [0.0, 12.0]]]], &device);
        let output = LpPool2dConfig::new([2, 2]).init().forward(input);

        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[[13.0]]]]), Tolerance::default());
    }

    #[test]
    fn preserves_autodiff_gradients() {
        let device = Device::default().autodiff();
        let input = Tensor::<4>::from_data([[[[3.0f32, 4.0]]]], &device).require_grad();
        let output = LpPool2dConfig::new([1, 2]).init().forward(input.clone());
        let gradients = output.sum().backward();

        input
            .grad(&gradients)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[[3.0 / 5.0, 4.0 / 5.0]]]]),
                Tolerance::default(),
            );
    }

    #[test]
    fn ceil_mode_includes_partial_windows_without_counting_padding() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(
            [[[[3.0f32, 4.0, 12.0], [0.0, 0.0, 0.0], [8.0, 15.0, 0.0]]]],
            &device,
        );
        let output = LpPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_ceil_mode(true)
            .init()
            .forward(input);

        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[5.0, 12.0], [17.0, 0.0]]]]),
            Tolerance::default(),
        );
    }

    #[test]
    fn ceil_mode_does_not_create_windows_beyond_the_input() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(
            [[[[3.0f32, 4.0, 12.0], [0.0, 0.0, 0.0], [8.0, 15.0, 0.0]]]],
            &device,
        );
        let output = LpPool2dConfig::new([2, 2])
            .with_strides([10, 10])
            .with_ceil_mode(true)
            .init()
            .forward(input);

        assert_eq!(output.dims(), [1, 1, 1, 1]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[[5.0]]]]), Tolerance::default());
    }
}
