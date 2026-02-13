use alloc::format;

use burn::tensor::module::interpolate;

use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::ops::InterpolateOptions;

use super::InterpolateMode;

/// Configuration for the 2D interpolation module.
///
/// This struct defines the configuration options for the 2D interpolation operation.
/// It allows specifying the output size, scale factor, and interpolation mode.
#[derive(Config, Debug)]
pub struct Interpolate2dConfig {
    /// Output size of the interpolated tensor.
    /// If specified, this takes precedence over `scale_factor`.
    #[config(default = "None")]
    pub output_size: Option<[usize; 2]>,

    /// Scale factor for resizing the input tensor.
    /// This is used when `output_size` is not specified.
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode to use for resizing.
    /// Determines how the output values are calculated.
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,

    /// If `true`, the input and output tensors are aligned by their corner pixels.
    /// If `false`, half-pixel coordinate mapping is used instead.
    #[config(default = true)]
    pub align_corners: bool,
}

/// Interpolate module for resizing tensors with shape [N, C, H, W].
///
/// This struct represents an interpolation module that can resize tensors
/// using various interpolation methods. It provides flexibility in specifying
/// either an output size or a scale factor for resizing, along with options
/// for the interpolation mode.
///
/// The module can be used to upsample or downsample tensors, preserving the
/// number of channels and batch size while adjusting the height and width
/// dimensions.
///
/// The module can be created using the [Interpolate2dConfig] struct and the
/// `init` method, which returns an instance of the [Interpolate2d] struct.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Interpolate2d {
    /// Output size of the interpolated tensor
    pub output_size: Option<[usize; 2]>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode used for resizing
    pub mode: Ignored<InterpolateMode>,

    /// Whether to align corner pixels
    pub align_corners: bool,
}

impl Interpolate2dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate2d {
        Interpolate2d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: Ignored(self.mode),
            align_corners: self.align_corners,
        }
    }
}
impl Interpolate2d {
    /// Performs the forward pass of the interpolation module
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape [N, C, H, W]
    ///
    /// # Returns
    ///
    /// Resized tensor with shape [N, C, H', W'], where H' and W' are determined by
    /// the output_size or scale_factor specified in the module configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::<Backend, 2>::random([1, 3, 64, 64], Distribution::Uniform(0.0, 1.0), &device);
    /// let interpolate = Interpolate2dConfig::new()
    ///     .with_output_size(Some([128, 128]))
    ///     .init();
    /// let output = interpolate.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 128, 128]);
    /// ```
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output_size = calculate_output_size(input.dims(), self.output_size, self.scale_factor);
        interpolate(
            input,
            output_size,
            InterpolateOptions::new(self.mode.0.clone().into())
                .with_align_corners(self.align_corners),
        )
    }
}

/// Calculates the output size for tensor interpolation.
///
/// # Arguments
///
/// * `input_dims` - The dimensions of the input tensor [N, C, H, W].
/// * `output_size` - Optional desired output size [H', W'].
/// * `scale_factor` - Optional scale factor for height and width [scale_h, scale_w].
///
/// # Returns
///
/// A tuple [H', W'] representing the calculated output size.
///
/// # Panics
///
/// Panics if neither `output_size` nor `scale_factor` is provided,
/// or if the scale factor results in dimensions exceeding usize::MAX.
fn calculate_output_size(
    input_dims: [usize; 4],
    output_size: Option<[usize; 2]>,
    scale_factor: Option<[f32; 2]>,
) -> [usize; 2] {
    match (output_size, scale_factor) {
        (Some(output_size), None) => {
            // Use provided
            output_size
        }
        (None, Some(scale_factor)) => {
            // Calculate output size based on scale factor
            let [_, _, h, w] = input_dims;

            let new_dim_h = (h as f64) * (scale_factor[0] as f64);

            if new_dim_h > usize::MAX as f64 {
                panic!("Scale factor for height is too large");
            }

            let new_dim_w = (w as f64) * (scale_factor[1] as f64);

            if new_dim_w > usize::MAX as f64 {
                panic!("Scale factor for width is too large");
            }

            [new_dim_h as usize, new_dim_w as usize]
        }
        _ => panic!("Either output_size or scale_factor must be provided"),
    }
}

impl ModuleDisplay for Interpolate2d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("mode", &self.mode)
            .add("output_size", &format!("{:?}", self.output_size))
            .add("scale_factor", &self.scale_factor)
            .optional()
    }
}
#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use crate::TestBackend;

    use super::*;

    #[test]
    fn test_calculate_output_size() {
        let input_dims = [1, 1, 4, 4];

        let output_size = calculate_output_size(input_dims, Some([2, 2]), None);
        assert_eq!(output_size, [2, 2]);

        let output_size = calculate_output_size(input_dims, None, Some([2.0, 2.0]));
        assert_eq!(output_size, [8, 8]);

        let output_size = calculate_output_size([1, 1, 4, 4], None, Some([0.5, 0.5]));
        assert_eq!(output_size, [2, 2]);

        let output_size = calculate_output_size([1, 1, 4, 4], None, Some([2.0, 1.5]));
        assert_eq!(output_size, [8, 6]);
    }

    #[test]
    #[should_panic(expected = "Either output_size or scale_factor must be provided")]
    fn test_missing_params() {
        calculate_output_size([1, 1, 4, 4], None, None);
    }

    #[test]
    #[should_panic(expected = "Scale factor for height is too large")]
    fn test_infinite_height() {
        calculate_output_size([1, 1, usize::MAX - 1, 4], None, Some([2.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "Scale factor for width is too large")]
    fn test_infinite_width() {
        calculate_output_size([1, 1, 4, usize::MAX - 1], None, Some([1.0, 2.0]));
    }

    #[test]
    fn test_module() {
        let input = Tensor::<TestBackend, 4>::random(
            [2, 3, 4, 4],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        // Test with output_size
        let config = Interpolate2dConfig::new().with_output_size(Some([8, 8]));
        let interpolate = config.init();
        let output = interpolate.forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 8, 8]);

        // Test with scale_factor
        let config = Interpolate2dConfig::new().with_scale_factor(Some([0.5, 0.5]));
        let interpolate = config.init();
        let output = interpolate.forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 2, 2]);

        // Test with different interpolation mode
        let config = Interpolate2dConfig::new()
            .with_output_size(Some([6, 6]))
            .with_mode(InterpolateMode::Linear);
        let interpolate = config.init();
        let output = interpolate.forward(input);
        assert_eq!(output.dims(), [2, 3, 6, 6]);
    }

    #[test]
    fn display() {
        let config = Interpolate2dConfig::new().with_output_size(Some([20, 20]));
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "Interpolate2d {mode: Nearest, output_size: Some([20, 20]), \
            scale_factor: None}"
        );
    }
}
