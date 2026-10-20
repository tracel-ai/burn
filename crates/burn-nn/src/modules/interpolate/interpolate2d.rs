use alloc::format;

use burn::tensor::module::interpolate;

use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::{
    CoordinateTransformMode, CubicOptions, InterpolateOptions, NearestRoundingMode,
};

use super::InterpolateMode;

/// Configuration for the 2D interpolation module.
///
/// This struct defines the configuration options for the 2D interpolation operation.
/// It allows specifying the output size, scale factor, and interpolation mode.
#[derive(Config, Debug)]
pub struct Interpolate2dConfig {
    /// Output size of the interpolated tensor.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub output_size: Option<[usize; 2]>,

    /// Scale factor for resizing the input tensor.
    /// The output size is `floor(input_size * scale_factor)`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
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

    /// Coordinate transformation mode.
    #[config(default = "None")]
    pub coordinate_transform: Option<CoordinateTransformMode>,

    /// Nearest rounding mode.
    #[config(default = "None")]
    pub nearest_mode: Option<NearestRoundingMode>,

    /// Cubic interpolation options.
    #[config(default = "None")]
    pub cubic_options: Option<CubicOptions>,
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
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Interpolate2d {
    /// Output size of the interpolated tensor
    pub output_size: Option<[usize; 2]>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode used for resizing
    #[module(skip)]
    pub mode: InterpolateMode,

    /// Whether to align corner pixels
    pub align_corners: bool,

    /// Coordinate transformation mode
    #[module(skip)]
    pub coordinate_transform: Option<CoordinateTransformMode>,

    /// Nearest rounding mode
    #[module(skip)]
    pub nearest_mode: Option<NearestRoundingMode>,

    /// Cubic interpolation options
    #[module(skip)]
    pub cubic_options: Option<CubicOptions>,
}

impl Interpolate2dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate2d {
        Interpolate2d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: self.mode,
            align_corners: self.align_corners,
            coordinate_transform: self.coordinate_transform,
            nearest_mode: self.nearest_mode,
            cubic_options: self.cubic_options,
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
    /// # Panics
    ///
    /// Panics unless exactly one of `output_size` or `scale_factor` is set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::<4>::random([1, 3, 64, 64], Distribution::Uniform(0.0, 1.0), &device);
    /// let interpolate = Interpolate2dConfig::new()
    ///     .with_output_size(Some([128, 128]))
    ///     .init();
    /// let output = interpolate.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 128, 128]);
    /// ```
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let mut options = InterpolateOptions::new(self.mode.clone().into());
        if let Some(ct) = self.coordinate_transform {
            options = options.with_coordinate_transform(ct);
        } else {
            options = options.with_align_corners(self.align_corners);
        }
        if let Some(nm) = self.nearest_mode {
            options = options.with_nearest_mode(nm);
        }
        if let Some(co) = self.cubic_options {
            options = options.with_cubic_options(co);
        }
        options.output_size = self.output_size;
        options.scale_factor = self.scale_factor;
        interpolate(input, options)
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
            .add_debug_attribute("mode", &self.mode)
            .add("output_size", &format!("{:?}", self.output_size))
            .add("scale_factor", &self.scale_factor)
            .optional()
    }
}
#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    #[test]
    fn test_module() {
        let input = Tensor::<4>::random(
            [2, 3, 4, 4],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        // Test with output_size
        let config = Interpolate2dConfig::new().with_output_size(Some([8, 8]));
        let interpolate = config.init();
        let output = interpolate.forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 8, 8]);

        // Test with scale_factor on a non-square input to catch swapped axes
        let non_square = Tensor::<4>::random(
            [2, 3, 4, 6],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );
        let config = Interpolate2dConfig::new().with_scale_factor(Some([0.5, 2.0]));
        let interpolate = config.init();
        let output = interpolate.forward(non_square);
        assert_eq!(output.dims(), [2, 3, 2, 12]);

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
