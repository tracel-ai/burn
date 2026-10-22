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

/// Configuration for the 1D interpolation module.
///
/// This struct defines the configuration options for the 1D interpolation operation.
/// It allows specifying the output size, scale factor, and interpolation mode.
#[derive(Config, Debug)]
pub struct Interpolate1dConfig {
    /// Output size of the interpolated tensor.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub output_size: Option<usize>,

    /// Scale factor for resizing the input tensor.
    /// The output size is `floor(input_size * scale_factor)`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub scale_factor: Option<f32>,

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

/// Interpolate module for resizing 1D tensors with shape [N, C, L].
///
/// This struct represents a 1D interpolation module that can resize tensors
/// using various interpolation methods. It provides flexibility in specifying
/// either an output size or a scale factor for resizing, along with options
/// for the interpolation mode.
///
/// The module can be used to upsample or downsample 1D tensors, preserving the
/// number of channels and batch size while adjusting the length dimension.
///
/// The module can be created using the [Interpolate1dConfig] struct and the
/// `init` method, which returns an instance of the [Interpolate1d] struct.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Interpolate1d {
    /// Output size of the interpolated tensor
    pub output_size: Option<usize>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<f32>,

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

impl Interpolate1dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate1d {
        Interpolate1d {
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

impl Interpolate1d {
    /// Performs the forward pass of the 1D interpolation module
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape [N, C, L]
    ///
    /// # Returns
    ///
    /// Resized tensor with shape [N, C, L'], where L' is determined by
    /// the output_size or scale_factor specified in the module configuration
    ///
    /// # Panics
    ///
    /// Panics unless exactly one of `output_size` or `scale_factor` is set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::<3>::random([1, 3, 64], Distribution::Uniform(0.0, 1.0), &device);
    /// let interpolate = Interpolate1dConfig::new()
    ///     .with_output_size(Some(128))
    ///     .init();
    /// let output = interpolate.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 128]);
    /// ```
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let mut options = InterpolateOptions::new(self.mode.clone().into());
        if let Some(mut ct) = self.coordinate_transform {
            if let CoordinateTransformMode::TfCropAndResize { ref mut roi, .. } = ct {
                roi[0] = [0.0, 1.0];
            }
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
        options.output_size = self.output_size.map(|size| [1, size]);
        options.scale_factor = self.scale_factor.map(|scale| [1.0, scale]);

        // Use the interpolate operation to resize the temporal input tensor
        // by adding a new dimension for the interpolation axis
        let input = input.unsqueeze_dim(2);

        let result = interpolate(input, options);

        result.squeeze_dims(&[2])
    }
}

impl ModuleDisplay for Interpolate1d {
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
        let input = Tensor::<3>::random(
            [2, 3, 4],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        // Test with output_size
        let config = Interpolate1dConfig::new().with_output_size(Some(8));
        let interpolate = config.init();
        let output = interpolate.forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 8]);

        // Test with scale_factor
        let config = Interpolate1dConfig::new().with_scale_factor(Some(0.5));
        let interpolate = config.init();
        let output = interpolate.forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 2]);

        // Test with different interpolation mode
        let config = Interpolate1dConfig::new()
            .with_output_size(Some(6))
            .with_mode(InterpolateMode::Linear);
        let interpolate = config.init();
        let output = interpolate.forward(input);
        assert_eq!(output.dims(), [2, 3, 6]);
    }

    #[test]
    #[should_panic]
    fn test_conflicting_size_and_scale_panics() {
        let input = Tensor::<3>::random(
            [2, 3, 4],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );
        let config = Interpolate1dConfig::new()
            .with_output_size(Some(8))
            .with_scale_factor(Some(2.0));
        let interpolate = config.init();
        let _ = interpolate.forward(input);
    }

    #[test]
    fn display() {
        let config = Interpolate1dConfig::new().with_output_size(Some(20));
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "Interpolate1d {mode: Nearest, output_size: Some(20), \
            scale_factor: None}"
        );
    }
}
