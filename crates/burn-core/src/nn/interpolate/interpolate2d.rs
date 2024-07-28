use burn_tensor::module::interpolate;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use crate::tensor::backend::Backend;
use crate::tensor::ops::InterpolateOptions;
use crate::tensor::Tensor;

use super::{CoordinateTransformationMode, InterpolateMode};

/// Configuration for the 2D interpolation module.
///
/// This struct defines the configuration options for the 2D interpolation operation.
/// It allows specifying the output size, scale factor, interpolation mode,
/// and coordinate transformation mode.
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

    /// Coordinate transformation mode.
    /// Defines how the input and output coordinates are related.
    #[config(default = "CoordinateTransformationMode::Asymmetric")]
    pub coordinate_transformation_mode: CoordinateTransformationMode,
}

/// Interpolate module for resizing tensors with shape [N, C, H, W].
///
/// This struct represents an interpolation module that can resize tensors
/// using various interpolation methods and coordinate transformation modes.
/// It provides flexibility in specifying either an output size or a scale factor
/// for resizing, along with options for the interpolation mode and coordinate
/// transformation mode.
///
/// The module can be used to upsample or downsample tensors, preserving the
/// number of channels and batch size while adjusting the height and width
/// dimensions.
///
/// The module can be created using the [Interpolate2dConfig] struct and the
/// [init] method, which returns an instance of the [Interpolate2d] struct.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Interpolate2d {
    /// Output size of the interpolated tensor
    pub output_size: Option<[usize; 2]>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode used for resizing
    pub mode: Ignored<InterpolateMode>,

    /// Coordinate transformation mode for input and output coordinates
    pub coordinate_transformation_mode: Ignored<CoordinateTransformationMode>,
}

impl Interpolate2dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate2d {
        Interpolate2d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: Ignored(self.mode),
            coordinate_transformation_mode: Ignored(self.coordinate_transformation_mode),
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
        let output_size = calculate_output_size(
            input.dims(),
            self.output_size,
            self.scale_factor,
            self.coordinate_transformation_mode.0,
        );
        interpolate(
            input,
            output_size,
            InterpolateOptions::new(self.mode.0.clone().into()),
        )
    }
}

/// Calculate output size based on input dimensions, output size, and scale factor
fn calculate_output_size(
    input_dims: [usize; 4],
    output_size: Option<[usize; 2]>,
    scale_factor: Option<[f32; 2]>,
    coordinate_transformation_mode: CoordinateTransformationMode,
) -> [usize; 2] {
    match (output_size, scale_factor) {
        (Some(output_size), None) => {
            // Use provided
            output_size
        }
        (None, Some(scale_factor)) => {
            // Calculate output size based on scale factor
            let [_, _, h, w] = input_dims;
            match coordinate_transformation_mode {
                CoordinateTransformationMode::HalfPixel => [
                    ((h as f32 + 0.5) * scale_factor[0] - 0.5) as usize, // Floor rounding
                    ((w as f32 + 0.5) * scale_factor[1] - 0.5) as usize, // Floor rounding
                ],
                CoordinateTransformationMode::Asymmetric => [
                    ((h as f32) * scale_factor[0]) as usize, // Floor rounding
                    ((w as f32) * scale_factor[1]) as usize, // Floor rounding
                ],
            }
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
            .add(
                "coordinate_transformation_mode",
                &self.coordinate_transformation_mode,
            )
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Distribution;

    use crate::{nn::interpolate::CoordinateTransformationMode, TestBackend};

    use super::*;

    #[test]
    fn test_calculate_output_size() {
        let input_dims = [1, 1, 4, 4];

        let output_size = calculate_output_size(
            input_dims,
            Some([2, 2]),
            None,
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, [2, 2]);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some([2.0, 2.0]),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, [8, 8]);

        let output_size = calculate_output_size(
            [1, 1, 4, 4],
            None,
            Some([0.5, 0.5]),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, [2, 2]);

        let output_size = calculate_output_size(
            [1, 1, 4, 4],
            None,
            Some([2.0, 1.5]),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, [8, 6]);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some([0.7, 0.7]),
            CoordinateTransformationMode::HalfPixel,
        );
        assert_eq!(output_size, [2, 2]);
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
            alloc::format!("{}", layer),
            "Interpolate2d {mode: Nearest, output_size: Some([20, 20]), \
            scale_factor: None, coordinate_transformation_mode: Asymmetric}"
        );
    }
}
