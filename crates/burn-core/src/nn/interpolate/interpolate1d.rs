use burn_tensor::module::interpolate;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use crate::tensor::backend::Backend;
use crate::tensor::ops::InterpolateOptions;
use crate::tensor::Tensor;

use super::{CoordinateTransformationMode, InterpolateMode};

/// Configuration for the 1D interpolation module.
///
/// This struct defines the configuration options for the 1D interpolation operation.
/// It allows specifying the output size, scale factor, interpolation mode,
/// and coordinate transformation mode.
#[derive(Config, Debug)]
pub struct Interpolate1dConfig {
    /// Output size of the interpolated tensor.
    /// If specified, this takes precedence over `scale_factor`.
    #[config(default = "None")]
    pub output_size: Option<usize>,

    /// Scale factor for resizing the input tensor.
    /// This is used when `output_size` is not specified.
    #[config(default = "None")]
    pub scale_factor: Option<f32>,

    /// Interpolation mode to use for resizing.
    /// Determines how the output values are calculated.
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,

    /// Coordinate transformation mode.
    /// Defines how the input and output coordinates are related.
    #[config(default = "CoordinateTransformationMode::Asymmetric")]
    pub coordinate_transformation_mode: CoordinateTransformationMode,
}

/// Interpolate module for resizing 1D tensors with shape [N, C, L].
///
/// This struct represents a 1D interpolation module that can resize tensors
/// using various interpolation methods and coordinate transformation modes.
/// It provides flexibility in specifying either an output size or a scale factor
/// for resizing, along with options for the interpolation mode and coordinate
/// transformation mode.
///
/// The module can be used to upsample or downsample 1D tensors, preserving the
/// number of channels and batch size while adjusting the length dimension.
///
/// The module can be created using the [Interpolate1dConfig] struct and the
/// [init] method, which returns an instance of the [Interpolate1d] struct.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Interpolate1d {
    /// Output size of the interpolated tensor
    pub output_size: Option<usize>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<f32>,

    /// Interpolation mode used for resizing
    pub mode: Ignored<InterpolateMode>,

    /// Coordinate transformation mode for input and output coordinates
    pub coordinate_transformation_mode: Ignored<CoordinateTransformationMode>,
}

impl Interpolate1dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate1d {
        Interpolate1d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: Ignored(self.mode),
            coordinate_transformation_mode: Ignored(self.coordinate_transformation_mode),
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
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::<Backend, 3>::random([1, 3, 64], Distribution::Uniform(0.0, 1.0), &device);
    /// let interpolate = InterpolateConfig::new()
    ///     .with_output_size(Some(128))
    ///     .init();
    /// let output = interpolate.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 128]);
    /// ```
    pub fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output_size = calculate_output_size(
            input.dims(),
            self.output_size,
            self.scale_factor,
            self.coordinate_transformation_mode.0,
        );

        // Use the interpolate operation to resize the temporal input tensor
        // by adding a new dimension for the interpolation axis
        let input = input.unsqueeze_dim(2);

        let result = interpolate(
            input,
            [1, output_size],
            InterpolateOptions::new(self.mode.0.clone().into()),
        );

        result.squeeze_dims(&[2])
    }
}

/// Calculate output size based on input dimensions, output size, and scale factor
fn calculate_output_size(
    input_dims: [usize; 3],
    output_size: Option<usize>,
    scale_factor: Option<f32>,
    coordinate_transformation_mode: CoordinateTransformationMode,
) -> usize {
    match (output_size, scale_factor) {
        (Some(output_size), None) => {
            // Use provided
            output_size
        }
        (None, Some(scale_factor)) => {
            // Calculate output size based on scale factor
            let [_, _, l] = input_dims;
            match coordinate_transformation_mode {
                CoordinateTransformationMode::HalfPixel => {
                    ((l as f32 + 0.5) * scale_factor - 0.5) as usize
                } // Floor rounding

                CoordinateTransformationMode::Asymmetric => ((l as f32) * scale_factor) as usize, // Floor rounding
            }
        }
        _ => panic!("Either output_size or scale_factor must be provided"),
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

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_calculate_output_size() {
        let input_dims = [1, 1, 4];

        let output_size = calculate_output_size(
            input_dims,
            Some(2),
            None,
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, 2);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some(2.0),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, 8);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some(0.5),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, 2);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some(1.5),
            CoordinateTransformationMode::Asymmetric,
        );
        assert_eq!(output_size, 6);

        let output_size = calculate_output_size(
            input_dims,
            None,
            Some(0.7),
            CoordinateTransformationMode::HalfPixel,
        );
        assert_eq!(output_size, 2);
    }

    #[test]
    fn test_module() {
        let input = Tensor::<TestBackend, 3>::random(
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
    fn display() {
        let config = Interpolate1dConfig::new().with_output_size(Some(20));
        let layer = config.init();

        assert_eq!(
            alloc::format!("{}", layer),
            "Interpolate1d {mode: Nearest, output_size: Some(20), \
            scale_factor: None, coordinate_transformation_mode: Asymmetric}"
        );
    }
}
