use burn_tensor::module::interpolate;

use crate as burn;

use crate::config::Config;
use crate::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use crate::tensor::backend::Backend;
use crate::tensor::ops::{InterpolateMode, InterpolateOptions};
use crate::tensor::Tensor;

/// Configuration for the interpolation module
#[derive(Config, Debug)]
pub struct InterpolateConfig {
    /// Output size
    #[config(default = "None")]
    pub output_size: Option<[usize; 2]>,

    /// Scale factor
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,

    /// Coordinate transformation mode
    #[config(default = "CoordinateTransformationMode::Asymmetric")]
    pub coordinate_transformation_mode: CoordinateTransformationMode,
}

/// Coordinate transformation mode using scale_factor
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum CoordinateTransformationMode {
    /// x_resized = (x_original + 0.5) * scale - 0.5
    HalfPixel,

    /// x_resized = x_original * scale
    Asymmetric,
}

/// Interpolation module
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Interpolate {
    /// Output size
    pub output_size: Option<[usize; 2]>,

    /// Scale factor
    pub scale_factor: Option<[f32; 2]>,

    /// Interpolation mode
    pub mode: Ignored<InterpolateMode>,

    /// Coordinate transformation mode
    pub coordinate_transformation_mode: Ignored<CoordinateTransformationMode>,
}

impl InterpolateConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate {
        Interpolate {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: Ignored(self.mode),
            coordinate_transformation_mode: Ignored(self.coordinate_transformation_mode),
        }
    }
}

impl Interpolate {
    /// Forward pass of the interpolation module
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Output tensor
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
            InterpolateOptions::new(self.mode.0.clone()),
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

impl ModuleDisplay for Interpolate {
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
    fn display() {
        let config = InterpolateConfig::new().with_output_size(Some([20, 20]));
        let layer = config.init();

        assert_eq!(
            alloc::format!("{}", layer),
            "Interpolate {mode: Nearest, output_size: Some([20, 20]), \
            scale_factor: None, coordinate_transformation_mode: Asymmetric}"
        );
    }
}
