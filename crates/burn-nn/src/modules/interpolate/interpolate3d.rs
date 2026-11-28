use alloc::format;

use burn_core as burn;

use crate::modules::bounds::{self, Windowing};
use burn::config::Config;
use burn::instrument::{Bounds, compute};
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::module::{interpolate3d, interpolate3d_output_size};
use burn::tensor::ops::Interpolate3dOptions;

use super::Interpolate3dMode;

/// Configuration for resizing a tensor's depth, height and width.
#[derive(Config, Debug)]
pub struct Interpolate3dConfig {
    /// Output spatial size `[depth, height, width]`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub output_size: Option<[usize; 3]>,

    /// Scale factors `[depth, height, width]`.
    /// Each output size is `floor(input_size * scale_factor)`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation algorithm and coordinate mapping.
    #[config(default = "Interpolate3dMode::Nearest")]
    pub mode: Interpolate3dMode,
}

/// Resizes tensors with shape `[N, C, D, H, W]`, preserving batch and channel dimensions.
///
/// Create this module with [`Interpolate3dConfig`]. Its sampling semantics and backend
/// restrictions are those of [`interpolate3d`].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Interpolate3d {
    /// Output spatial size `[depth, height, width]`.
    pub output_size: Option<[usize; 3]>,

    /// Scale factors `[depth, height, width]`.
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation algorithm and coordinate mapping.
    #[module(skip)]
    pub mode: Interpolate3dMode,
}

impl Interpolate3dConfig {
    /// Initializes the interpolation module.
    pub fn init(self) -> Interpolate3d {
        Interpolate3d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: self.mode,
        }
    }
}

impl Interpolate3d {
    fn options(&self) -> Interpolate3dOptions {
        let mut options = Interpolate3dOptions::new(self.mode);
        options.output_size = self.output_size;
        options.scale_factor = self.scale_factor;
        options
    }

    fn bounds(&self, input: &Tensor<5>) -> Bounds {
        let input_size = bounds::spatial(input);
        let output_size = interpolate3d_output_size(input_size, &self.options());
        Windowing {
            output_size,
            read_size: core::array::from_fn(|i| input_size[i].min(output_size[i])),
        }
        .resampling(input)
    }

    /// Resizes an input `[N, C, D, H, W]` to `[N, C, D_out, H_out, W_out]`.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one of `output_size` or `scale_factor` is set, or if the
    /// sizes, scales or selected backend do not satisfy [`interpolate3d`]'s requirements.
    #[compute(bounds = Self::bounds)]
    pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
        interpolate3d(input, self.options())
    }
}

impl ModuleDisplay for Interpolate3d {
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
    use super::*;
    use burn::config::config_to_json;
    use burn::tensor::ops::CoordinateTransformMode;

    #[test]
    fn config_roundtrip_and_forward() {
        let config = Interpolate3dConfig::new()
            .with_output_size(Some([3, 3, 3]))
            .with_mode(Interpolate3dMode::Trilinear {
                coordinate_transform: CoordinateTransformMode::AlignCorners,
            });
        let config = Interpolate3dConfig::load_binary(config_to_json(&config).as_bytes()).unwrap();
        let input = Tensor::<5>::from_floats(
            [[[[[0.0, 4.0], [8.0, 12.0]], [[16.0, 20.0], [24.0, 28.0]]]]],
            &Default::default(),
        );

        let output = config.init().forward(input.clone());
        assert_eq!(output.dims(), [1, 1, 3, 3, 3]);
        assert_eq!(output.into_data().as_slice::<f32>().unwrap()[13], 14.0);

        let output = Interpolate3dConfig::new()
            .with_scale_factor(Some([2.0, 3.0, 4.0]))
            .init()
            .forward(input);
        assert_eq!(output.dims(), [1, 1, 4, 6, 8]);
    }

    #[test]
    fn display() {
        let layer = Interpolate3dConfig::new()
            .with_output_size(Some([4, 6, 8]))
            .init();
        assert_eq!(
            format!("{layer}"),
            "Interpolate3d {mode: Nearest, output_size: Some([4, 6, 8]), scale_factor: None}"
        );
    }
}
