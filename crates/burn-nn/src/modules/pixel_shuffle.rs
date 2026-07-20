use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;

/// Configuration to create a [PixelShuffle](PixelShuffle) layer using the
/// [init function](PixelShuffleConfig::init).
#[derive(Config, Debug)]
pub struct PixelShuffleConfig {
    /// The upscale factor applied to the spatial dimensions.
    pub upscale_factor: usize,
}

impl PixelShuffleConfig {
    /// Initialize a new [PixelShuffle](PixelShuffle) layer.
    pub fn init(&self) -> PixelShuffle {
        PixelShuffle {
            upscale_factor: self.upscale_factor,
        }
    }
}

/// Rearranges a tensor of shape `[N, C * r^2, H, W]` into `[N, C, H * r, W * r]`, where `r` is the
/// upscale factor. Inverse of [PixelUnshuffle], matching
/// [`torch.nn.PixelShuffle`](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html).
///
/// Should be created with [PixelShuffleConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PixelShuffle {
    /// The upscale factor applied to the spatial dimensions.
    pub upscale_factor: usize,
}

impl ModuleDisplay for PixelShuffle {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("upscale_factor", &self.upscale_factor)
            .optional()
    }
}

impl PixelShuffle {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, channels * upscale_factor^2, height, width]`
    /// - output: `[batch_size, channels, height * upscale_factor, width * upscale_factor]`
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let [batch_size, channels, height, width] = input.dims();
        let factor = self.upscale_factor;

        assert!(
            factor > 0,
            "PixelShuffle: upscale_factor must be greater than 0"
        );
        assert_eq!(
            channels % (factor * factor),
            0,
            "PixelShuffle: input channels ({channels}) must be divisible by upscale_factor^2 ({})",
            factor * factor
        );
        let out_channels = channels / (factor * factor);

        input
            .reshape([batch_size, out_channels, factor, factor, height, width])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([batch_size, out_channels, height * factor, width * factor])
    }
}

/// Configuration to create a [PixelUnshuffle](PixelUnshuffle) layer using the
/// [init function](PixelUnshuffleConfig::init).
#[derive(Config, Debug)]
pub struct PixelUnshuffleConfig {
    /// The downscale factor applied to the spatial dimensions.
    pub downscale_factor: usize,
}

impl PixelUnshuffleConfig {
    /// Initialize a new [PixelUnshuffle](PixelUnshuffle) layer.
    pub fn init(&self) -> PixelUnshuffle {
        PixelUnshuffle {
            downscale_factor: self.downscale_factor,
        }
    }
}

/// Rearranges a tensor of shape `[N, C, H * r, W * r]` into `[N, C * r^2, H, W]`, where `r` is the
/// downscale factor. Inverse of [PixelShuffle], matching
/// [`torch.nn.PixelUnshuffle`](https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html).
///
/// Should be created with [PixelUnshuffleConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PixelUnshuffle {
    /// The downscale factor applied to the spatial dimensions.
    pub downscale_factor: usize,
}

impl ModuleDisplay for PixelUnshuffle {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("downscale_factor", &self.downscale_factor)
            .optional()
    }
}

impl PixelUnshuffle {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, channels, height * downscale_factor, width * downscale_factor]`
    /// - output: `[batch_size, channels * downscale_factor^2, height, width]`
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let [batch_size, channels, height, width] = input.dims();
        let factor = self.downscale_factor;

        assert!(
            factor > 0,
            "PixelUnshuffle: downscale_factor must be greater than 0"
        );
        assert_eq!(
            height % factor,
            0,
            "PixelUnshuffle: input height ({height}) must be divisible by downscale_factor ({factor})"
        );
        assert_eq!(
            width % factor,
            0,
            "PixelUnshuffle: input width ({width}) must be divisible by downscale_factor ({factor})"
        );
        let out_height = height / factor;
        let out_width = width / factor;

        input
            .reshape([batch_size, channels, out_height, factor, out_width, factor])
            .permute([0, 1, 3, 5, 2, 4])
            .reshape([
                batch_size,
                channels * factor * factor,
                out_height,
                out_width,
            ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    fn sample_input() -> TensorData {
        // (1, 4, 2, 2) with values 0..16.
        TensorData::from([[
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]],
            [[12.0, 13.0], [14.0, 15.0]],
        ]])
    }

    #[test]
    fn pixel_shuffle_known_values() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(sample_input(), &device);

        let output = PixelShuffleConfig::new(2).init().forward(input);

        // (1, 1, 4, 4); reference from a numpy pixel_shuffle implementation.
        let expected = TensorData::from([[[
            [0.0, 4.0, 1.0, 5.0],
            [8.0, 12.0, 9.0, 13.0],
            [2.0, 6.0, 3.0, 7.0],
            [10.0, 14.0, 11.0, 15.0],
        ]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn pixel_unshuffle_inverts_pixel_shuffle() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(sample_input(), &device);

        let shuffled = PixelShuffleConfig::new(2).init().forward(input.clone());
        let restored = PixelUnshuffleConfig::new(2).init().forward(shuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn pixel_shuffle_inverts_pixel_unshuffle() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(sample_input(), &device);

        let unshuffled = PixelUnshuffleConfig::new(2).init().forward(input.clone());
        let restored = PixelShuffleConfig::new(2).init().forward(unshuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn pixel_shuffle_round_trip_factor_3() {
        let device = Default::default();
        // (1, 9, 1, 1): C * r^2 = 1 * 9 channels, r = 3.
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [[0.0]],
                [[1.0]],
                [[2.0]],
                [[3.0]],
                [[4.0]],
                [[5.0]],
                [[6.0]],
                [[7.0]],
                [[8.0]],
            ]]),
            &device,
        );

        let shuffled = PixelShuffleConfig::new(3).init().forward(input.clone());
        let restored = PixelUnshuffleConfig::new(3).init().forward(shuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn display() {
        assert_eq!(
            alloc::format!("{}", PixelShuffleConfig::new(2).init()),
            "PixelShuffle {upscale_factor: 2}"
        );
        assert_eq!(
            alloc::format!("{}", PixelUnshuffleConfig::new(2).init()),
            "PixelUnshuffle {downscale_factor: 2}"
        );
    }
}
