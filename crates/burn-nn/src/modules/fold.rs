use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};

use burn::tensor::Tensor;
use burn::tensor::module::fold4d;
use burn::tensor::ops::UnfoldOptions;

/// Configuration to create a [fold 4d](Fold4d) layer using the [init function](Fold4dConfig::init).
#[derive(Config, Debug)]
pub struct Fold4dConfig {
    /// The spatial size `[height, width]` of the output tensor.
    pub output_size: [usize; 2],
    /// The size of the sliding blocks.
    pub kernel_size: [usize; 2],
    /// The stride of the sliding blocks.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// The padding configuration.
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
}

/// Four-dimensional folding, the inverse of [Unfold4d](crate::Unfold4d).
///
/// Combines an array of sliding local blocks into a large containing tensor, summing overlaps.
///
/// Should be created with [Fold4dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Fold4d {
    /// The spatial size of the output tensor.
    pub output_size: [usize; 2],
    /// The size of the sliding blocks.
    pub kernel_size: [usize; 2],
    /// The stride of the sliding blocks.
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    pub dilation: [usize; 2],
    /// The padding configuration.
    pub padding: [usize; 2],
}

impl ModuleDisplay for Fold4d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("output_size", &alloc::format!("{:?}", self.output_size))
            .add("kernel_size", &alloc::format!("{:?}", self.kernel_size))
            .add("stride", &alloc::format!("{:?}", self.stride))
            .add("dilation", &alloc::format!("{:?}", self.dilation))
            .add("padding", &alloc::format!("{:?}", self.padding))
            .optional()
    }
}

impl Fold4dConfig {
    /// Initializes a new [Fold4d] module.
    pub fn init(&self) -> Fold4d {
        Fold4d {
            output_size: self.output_size,
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            padding: self.padding,
        }
    }
}

impl Fold4d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [fold4d](burn::tensor::module::fold4d) for more information.
    ///
    /// # Shapes
    ///
    /// input:   `[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]`
    /// returns: `[batch_size, channels_in, output_size_1, output_size_2]`
    pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
        fold4d(
            input,
            self.output_size,
            self.kernel_size,
            UnfoldOptions::new(self.stride, self.padding, self.dilation),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::module::unfold4d;
    use burn::tensor::{TensorData, Tolerance};

    type FT = f32;

    #[test]
    fn display() {
        let config = Fold4dConfig::new([4, 4], [2, 2]);
        let fold = config.init();

        assert_eq!(
            alloc::format!("{fold}"),
            "Fold4d {output_size: [4, 4], kernel_size: [2, 2], stride: [1, 1], dilation: [1, 1], padding: [0, 0]}"
        );
    }

    #[test]
    fn fold4d_known_values() {
        let device = Default::default();
        // 4 kernel-position channels x 4 blocks (a 2x2 block grid), folded into a 3x3 output.
        let cols = Tensor::<3>::from_data(
            TensorData::from([[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]]),
            &device,
        );

        let output = Fold4dConfig::new([3, 3], [2, 2]).init().forward(cols);

        // Reference computed with a numpy col2im implementation.
        let expected =
            TensorData::from([[[[1.0, 7.0, 6.0], [12.0, 34.0, 22.0], [11.0, 27.0, 16.0]]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn fold_of_unfold_matches_reference() {
        let device = Default::default();
        let x = Tensor::<4>::from_data(
            TensorData::from([[[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]]]),
            &device,
        );

        let cols = unfold4d(x, [2, 2], UnfoldOptions::new([1, 1], [0, 0], [1, 1]));
        let folded = Fold4dConfig::new([4, 4], [2, 2]).init().forward(cols);

        // fold(unfold(x)) sums overlaps; reference from numpy col2im(im2col(x)).
        let expected = TensorData::from([[[
            [1.0, 4.0, 6.0, 4.0],
            [10.0, 24.0, 28.0, 16.0],
            [18.0, 40.0, 44.0, 24.0],
            [13.0, 28.0, 30.0, 16.0],
        ]]]);
        folded
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
