use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::unfold4d;
use burn::tensor::ops::UnfoldOptions;

/// Configuration to create an [unfold 4d](Unfold4d) layer using the [init function](Unfold4dConfig::init).
#[derive(Config, Debug)]
pub struct Unfold4dConfig {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// The padding configuration.
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
}

/// Four-dimensional unfolding.
///
/// Should be created with [Unfold4dConfig].
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Unfold4d {
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    pub dilation: [usize; 2],
    /// The padding configuration.
    pub padding: [usize; 2],
}

impl ModuleDisplay for Unfold4d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("kernel_size", &alloc::format!("{:?}", &self.kernel_size))
            .add("stride", &alloc::format!("{:?}", &self.stride))
            .add("dilation", &alloc::format!("{:?}", &self.dilation))
            .add("padding", &alloc::format!("{:?}", &self.padding))
            .optional()
    }
}

impl Unfold4dConfig {
    /// Initializes a new [Unfold4d] module.
    pub fn init(&self) -> Unfold4d {
        Unfold4d {
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            padding: self.padding,
        }
    }
}

impl Unfold4d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [unfold4d](burn::tensor::module::unfold4d) for more information.
    ///
    /// # Shapes
    ///
    /// input:   `[batch_size, channels_in, height, width]`
    /// returns: `[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]`
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        unfold4d(
            input,
            self.kernel_size,
            UnfoldOptions::new(self.stride, self.padding, self.dilation),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = Unfold4dConfig::new([3, 3]);
        let unfold = config.init();

        assert_eq!(
            alloc::format!("{unfold}"),
            "Unfold4d {kernel_size: [3, 3], stride: [1, 1], dilation: [1, 1], padding: [0, 0]}"
        );
    }
}
