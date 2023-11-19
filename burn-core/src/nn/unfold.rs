use crate as burn;

use crate::config::Config;
use crate::module::Module;
use burn_tensor::backend::Backend;
use burn_tensor::module::unfold4d;
use burn_tensor::ops::UnfoldOptions;
use burn_tensor::Tensor;

/// Configuration to create an [unfold 4D](Unfold4d) layer.
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
#[derive(Module, Clone, Debug)]
pub struct Unfold4d {
    config: Unfold4dConfig,
}

impl Unfold4dConfig {
    /// Initialize a new [unfold 4k](Unfold4d) module.
    pub fn init(&self) -> Unfold4d {
        Unfold4d {
            config: self.clone(),
        }
    }
}

impl Unfold4d {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// input:      `[batch_size, channels_in, height, width]`,
    /// returns: `[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]`,
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        unfold4d(
            input,
            self.config.kernel_size,
            UnfoldOptions::new(
                self.config.stride,
                self.config.padding,
                self.config.dilation,
            ),
        )
    }
}
