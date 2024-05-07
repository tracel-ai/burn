use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::ops::UnfoldOptions;
use crate::tensor::Tensor;

use crate::tensor::module::unfold4d;

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
pub struct Unfold4d {
    config: Unfold4dConfig,
}

impl Unfold4dConfig {
    /// Initializes a new [Unfold4d] module.
    pub fn init(&self) -> Unfold4d {
        Unfold4d {
            config: self.clone(),
        }
    }
}

impl Unfold4d {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [unfold4d](crate::tensor::module::unfold4d) for more information.
    ///
    /// # Shapes
    ///
    /// input:   `[batch_size, channels_in, height, width]`  
    /// returns: `[batch_size, channels_in * kernel_size_1 * kernel_size_2, number of blocks]`
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
