use crate as burn;
use crate::config::Config;
use crate::module::Param;
use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
/// Parametric Relu layer.
///
/// Should be created using [PReluConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PRelu<B: Backend> {
    /// the weights learnt for PReLu. can be of shape \[1\] or \[num_parameters\] in which case it must
    /// be the same as number of channels in the input tensor
    pub alpha: Param<Tensor<B, 1>>,

    /// Alpha value for the PRelu layer
    pub alpha_value: f64,
}

impl<B: Backend> ModuleDisplay for PRelu<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [num_parameters] = self.alpha.shape().dims();

        content
            .add("num_parameters", &num_parameters)
            .add("alpha_value", &self.alpha_value)
            .optional()
    }
}

/// Configuration to create a [Parametric Relu](PRelu) layer using the [init function](PReluConfig::init).
#[derive(Config, Debug)]
pub struct PReluConfig {
    /// The number of parameters.
    #[config(default = "1")]
    pub num_parameters: usize,
    /// The learnable weight alpha. Default is 0.25
    #[config(default = "0.25")]
    pub alpha: f64,
}

impl PReluConfig {
    /// Initialize a new [Parametric Relu](PRelu) Layer
    pub fn init<B: Backend>(&self, device: &B::Device) -> PRelu<B> {
        PRelu {
            // alpha is a tensor of length num_parameters
            alpha: Initializer::Constant { value: self.alpha }.init([self.num_parameters], device),
            alpha_value: self.alpha,
        }
    }
}

impl<B: Backend> PRelu<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    ///
    /// See also [prelu](crate::tensor::activation::prelu) for more information.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        crate::tensor::activation::prelu(input, self.alpha.val())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn display() {
        let layer = PReluConfig::new().init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", layer),
            "PRelu {num_parameters: 1, alpha_value: 0.25, params: 1}"
        );
    }
}
