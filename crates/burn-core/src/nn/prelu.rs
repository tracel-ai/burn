use crate as burn;
use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::Tensor;
/// Applies the Gaussian Error Linear Units function element-wise.
#[derive(Module, Debug)]
pub struct PRelu<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    num_parameters: usize,
}
#[derive(Config, Debug)]
pub struct PReluConfig {
    /// The number of parameters.
    #[config(default = "1")]
    pub num_parameters: usize,
    /// The initial value of the parameters.
    #[config(default = "0.25")]
    pub alpha: f64,
}
impl PReluConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PRelu<B> {
        PRelu {
            num_parameters: self.num_parameters,
            // alpha is a tensor of length num_parameters
            alpha: Param::from(
                Initializer::Constant { value: self.alpha }.init([self.num_parameters], device),
            ),
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
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let channels = if input.dims().len() >= 2 {
            input.dims()[1]
        } else {
            // keep as 1 channel
            1
        };
        let msg = format!("Number of parameters to PRelu must be either 1 or number of channels in input tensor (if dimension > 2). Got num_parameters:{} and tensor of shape: {:?} with channels: {}", self.num_parameters,input.dims(),channels);
        assert!(
            self.num_parameters == 1 || self.num_parameters == channels,
            "{}",
            &msg
        );
        crate::tensor::activation::prelu(input, self.alpha.val())
    }
}
