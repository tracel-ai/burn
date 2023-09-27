use crate as burn;

use core::marker::PhantomData;

use crate::{config::Config, module::Module};
use burn_tensor::{backend::Backend, Tensor};

/// Implementation of the KL-divergence between a given normal distribution to the standard normal distribution.  
///
/// It return the ratio of the standard normal distribution that is covered by the other normal distribution.      
/// This loss is useful when trying to coerce a distribution into the standard normal one.
#[derive(Debug, Config)]
pub struct KLDivergenceNormConfig {}

impl KLDivergenceNormConfig {
    /// Initialize KL-divergence
    pub fn init<B: Backend>(&self) -> KLDivergenceNorm<B> {
        KLDivergenceNorm::default()
    }
}

/// KL-divergence struct.
#[derive(Debug, Module, Default)]
pub struct KLDivergenceNorm<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> KLDivergenceNorm<B> {
    /// Calculates the kl-divergence against the standard normal distribution.
    pub fn forward(&self, mean: Tensor<B, 2>, log_var: Tensor<B, 2>) -> Tensor<B, 1> {
        (log_var.clone().add_scalar(1.) - mean.powf(2.) - log_var.exp())
            .sum_dim(1)
            .mean()
            .mul_scalar(-0.5)
    }
}

#[cfg(test)]
mod test {
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_kl_divergence() {
        let log_var = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.85486912, 0.17898201, 0.18947457],
            [0.01386812, 0.55957644, 0.31823984],
            [0.54037822, 0.32283059, 0.83637678],
            [0.70825881, 0.53305515, 0.79408598],
            [0.71504417, 0.14863746, 0.30203206],
        ]));
        let mean = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.02741296, 0.98073045, 0.77091668],
            [0.44853223, 0.08488636, 0.65032839],
            [0.40951806, 0.18085039, 0.13209234],
            [0.4218156, 0.61086605, 0.27888109],
            [0.30099154, 0.93819986, 0.62995347],
        ]));

        let loss = KLDivergenceNorm::default().forward(mean, log_var);
        // np.mean(-0.5 * np.sum(1 + x - y ** 2 - np.exp(x), axis = 1), axis = 0) = 0.71909...
        loss.into_data().assert_approx_eq(&Data::from([0.71909]), 4); // Checked againsts numpy calc.
    }
}
