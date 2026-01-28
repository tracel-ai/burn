use burn_core as burn;

use super::Reduction;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::{config::Config, module::Module};

/// Configuration to create a [KLDiv loss](KLDivLoss).
#[derive(Config, Debug)]
pub struct KLDivLossConfig {
    /// Specifies whether target is the log space. Default: False.
    #[config(default = false)]
    pub log_target: bool,
}

impl KLDivLossConfig {
    /// Initialize [KLDiv Loss](KLDivLoss).
    pub fn init(&self) -> KLDivLoss {
        KLDivLoss {
            log_target: self.log_target,
        }
    }
}

/// Kullback-Leibler Divergence Loss
///
/// KL Divergence shows the difference between two probability distributions by measuring information loss
///
/// KLDivLoss =
/// ```tex
/// y_{true} \cdot (\log{y_{true}} - \log{y_{pred}})
///     ```
/// By Default,this loss expects the argument `inp` in the log-space, and the output will be applied `batchmean`.
/// The argument `target` may also be provided in the log-space if `log_target` = true
///
/// See
/// - [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)
#[derive(Module, Debug, Clone)]
#[module(custom_display)]
pub struct KLDivLoss {
    /// Specifies whether target is the log space. Default: False.
    pub log_target: bool,
}

impl ModuleDisplay for KLDivLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("log_target", &self.log_target).optional()
    }
}

impl KLDivLoss {
    /// Compute the criterion on the input tensor.
    /// # Shapes
    ///
    /// - predictions: [batch_size,num_targets]
    /// - targets: [batch_size,num_targets]
    /// - output: [1]
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::BatchMean | Reduction::Auto => {
                let batch_size = loss.dims()[0] as f32;
                loss.sum().div_scalar(batch_size)
            }
            Reduction::Mean => {
                // log::warn!("reduction: 'Reduction::Mean' divides the total loss by both the batch size and the support size.'Reduction::BatchMean' divides only by the batch size, and aligns with the KL div math definition.");
                loss.mean()
            }
            Reduction::Sum => loss.sum(),
        }
    }
    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self.log_target {
            true => targets.clone().exp().mul(targets.sub(predictions)),
            false => {
                let epsilon = 1e-8;
                let log_target = targets.clone().clamp(epsilon, 1.0).log();
                targets.mul(log_target.sub(predictions))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_kl_div_loss() {
        let predict = TensorData::from([[-1.0, -0.5], [-2.0, -0.2]]);
        let targets = TensorData::from([[0.4, 0.6], [0.1, 0.9]]);

        let device = Default::default();
        let predict = TestTensor::<2>::from_data(predict, &device);
        let targets = TestTensor::<2>::from_data(targets, &device);

        let kl_loss = KLDivLossConfig { log_target: false }.init();

        let loss_sum = kl_loss.forward(predict.clone(), targets.clone(), Reduction::Sum);
        let loss_batch_mean =
            kl_loss.forward(predict.clone(), targets.clone(), Reduction::BatchMean);
        let loss_no_reduction = kl_loss.forward_no_reduction(predict, targets);

        let expected_no_reduction =
            TensorData::from([[0.0334837139, -0.0064953566], [-0.0302585065, 0.0851755068]]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::absolute(1e-5));

        let expected_sum = TensorData::from([0.08191]);
        loss_sum
            .into_data()
            .assert_approx_eq::<FT>(&expected_sum, Tolerance::absolute(1e-5));

        let expected_batch_mean = TensorData::from([0.04095]);
        loss_batch_mean
            .into_data()
            .assert_approx_eq::<FT>(&expected_batch_mean, Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_kl_div_loss_log_target() {
        let device = Default::default();
        let predict = TestTensor::<1>::from_data([-1.0, -2.0], &device);
        let targets = TestTensor::<1>::from_data([-0.5, -1.5], &device);

        let kl_loss = KLDivLossConfig { log_target: true }.init();

        let loss_no_reduction = kl_loss.forward_no_reduction(predict.clone(), targets.clone());
        let expected_none = TensorData::from([0.3032653299, 0.1115650801]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected_none, Tolerance::absolute(1e-5));

        let loss_batch_mean =
            kl_loss.forward(predict.clone(), targets.clone(), Reduction::BatchMean);
        let expected_bm = TensorData::from([0.207415204965]);
        loss_batch_mean
            .into_data()
            .assert_approx_eq::<FT>(&expected_bm, Tolerance::absolute(1e-5));

        let loss_sum = kl_loss.forward(predict, targets, Reduction::Sum);
        let expected_sum = TensorData::from([0.414830409931]);
        loss_sum
            .into_data()
            .assert_approx_eq::<FT>(&expected_sum, Tolerance::absolute(1e-5));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_kl_div_ad_loss() {
        type TestAutodiffTensor = Tensor<crate::TestAutodiffBackend, 2>;

        let device = Default::default();
        let predict = TestAutodiffTensor::from_data([[-1.0, -0.5]], &device).require_grad();
        let targets = TestAutodiffTensor::from_data([[0.4, 0.6]], &device);

        let kl_loss = KLDivLossConfig { log_target: false }.init();
        let loss = kl_loss.forward(predict.clone(), targets, Reduction::Sum);

        let grads = loss.backward();
        let grads_predict = predict.grad(&grads).unwrap();

        // d/d_pred [target * (log_target - pred)] = -target
        let expected = TensorData::from([[-0.4, -0.6]]);
        grads_predict
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = KLDivLossConfig { log_target: true };
        let loss = config.init();

        assert_eq!(alloc::format!("{loss}"), "KLDivLoss {log_target: true}");
    }
}
