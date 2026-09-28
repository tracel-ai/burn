/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adafactor;
mod adagrad;
mod adam;
mod adamw;
mod adan;
mod base;
mod grad_accum;
mod grads;
mod lamb;
mod lbfgs;
mod lion;
mod module;
mod muon;
mod rmsprop;
mod sgd;
mod state;
mod visitor;

pub use adafactor::*;
pub use adagrad::*;
pub use adam::*;
pub use adamw::*;
pub use adan::*;
pub use base::*;
pub use grad_accum::*;
pub use grads::*;
pub use lamb::*;
pub use lbfgs::*;
pub use lion::*;
pub use module::*;
pub use muon::*;
pub use rmsprop::*;
pub use sgd::*;
pub use state::*;

#[cfg(test)]
mod test_utils {
    use super::{GradientsParams, ModuleOptimizer};
    use crate::LearningRate;
    use burn_core::tensor::{Tensor, Tolerance};
    use burn_nn::Linear;

    /// Check that a byte round-trip preserves the optimizer's next parameter update.
    pub(super) fn assert_optimizer_resume(
        create_optimizer: impl Fn() -> ModuleOptimizer,
        mut model: Linear,
        lr: LearningRate,
    ) {
        let device = model.weight.val().device();
        let [input_size, _] = model.weight.val().dims();
        let input = Tensor::<2>::ones([2, input_size], &device);
        let grads = |model: &Linear, scale: f32| {
            GradientsParams::from_grads(
                model.forward(input.clone()).mul_scalar(scale).backward(),
                model,
            )
        };

        let mut original = create_optimizer();
        // Different gradients populate Adan's gradient-difference buffer; decreasing their
        // magnitude also makes AMSGrad's historical maximum differ from its current moment.
        for scale in [1.0, 0.01] {
            let grads = grads(&model, scale);
            model = original.step(lr, model, grads);
        }
        assert!(!original.to_record().is_empty());
        let bytes = original.into_bytes().unwrap();
        let mut restored = create_optimizer().from_bytes(bytes).unwrap();

        // A small opposing gradient exercises stored momentum, including Lion's.
        let expected = original.step(lr, model.clone(), grads(&model, -0.01));
        let actual = restored.step(lr, model.clone(), grads(&model, -0.01));

        expected
            .weight
            .to_data()
            .assert_approx_eq::<f32>(&actual.weight.to_data(), Tolerance::absolute(1e-6));
        if let Some(bias) = expected.bias {
            bias.to_data().assert_approx_eq::<f32>(
                &actual.bias.unwrap().to_data(),
                Tolerance::absolute(1e-6),
            );
        }
    }
}
