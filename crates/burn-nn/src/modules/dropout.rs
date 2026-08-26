use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, TrainingFlag};
use burn::tensor::{Distribution, Tensor};

/// Configuration to create a [Dropout](Dropout) layer using the [init function](DropoutConfig::init).
#[derive(Config, Debug)]
pub struct DropoutConfig {
    /// The probability of randomly zeroes some elements of the input tensor during training.
    pub prob: f64,
}

/// Set at random some elements of the input tensor to zero during training.
///
/// This is an effective regularization technique as describe in the paper
/// [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580).
///
/// The input is also scaled during training to `1 / (1 - prob_keep)`.
///
/// Should be created with [DropoutConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Dropout {
    /// The probability of randomly zeroes some elements of the input tensor during training.
    pub prob: f64,
    /// Whether to behave as during training. Cleared by
    /// [`no_grad`](burn::module::Module::no_grad) and
    /// [`valid`](burn::module::AutodiffModule::valid), because a layer with no parameters has no
    /// `require_grad` of its own to read and has to be told.
    pub training: TrainingFlag,
}

impl DropoutConfig {
    /// Initialize a new [dropout](Dropout) module.
    pub fn init(&self) -> Dropout {
        if self.prob < 0.0 || self.prob > 1.0 {
            panic!(
                "Dropout probability should be between 0 and 1, but got {}",
                self.prob
            );
        }
        Dropout {
            prob: self.prob,
            training: TrainingFlag::default(),
        }
    }
}

impl Dropout {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [Dropout](Dropout) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        // Both, and for different reasons. The device says a backward is possible at all; the
        // flag says this layer takes part in one, which a subtree frozen in place on the training
        // device does not.
        if !self.training.is_training() || !input.device().is_autodiff() || self.prob == 0.0 {
            return input;
        }

        let prob_keep = 1.0 - self.prob;
        let random = input.random_like(Distribution::Bernoulli(prob_keep));
        let x = input * random;

        x * (1.0 / prob_keep)
    }
}

impl ModuleDisplay for Dropout {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("prob", &self.prob).optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;

    #[cfg(feature = "std")]
    #[test]
    fn frozen_dropout_on_a_training_device_passes_its_input_through() {
        use burn::tensor::Device;
        // Frozen where partial finetuning leaves it: on the training device,
        // because that is where the rest of the graph is. The device alone
        // cannot tell this apart from a layer that really is being trained.
        let device = Device::default().autodiff();
        let tensor = Tensor::<2>::ones(Shape::new([100, 100]), &device);
        let dropout = DropoutConfig::new(0.5).init().no_grad();

        let output = dropout.forward(tensor.clone());

        assert_eq!(
            output.to_data(),
            tensor.to_data(),
            "a frozen dropout should not perturb a subtree the caller froze"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn dropout_freezing_reaches_through_an_enclosing_module() {
        use burn::tensor::Device;
        // The flag is cleared by the traversal, not by the layer being frozen
        // directly, so nesting is the case that matters.
        #[derive(Module, Debug)]
        struct Wrapper {
            dropout: Dropout,
        }

        let device = Device::default().autodiff();
        let tensor = Tensor::<2>::ones(Shape::new([100, 100]), &device);
        let wrapper = Wrapper {
            dropout: DropoutConfig::new(0.5).init(),
        }
        .no_grad();

        let output = wrapper.dropout.forward(tensor.clone());

        assert_eq!(output.to_data(), tensor.to_data());
    }

    #[cfg(feature = "std")]
    #[test]
    fn dropout_freezing_reaches_through_a_container() {
        // The layout of every stack of blocks. Freezing walks the tree with a
        // mapper, which containers forward, so a dropout is reached wherever it
        // is held.
        #[derive(Module, Debug)]
        struct Wrapper {
            blocks: Vec<Dropout>,
            optional: Option<Dropout>,
            fixed: [Dropout; 2],
            pair: (Dropout, Dropout),
        }

        let dropout = || DropoutConfig::new(0.5).init();
        let wrapper = Wrapper {
            blocks: alloc::vec![dropout()],
            optional: Some(dropout()),
            fixed: [dropout(), dropout()],
            pair: (dropout(), dropout()),
        }
        .no_grad();

        assert!(!wrapper.blocks[0].training.is_training(), "Vec");
        assert!(!wrapper.optional.unwrap().training.is_training(), "Option");
        assert!(!wrapper.fixed[0].training.is_training(), "array");
        assert!(!wrapper.pair.0.training.is_training(), "tuple");
    }

    #[cfg(feature = "std")]
    #[test]
    fn group_freezing_a_subtree_reaches_the_dropout_in_it() {
        use crate::{Linear, LinearConfig};
        use burn::module::ParamGroup;
        use burn::tensor::Device;

        // A group built off a subtree, which is how a caller says "freeze this
        // half": `ids_from_module` collects the ids under it, and the flag is
        // collected with them because it carries one.
        #[derive(Module, Debug)]
        struct Half {
            linear: Linear,
            dropout: Dropout,
        }
        #[derive(Module, Debug)]
        struct Whole {
            frozen: Half,
            trained: Half,
        }

        let device = Device::default().autodiff();
        let half = |device: &Device| Half {
            linear: LinearConfig::new(4, 4).init(device),
            dropout: DropoutConfig::new(0.5).init(),
        };
        let model = Whole {
            frozen: half(&device),
            trained: half(&device),
        };
        let group = ParamGroup::ids_from_module(model.frozen.clone());
        let model = model.freeze_group(group);

        assert!(
            !model.frozen.dropout.training.is_training(),
            "the frozen half's dropout should have been reached"
        );
        assert!(
            model.trained.dropout.training.is_training(),
            "the half outside the group should be untouched"
        );
        assert!(!model.frozen.linear.weight.is_require_grad());
        assert!(model.trained.linear.weight.is_require_grad());
    }

    #[cfg(feature = "std")]
    #[test]
    fn with_ad_backend_should_mark_input() {
        use burn::tensor::Device;
        let device = Device::default().autodiff();
        let tensor = Tensor::<2>::ones(Shape::new([100, 100]), &device);
        let dropout = DropoutConfig::new(0.5).init();

        let output = dropout.forward(tensor.clone());

        assert_ne!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn without_ad_backend_should_not_change_input() {
        let tensor = Tensor::<2>::ones(Shape::new([100, 100]), &Default::default());
        let dropout = DropoutConfig::new(0.5).init();

        let output = dropout.forward(tensor.clone());

        assert_eq!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn display() {
        let config = DropoutConfig::new(0.5);
        let layer = config.init();

        assert_eq!(alloc::format!("{layer}"), "Dropout {prob: 0.5}");
    }

    #[test]
    #[should_panic = "Dropout probability should be between 0 and 1,"]
    fn dropout_prob_invalid() {
        let config = DropoutConfig::new(-10.);
        let _layer = config.init();
    }
}
