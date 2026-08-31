use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Flag, Module, ModuleDisplay, Param};
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
    /// [`freeze`](burn::module::Module::freeze) and matching
    /// [`freeze_group`](burn::module::Module::freeze_group) traversals.
    pub training: Param<Flag>,
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
            training: Param::from_bool(true),
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
        if !self.training.is_enabled() || !input.device().is_autodiff() || self.prob == 0.0 {
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
        // A layer behaving as it does during training is the ordinary case and says nothing; a
        // frozen one is the case worth seeing, and the only way to see it at all.
        let content = content.add("prob", &self.prob);
        match self.training.is_enabled() {
            true => content.optional(),
            false => content.add("training", &self.training).optional(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;

    #[cfg(feature = "std")]
    #[derive(Module, Debug)]
    struct TrainableDropout {
        linear: crate::Linear,
        dropout: Dropout,
    }

    #[cfg(feature = "std")]
    fn trainable_dropout() -> TrainableDropout {
        use crate::LinearConfig;
        use burn::tensor::Device;

        let device = Device::default().autodiff();
        TrainableDropout {
            linear: LinearConfig::new(4, 4).init(&device),
            dropout: DropoutConfig::new(0.5).init(),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn no_grad_only_disables_tensor_gradients() {
        let model = trainable_dropout().no_grad();

        assert!(!model.linear.weight.is_require_grad());
        assert!(model.dropout.training.is_enabled());
    }

    #[cfg(feature = "std")]
    #[test]
    fn require_grad_group_only_changes_matched_tensors() {
        use burn::module::ParamGroup;

        let model = trainable_dropout();
        let group = ParamGroup::ids_from_module(model.clone());
        let model = model.set_require_grad_group(group.clone(), false);

        assert!(!model.linear.weight.is_require_grad());
        assert!(model.dropout.training.is_enabled());

        let model = model.set_require_grad_group(group, true);

        assert!(model.linear.weight.is_require_grad());
        assert!(model.dropout.training.is_enabled());
    }

    #[cfg(feature = "std")]
    #[test]
    fn freeze_and_unfreeze_change_tensors_and_flags() {
        let model = trainable_dropout().freeze();

        assert!(!model.linear.weight.is_require_grad());
        assert!(!model.dropout.training.is_enabled());

        let model = model.unfreeze();

        assert!(model.linear.weight.is_require_grad());
        assert!(model.dropout.training.is_enabled());
    }

    #[cfg(feature = "std")]
    #[test]
    fn frozen_dropout_on_a_training_device_passes_its_input_through() {
        use burn::tensor::Device;
        // Frozen where partial finetuning leaves it: on the training device,
        // because that is where the rest of the graph is. The device alone
        // cannot tell this apart from a layer that really is being trained.
        let device = Device::default().autodiff();
        let tensor = Tensor::<2>::ones(Shape::new([100, 100]), &device);
        let dropout = DropoutConfig::new(0.5).init().freeze();

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
        .freeze();

        let output = wrapper.dropout.forward(tensor.clone());

        assert_eq!(output.to_data(), tensor.to_data());
    }

    #[test]
    fn freeze_reaches_flags_inside_module_containers() {
        let dropouts = alloc::vec![DropoutConfig::new(0.5).init()].freeze();

        assert!(!dropouts[0].training.is_enabled());
    }

    #[test]
    fn backend_transitions_preserve_a_frozen_flag() {
        use burn::module::AutodiffModule;

        let dropout = DropoutConfig::new(0.5).init().freeze();
        let dropout = dropout.valid().train();

        assert!(!dropout.training.is_enabled());
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
        .freeze();

        assert!(!wrapper.blocks[0].training.is_enabled(), "Vec");
        assert!(!wrapper.optional.unwrap().training.is_enabled(), "Option");
        assert!(!wrapper.fixed[0].training.is_enabled(), "array");
        assert!(!wrapper.pair.0.training.is_enabled(), "tuple");
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_frozen_dropout_comes_back_frozen_from_train() {
        // `train()` is `from_inner`, which reinstates what the caller asked for
        // the way it does for a parameter's `require_grad`. A frozen half coming
        // back armed while its parameters stayed frozen is the bug.
        let dropout = DropoutConfig::new(0.5).init().freeze().train();

        assert!(!dropout.training.is_enabled());
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
            !model.frozen.dropout.training.is_enabled(),
            "the frozen half's dropout should have been reached"
        );
        assert!(
            model.trained.dropout.training.is_enabled(),
            "the half outside the group should be untouched"
        );
        assert!(!model.frozen.linear.weight.is_require_grad());
        assert!(model.trained.linear.weight.is_require_grad());
    }

    #[cfg(feature = "std")]
    #[test]
    fn path_group_can_freeze_and_unfreeze_a_flag() {
        use burn::module::ParamGroup;

        #[derive(Module, Debug)]
        struct Wrapper {
            selected: Dropout,
            untouched: Dropout,
        }

        let dropout = || DropoutConfig::new(0.5).init();
        let model = Wrapper {
            selected: dropout(),
            untouched: dropout(),
        }
        .freeze_group(ParamGroup::from_path("selected.training"));

        assert!(!model.selected.training.is_enabled());
        assert!(model.untouched.training.is_enabled());

        let model = model.unfreeze_group(ParamGroup::from_path("selected.training"));

        assert!(model.selected.training.is_enabled());
        assert!(model.untouched.training.is_enabled());
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

        let frozen = config.init().freeze();
        assert_eq!(
            alloc::format!("{frozen}"),
            "Dropout {prob: 0.5, training: disabled}"
        );
    }

    #[test]
    #[should_panic = "Dropout probability should be between 0 and 1,"]
    fn dropout_prob_invalid() {
        let config = DropoutConfig::new(-10.);
        let _layer = config.init();
    }
}
