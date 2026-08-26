use alloc::format;
use core::fmt::{Display, Formatter, Result as FmtResult};

use crate as burn;
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, ParamId,
};

/// Whether the layer holding it should behave as it does during training.
///
/// A layer with parameters does not need this: freezing is expressed by clearing `require_grad`,
/// so [`BatchNorm`](https://docs.rs/burn-nn) can read its own weight and know. A layer with no
/// parameters — dropout, noise, randomized activations — has nothing to read, so the answer has
/// to be *written* to it instead, and this is what holds it.
///
/// It is written by the module tree's own transitions, and reading a field is all a layer has to
/// do:
///
/// - [`no_grad`](Module::no_grad) clears it. Freezing a subtree says the caller does not want it
///   trained, and a frozen dropout still perturbing its activations is not what that means. This
///   is the case the device alone cannot answer, because a frozen subtree stays on the training
///   device — that is where the rest of the graph lives.
/// - [`freeze_group`](Module::freeze_group) clears it for the layers the group names, and
///   [`unfreeze_group`](Module::unfreeze_group) sets it again. A group matches by
///   [`ParamId`](crate::module::ParamId) or by module path, and a flag carries both, so the
///   group-scoped traversal reaches it the way it reaches a parameter.
/// - [`valid`](AutodiffModule::valid) clears it for the duration of the inference module it
///   builds, and [`from_inner`](AutodiffModule::from_inner) — which is what
///   [`train`](Module::train) is — restores whatever the caller last asked for.
///
/// It starts set, because a module is built before it is placed and the alternative default would
/// silently disable dropout for every run that never calls `train()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingFlag {
    /// Its identity in the module tree, so a [`ParamGroup`](crate::module::ParamGroup) can name it
    /// the way it names a parameter. Without one, freezing a group could not reach it: the group
    /// most callers build is a set of ids collected off a subtree, and a flag with no id is not in
    /// any such set.
    pub id: ParamId,
    /// What the caller asked for, which outlives the round trip through an inference module the
    /// way [`Param`](crate::module::Param) keeps `require_grad` across one.
    ///
    /// Freezing writes it; `valid()` does not, so `train()` has something to restore from. Without
    /// it a frozen dropout would come back armed on the far side of that round trip, while the
    /// parameters next to it stayed frozen.
    trainable: bool,
    /// Whether the layer should behave as during training *right now*, which is what the layer
    /// reads. Freezing clears it, and so does the inference module `valid()` builds.
    training: bool,
}

impl Default for TrainingFlag {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TrainingFlag {
    /// A flag in the given state, with a fresh identity.
    pub fn new(training: bool) -> Self {
        Self {
            id: ParamId::new(),
            trainable: training,
            training,
        }
    }

    /// Whether the layer holding this should behave as it does during training.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Whether the caller wants the layer holding this trained at all, which an inference module
    /// does not change.
    pub fn is_trainable(&self) -> bool {
        self.trainable
    }

    /// The same flag, in the given state, keeping its identity.
    ///
    /// This is what the caller asks for, so it moves both what the layer reads now and what a
    /// later [`train`](Module::train) restores.
    pub fn with(self, training: bool) -> Self {
        Self {
            trainable: training,
            training,
            ..self
        }
    }
}

impl Display for TrainingFlag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.training {
            true => f.write_str("train"),
            false => f.write_str("eval"),
        }
    }
}

impl Module for TrainingFlag {
    /// Offered to the mapper, which is how every transition reaches it: freezing a subtree — whole
    /// with [`no_grad`](Module::no_grad) or by group — is a mapper, and so the layers in it that
    /// have no parameters to freeze are reached like the ones that do. The default hook is the
    /// identity, so every mapper that does not care — records, quantization, device moves — is
    /// unaffected.
    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_training(self)
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_training(self);
    }

    fn to_device(self, _: &burn::tensor::Device) -> Self {
        self
    }

    fn fork(self, _: &burn::tensor::Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: burn::module::Devices) -> burn::module::Devices {
        devices
    }
}

impl AutodiffModule for TrainingFlag {
    fn valid(&self) -> Self {
        // Only what the layer reads: `trainable` is the caller's own answer, and an inference
        // module is not the caller changing their mind.
        Self {
            training: false,
            ..*self
        }
    }

    fn from_inner(module: Self) -> Self {
        Self {
            training: module.trainable,
            ..module
        }
    }
}

impl ModuleDisplayDefault for TrainingFlag {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(&format!("{self}")).optional()
    }
}

impl ModuleDisplay for TrainingFlag {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freezing_survives_the_round_trip_through_an_inference_module() {
        // What `Param` does with `require_grad`: an inference module is a view, not a decision,
        // so `train()` comes back to what the caller last asked for.
        let frozen = TrainingFlag::default().with(false);

        assert!(!TrainingFlag::from_inner(frozen).is_training());
        assert!(!TrainingFlag::from_inner(frozen.valid()).is_training());
    }

    #[test]
    fn an_inference_module_does_not_freeze_the_module_it_came_from() {
        let flag = TrainingFlag::default();

        assert!(!flag.valid().is_training(), "the inference module is eval");
        assert!(
            TrainingFlag::from_inner(flag.valid()).is_training(),
            "and taking it back to training restores what the caller asked for"
        );
    }

    #[test]
    fn transitions_keep_the_identity_a_group_names_it_by() {
        let flag = TrainingFlag::default();

        assert_eq!(flag.with(false).id, flag.id);
        assert_eq!(flag.valid().id, flag.id);
        assert_eq!(TrainingFlag::from_inner(flag).id, flag.id);
    }
}
