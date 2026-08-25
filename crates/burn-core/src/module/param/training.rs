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
/// - [`valid`](AutodiffModule::valid) clears it, and [`from_inner`](AutodiffModule::from_inner) —
///   which is what [`train`](Module::train) is — sets it.
///
/// It starts set, because a module is built before it is placed and the alternative default would
/// silently disable dropout for every run that never calls `train()`.
///
/// [`freeze_group`](Module::freeze_group) does *not* clear it: that matches parameters by
/// [`ParamId`](crate::module::ParamId), and a flag is not a parameter and has none. A
/// group-frozen subtree keeps behaving as it does during training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingFlag {
    /// Its identity in the module tree, so a [`ParamGroup`](crate::module::ParamGroup) can name it
    /// the way it names a parameter. Without one, freezing a group could not reach it: the group
    /// most callers build is a set of ids collected off a subtree, and a flag with no id is not in
    /// any such set.
    pub id: ParamId,
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
            training,
        }
    }

    /// Whether the layer holding this should behave as it does during training.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// The same flag, in the given state, keeping its identity.
    pub fn with(self, training: bool) -> Self {
        Self { training, ..self }
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
    /// The whole point of the type: freezing a subtree reaches the layers in it that have no
    /// parameters to freeze.
    fn no_grad(self) -> Self {
        self.with(false)
    }

    /// Offered to the mapper so a group-scoped traversal can reach it. The default hook is the
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
        self.with(false)
    }

    fn from_inner(module: Self) -> Self {
        module.with(true)
    }
}

impl ModuleDisplayDefault for TrainingFlag {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(&format!("{self}")).optional()
    }
}

impl ModuleDisplay for TrainingFlag {}
