use alloc::format;
use core::fmt::{Display, Formatter, Result as FmtResult};

use crate as burn;
use crate::empty;
use crate::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};

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
pub struct TrainingFlag(pub bool);

impl Default for TrainingFlag {
    fn default() -> Self {
        Self(true)
    }
}

impl TrainingFlag {
    /// Whether the layer holding this should behave as it does during training.
    pub fn is_training(&self) -> bool {
        self.0
    }
}

impl Display for TrainingFlag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.0 {
            true => f.write_str("train"),
            false => f.write_str("eval"),
        }
    }
}

impl Module for TrainingFlag {
    /// The whole point of the type: freezing a subtree reaches the layers in it that have no
    /// parameters to freeze.
    fn no_grad(self) -> Self {
        Self(false)
    }

    empty!(module);
}

impl AutodiffModule for TrainingFlag {
    fn valid(&self) -> Self {
        Self(false)
    }

    fn from_inner(_module: Self) -> Self {
        Self(true)
    }
}

impl ModuleDisplayDefault for TrainingFlag {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(&format!("{self}")).optional()
    }
}

impl ModuleDisplay for TrainingFlag {}
