//! Learning-rate policy for a whole module.
//!
//! This lives outside the `std`-only `lr_scheduler` module so `ModuleOptimizer` can take a fixed
//! learning rate without the `std` feature.

use alloc::{vec, vec::Vec};

use burn_core::module::{ParamGroup, ParamId};

use crate::LearningRate;

#[derive(Clone)]
pub(crate) struct LrGroup {
    pub(crate) group: ParamGroup,
    pub(crate) lr: f64,
}

/// Determines what learning rate to use for a given trainable parameter.
#[derive(Clone, Default)]
pub struct ModuleLearningRate {
    pub(crate) groups: Vec<LrGroup>,
}

impl From<LearningRate> for ModuleLearningRate {
    fn from(value: LearningRate) -> Self {
        Self {
            groups: vec![LrGroup {
                group: ParamGroup::all(),
                lr: value,
            }],
        }
    }
}

impl ModuleLearningRate {
    /// Iterate over group rates in scheduler registration order, with the base rate first.
    pub fn group_rates(&self) -> impl ExactSizeIterator<Item = LearningRate> + '_ {
        self.groups.iter().map(|group| group.lr)
    }

    /// Get the effective learning rate for the given parameter.
    pub fn lr_from_param(&self, id: ParamId, path: Option<&str>) -> LearningRate {
        self.groups
            .iter()
            .filter_map(|val| val.group.matches(&id, path).then_some(val.lr))
            .next_back()
            .expect("Should match at least one parameter group.")
    }

    /// Get the base learning rate value which's group matches all parameters.
    pub fn base(&self) -> LearningRate {
        self.groups
            .first()
            .expect("Should have at least one learning rate.")
            .lr
    }
}
