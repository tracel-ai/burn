//! Learning-rate policy for a whole module.
//!
//! This lives outside the `std`-only `lr_scheduler` module so `ModuleOptimizer` can take a fixed
//! learning rate without the `std` feature.

use alloc::vec::Vec;

use burn_core::module::{ParamGroup, ParamId};

use crate::LearningRate;

#[derive(Clone)]
struct LrGroup {
    group: ParamGroup,
    lr: LearningRate,
}

/// Determines what learning rate to use for a given trainable parameter.
#[derive(Clone, Default)]
pub struct ModuleLearningRate {
    groups: Vec<LrGroup>,
}

impl From<LearningRate> for ModuleLearningRate {
    fn from(value: LearningRate) -> Self {
        Self::from_ordered_groups([(ParamGroup::all(), value)])
    }
}

impl ModuleLearningRate {
    /// The first group must match all parameters. Later matching groups take precedence.
    pub(crate) fn from_ordered_groups(
        groups: impl IntoIterator<Item = (ParamGroup, LearningRate)>,
    ) -> Self {
        Self {
            groups: groups
                .into_iter()
                .map(|(group, lr)| LrGroup { group, lr })
                .collect(),
        }
    }

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
