use super::{starter::Starters, ExistingOptimizations};
use crate::stream::OptimizationItem;
use crate::{FusionBackend, Optimization};
use serde::{Deserialize, Serialize};

#[derive(new, Serialize, Deserialize)]
struct OptimizationCacheState<O> {
    optimizations: Vec<OptimizationItem<O>>,
    starters: Starters,
}

impl<O> ExistingOptimizations<O> {
    #[allow(dead_code)]
    /// TODO: save the cache state.
    pub(crate) fn save<B: FusionBackend>(&self, _device: &B::Device)
    where
        O: Optimization<B>,
    {
        let _state = OptimizationCacheState {
            optimizations: self
                .optimizations
                .iter()
                .map(|op| OptimizationItem {
                    stream: op.stream.clone(),
                    end_conditions: op.end_conditions.clone(),
                    value: op.value.to_state(),
                })
                .collect(),
            starters: self.starters.clone(),
        };
        todo!("Save the state");
    }

    #[allow(dead_code)]
    #[allow(unreachable_code)]
    #[allow(unused_variables)]
    #[allow(clippy::diverging_sub_expression)]
    /// TODO: load the cache state.
    pub(crate) fn load<B: FusionBackend>(device: &B::Device) -> Self
    where
        O: Optimization<B>,
    {
        let state = todo!("Load the state");

        Self::from_state(device, state)
    }

    fn from_state<B: FusionBackend>(
        device: &B::Device,
        state: OptimizationCacheState<B::OptimizationState>,
    ) -> Self
    where
        O: Optimization<B>,
    {
        Self {
            optimizations: state
                .optimizations
                .into_iter()
                .map(|state| OptimizationItem {
                    stream: state.stream,
                    end_conditions: state.end_conditions,
                    value: O::from_state(device, state.value),
                })
                .collect(),
            starters: state.starters,
        }
    }
}
