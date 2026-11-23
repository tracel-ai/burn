use crate::{
    CubeFusionHandle, FallbackOperation,
    optim::{
        FusedOperation,
        elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
        reduce::{ReduceOptimizationInfo, ReduceOptimizationState, ReduceOptimizationTuneArg},
    },
};
use burn_fusion::stream::Context;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Reductions along one axis of one shape, with the element-wise work around and between them,
/// broadcasting their results back: run as a chain of blocks, each reduce block a fused reduce
/// (fused where its autotune measured that faster) and each element-wise block its own launch.
pub struct ReduceBroadcastedOptimization {
    pub(crate) info: Arc<ReduceBroadcastedOptimizationInfo>,
    pub(crate) num_ops: usize,
}

pub(crate) struct ReduceBroadcastedOptimizationInfo {
    pub(crate) blocks: Vec<ReduceBlockOptimInfo>,
}

pub(crate) enum ReduceBlockOptimInfo {
    Reduce(Arc<ReduceOptimizationInfo>),
    Elemwise(Arc<ElemwiseOptimization>),
}

impl ReduceBlockOptimInfo {
    pub fn from_state(device: &cubecl::Device, state: ReduceBlockState) -> Self {
        match state {
            ReduceBlockState::Reduce(state) => {
                Self::Reduce(Arc::new(ReduceOptimizationInfo::from_state(device, state)))
            }
            ReduceBlockState::Elemwise(state) => {
                Self::Elemwise(Arc::new(ElemwiseOptimization::from_state(device, state)))
            }
        }
    }
    pub fn to_state(&self) -> ReduceBlockState {
        match self {
            Self::Reduce(info) => ReduceBlockState::Reduce(info.to_state()),
            Self::Elemwise(info) => ReduceBlockState::Elemwise(info.to_state()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReduceBroadcastedOptimizationState {
    blocks: Vec<ReduceBlockState>,
    num_ops: usize,
}

#[derive(Serialize, Deserialize, Debug)]
#[allow(clippy::large_enum_variant)] // Only for serialization.
pub enum ReduceBlockState {
    Reduce(ReduceOptimizationState),
    Elemwise(ElemwiseOptimizationState),
}

impl ReduceBroadcastedOptimization {
    /// Execute the optimization: each block in turn.
    pub fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle>,
        fallback: impl Fn(usize) -> Box<dyn FallbackOperation>,
    ) {
        let mut current_index = 0;
        for block in self.info.blocks.iter() {
            match block {
                ReduceBlockOptimInfo::Reduce(info) => {
                    // The index of the fallback reduce is the number of ops fused as read.
                    let fallback = fallback(current_index + info.len_read);
                    current_index += info.len;
                    ReduceOptimizationTuneArg {
                        info: info.clone(),
                        fallback: Arc::new(fallback),
                    }
                    .run(context);
                }
                ReduceBlockOptimInfo::Elemwise(elemwise) => {
                    current_index += elemwise.num_ops_fused();
                    elemwise.execute(context);
                }
            }
        }
    }

    pub fn to_state(&self) -> ReduceBroadcastedOptimizationState {
        ReduceBroadcastedOptimizationState {
            blocks: self
                .info
                .blocks
                .iter()
                .map(|info| info.to_state())
                .collect(),
            num_ops: self.num_ops,
        }
    }

    pub fn from_state(device: &cubecl::Device, state: ReduceBroadcastedOptimizationState) -> Self {
        Self {
            info: Arc::new(ReduceBroadcastedOptimizationInfo {
                blocks: state
                    .blocks
                    .into_iter()
                    .map(|state| ReduceBlockOptimInfo::from_state(device, state))
                    .collect(),
            }),
            num_ops: state.num_ops,
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.num_ops
    }
}

/// Name of the broadcasted-reduce fusion optimization.
pub const NAME: &str = "ReduceBroadcasted";

impl FusedOperation for ReduceBroadcastedOptimization {
    fn max_relative_shape_id(&self) -> Option<usize> {
        self.info
            .blocks
            .iter()
            .filter_map(|block| match block {
                ReduceBlockOptimInfo::Reduce(info) => info.max_relative_shape_id(),
                ReduceBlockOptimInfo::Elemwise(opt) => {
                    FusedOperation::max_relative_shape_id(opt.as_ref())
                }
            })
            .max()
    }

    const NAME: &'static str = self::NAME;
    type State = ReduceBroadcastedOptimizationState;

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation>,
    ) {
        Self::execute(self, context, |index| fallback(index))
    }

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &cubecl::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}
