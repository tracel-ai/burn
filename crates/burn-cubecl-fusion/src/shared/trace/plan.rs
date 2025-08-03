use std::collections::BTreeMap;

use crate::{
    CubeFusionHandle,
    shared::ir::{Arg, FuseOp, FusePrecision},
};
use burn_ir::{TensorId, TensorIr};
use cubecl::Runtime;

use super::{block::FuseBlock, vectorization::Vect};

/// The plan is responsible to keep runtime information related to the launch of a fused kernel
/// at one place.
#[derive(Debug)]
pub(crate) struct LaunchPlan<'a, R: Runtime> {
    pub global_inputs: Vec<TensorIr>,
    pub global_outputs: Vec<TensorIr>,
    pub handle_inputs: Vec<HandleInput<R>>,
    pub handle_outputs: Vec<HandleOutput<R>>,
    pub rank: usize,
    pub blocks: Vec<BlockPlan<'a>>,
    pub vectorizations: BTreeMap<TensorId, Vect>,
    pub cleared: Vec<TensorId>,
}

#[derive(Debug)]
pub(crate) struct BlockPlan<'a> {
    pub potential_inplaces: Vec<PotentialInplace<'a>>,
    pub potential_reference_input: Option<InputReference>,
    pub reference: ReferenceSelection,
    pub reads: BTreeMap<TensorId, Vec<FuseOp>>,
    pub writes: BTreeMap<TensorId, FuseOp>,
    pub width: u8,
}

#[derive(Debug)]
pub enum InputReference {
    Normal {
        input_pos: usize,
    },
    SwapDims {
        original_pos: usize,
        dims: (u32, u32),
    },
    Reshaped {
        reshape_pos: usize,
    },
}

#[derive(Debug)]
/// Determine how the reference layout is chosen.
pub enum ReferenceSelection {
    Searching,
    NotFound,
    Concrete {
        layout: Arg,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    SwapDims {
        original: Arg,
        dims: (u32, u32),
    },
    Reshaped {
        reshape_pos: usize,
    },
    VirtualShape {
        original: Arg,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

impl ReferenceSelection {
    pub fn is_found(&self) -> bool {
        !matches!(self, Self::Searching | Self::NotFound)
    }

    pub fn compatible_strides_for_inplace(&self, strides_inplace: &[usize]) -> bool {
        match self {
            ReferenceSelection::Concrete { strides, .. } => strides == strides_inplace,
            _ => false,
        }
    }
}

impl<R: Runtime> LaunchPlan<'_, R> {
    pub fn new(fuse_blocks: &[FuseBlock]) -> Self {
        let mut rank = 0;
        let mut blocks = Vec::with_capacity(fuse_blocks.len());

        for b in fuse_blocks.iter() {
            rank = usize::max(b.shape_ref.len(), rank);
            let block = BlockPlan {
                reference: ReferenceSelection::Searching,
                reads: b.reads.clone(),
                writes: b.writes.clone(),
                width: 0,
                potential_inplaces: Vec::new(),
                potential_reference_input: None,
            };
            blocks.push(block);
        }

        LaunchPlan {
            global_inputs: Vec::new(),
            global_outputs: Vec::new(),
            handle_inputs: Vec::new(),
            handle_outputs: Vec::new(),
            rank,
            blocks,
            vectorizations: Default::default(),
            cleared: Default::default(),
        }
    }
}

#[cfg(feature = "autotune-checks")]
#[derive(Debug)]
pub struct HandleOutputAliasDebugInfo<R: Runtime> {
    pub handle: CubeFusionHandle<R>,
    pub relative_id: TensorId,
    pub global_shape: Vec<usize>,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum HandleOutput<R: Runtime> {
    Alias {
        input_pos: usize,
        precision: FusePrecision,
        #[cfg(feature = "autotune-checks")]
        debug_info: HandleOutputAliasDebugInfo<R>,
    },
    Owned {
        global_id: TensorId,
        relative_id: TensorId,
        precision: FusePrecision,
        handle: CubeFusionHandle<R>,
        global_shape: Vec<usize>,
        vectorization: u8,
    },
}

#[derive(Debug)]
pub struct HandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_id: TensorId,
    pub precision: FusePrecision,
    pub handle: CubeFusionHandle<R>,
    pub global_shape: Vec<usize>,
    pub vectorization: u8,
    pub broadcated: bool,
    // Strides can be modified during plan execution, but need to be restored on rollback
    pub orig_strides: Vec<usize>,
}

impl<R: Runtime> HandleInput<R> {
    pub fn new(
        tensor_global: &TensorIr,
        tensor_relative: &TensorIr,
        precision: FusePrecision,
        mut handle: CubeFusionHandle<R>,
        mut strides: Vec<usize>,
    ) -> Self {
        // For rollback
        core::mem::swap(&mut handle.strides, &mut strides);
        Self {
            precision,
            handle,
            relative_id: tensor_relative.id,
            global_id: tensor_global.id,
            global_shape: tensor_global.shape.clone(),
            vectorization: 1,
            broadcated: false,
            orig_strides: strides,
        }
    }

    pub fn handle_rollback(mut self) -> CubeFusionHandle<R> {
        core::mem::swap(&mut self.handle.strides, &mut self.orig_strides);
        self.handle
    }
}

#[derive(Debug)]
pub struct PotentialInplace<'a> {
    pub input_pos: usize,
    pub tensor_relative: &'a TensorIr,
    pub strides: Vec<usize>,
}
