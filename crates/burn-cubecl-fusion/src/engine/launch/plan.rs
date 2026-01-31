use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{FuseArg, FuseOp, FuseType},
        launch::vectorization::Vect,
        trace::{RuntimeLayout, block::FuseBlock},
    },
};
use burn_ir::{TensorId, TensorIr};
use cubecl::{Runtime, ir::LineSize};
use std::collections::BTreeMap;

/// The plan is responsible to keep runtime information related to the launch of a fused kernel
/// at one place.
#[derive(Debug)]
pub struct LaunchPlan<'a, R: Runtime> {
    pub global_outputs: Vec<TensorIr>,
    pub handle_inputs: Vec<HandleInput<R>>,
    pub handle_outputs: Vec<HandleOutput<R>>,
    pub rank: usize,
    pub blocks: Vec<BlockPlan<'a>>,
    pub vectorizations: BTreeMap<TensorId, Vect>,
    pub cleared: Vec<TensorId>,
    /// Sometimes we fuse too much that we can't rely on the inputs metadata to get access to the
    /// inner shape/strides for a block. This is when we use a [RuntimeLayout] where the shape and
    /// strides are passed from the host.
    pub runtime_layouts: Vec<RuntimeLayout>,
}

#[derive(Debug)]
pub struct BlockPlan<'a> {
    pub potential_inplaces: Vec<PotentialInplace<'a>>,
    pub potential_reference_input: Option<InputReference>,
    pub reference: ReferenceSelection,
    pub reads: BTreeMap<TensorId, Vec<FuseOp>>,
    pub writes: BTreeMap<TensorId, Vec<FuseOp>>,
    pub width: LineSize,
}

#[derive(Debug)]
pub enum InputReference {
    Normal {
        input_pos: usize,
    },
    SwapDims {
        original_pos: usize,
        dims: (usize, usize),
    },
    Reshaped {
        reshape_pos: usize,
    },
}

#[derive(Clone, Debug)]
/// Determine how the reference layout is chosen.
pub enum ReferenceSelection {
    Searching,
    Concrete {
        layout: FuseArg,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    SwapDims {
        original: FuseArg,
        dims: (usize, usize),
    },
    Reshaped {
        reshape_pos: usize,
    },
    VirtualShape {
        original: FuseArg,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// We can't rely on input or output tensor metadata to know the layout of a block.
    /// Therefore we pass values from the host at runtime.
    Runtime {
        // /// The actual shape to be used as reference layout.
        // shape: Vec<usize>,
        // /// The actual strides to be used as reference layout.
        // strides: Vec<usize>,
        /// The index relative to the number of runtime layout used for a kernel.
        ///
        /// Since all runtime shapes are stored in the same buffer, the pos is necessary to fetch
        /// the right one during execution.
        pos: usize,
    },
}

impl ReferenceSelection {
    pub fn is_found(&self) -> bool {
        !matches!(self, Self::Searching)
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
            global_outputs: Vec::new(),
            handle_inputs: Vec::new(),
            handle_outputs: Vec::new(),
            rank,
            blocks,
            vectorizations: Default::default(),
            cleared: Default::default(),
            runtime_layouts: Default::default(),
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
        precision: FuseType,
        #[cfg(feature = "autotune-checks")]
        debug_info: HandleOutputAliasDebugInfo<R>,
    },
    Owned {
        global_id: TensorId,
        relative_id: TensorId,
        precision: FuseType,
        handle: CubeFusionHandle<R>,
        global_shape: Vec<usize>,
        vectorization: LineSize,
    },
}

#[derive(Debug)]
pub struct NormalHandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_ir: TensorIr,
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub line_size: LineSize,
    pub broadcated: bool,
    // Strides can be modified during plan execution, but need to be restored on rollback
    pub orig_strides: Vec<usize>,
}

#[derive(Debug)]
pub struct QuantValuesHandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_ir: TensorIr,
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub line_size: LineSize,
}

#[derive(Debug)]
pub struct QuantParamsHandleInput<R: Runtime> {
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
pub enum HandleInput<R: Runtime> {
    Normal(NormalHandleInput<R>),
    QuantValues(QuantValuesHandleInput<R>),
    QuantParams(QuantParamsHandleInput<R>),
}

impl<R: Runtime> HandleInput<R> {
    pub fn as_normal(&self) -> Option<&NormalHandleInput<R>> {
        match self {
            HandleInput::Normal(normal) => Some(normal),
            _ => None,
        }
    }
}

impl<R: Runtime> NormalHandleInput<R> {
    pub fn new(
        tensor_global: TensorIr,
        tensor_relative: &TensorIr,
        precision: FuseType,
        mut handle: CubeFusionHandle<R>,
        mut strides: Vec<usize>,
    ) -> Self {
        // For rollback
        core::mem::swap(&mut handle.strides, &mut strides);
        Self {
            precision,
            handle,
            relative_id: tensor_relative.id,
            global_ir: tensor_global,
            line_size: 1,
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
