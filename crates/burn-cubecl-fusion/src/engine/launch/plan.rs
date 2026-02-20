use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{FuseArg, FuseOp, FuseType},
        launch::vectorization::Vect,
        trace::{RuntimeLayout, block::FuseBlock},
    },
};
use burn_ir::{TensorId, TensorIr};
use burn_std::{Shape, Strides};
use cubecl::{Runtime, ir::LineSize};
use std::collections::BTreeMap;

/// The `LaunchPlan` is responsible for aggregating all runtime information required
/// to dispatch a fused kernel.
///
/// It maps abstract IR tensors to memory handles, manages vectorization
/// strategies, and tracks layout transformations.
#[derive(Debug)]
pub struct LaunchPlan<'a, R: Runtime> {
    /// The IR representation of tensors that are results of the fusion.
    pub global_outputs: Vec<TensorIr>,
    /// Memory handles and metadata for all input tensors.
    pub handle_inputs: Vec<HandleInput<R>>,
    /// Memory handles and metadata for all output tensors, including aliased inputs.
    pub handle_outputs: Vec<HandleOutput<R>>,
    /// The rank across all tensors in the plan.
    ///
    /// Smaller tensors are unsqueezed during launch.
    pub rank: usize,
    /// Detailed planning for each individual computation block within the fusion.
    pub blocks: Vec<BlockPlan<'a>>,
    /// Mapping of tensor IDs to their specific vectorization factors.
    pub vectorizations: BTreeMap<TensorId, Vect>,
    /// Tensors that can be cleared or deallocated after this plan executes.
    pub cleared: Vec<TensorId>,
    /// Metadata for shapes and strides passed from the host when they cannot be
    /// inferred from input tensors (e.g., complex deep fusions).
    pub runtime_layouts: Vec<RuntimeLayout>,
}

/// Information regarding the execution of a specific block of operations within a fusion.
#[derive(Debug)]
pub struct BlockPlan<'a> {
    /// List of inputs that are candidates for in-place memory reuse within this block.
    pub potential_inplaces: Vec<PotentialInplace<'a>>,
    /// The input tensor chosen to define the iteration space, if any.
    pub potential_reference_input: Option<InputReference>,
    /// How the master layout is determined for this block.
    pub reference: ReferenceSelection,
    /// Mapping of tensor IDs to the read operations performed on them.
    pub reads: BTreeMap<TensorId, Vec<FuseOp>>,
    /// Mapping of tensor IDs to the write operations performed on them.
    pub writes: BTreeMap<TensorId, Vec<FuseOp>>,
    /// The width for the operations in this block.
    pub width: LineSize,
}

/// Metadata for an input tensor being used as a reference for a block's layout.
#[derive(Debug)]
pub enum InputReference {
    /// Standard input at the specified position.
    Normal { input_pos: usize },
    /// Input that has an axis swapped.
    SwapDims {
        original_pos: usize,
        dims: (usize, usize),
    },
    /// Input that has been reshaped.
    Reshaped { reshape_pos: usize },
}

/// Strategies for selecting the reference layout of a fused block.
///
/// The reference layout determines how global indices are mapped to tensor coordinates.
#[derive(Clone, Debug)]
pub enum ReferenceSelection {
    /// The engine is still calculating the optimal reference.
    Searching,
    /// Layout from a normal tensor.
    Concrete {
        layout: FuseArg,
        shape: Shape,
        strides: Strides,
    },
    /// Layout from a swapped dim tensor.
    SwapDims {
        original: FuseArg,
        dims: (usize, usize),
    },
    /// Layout from a reshaped tensor.
    Reshaped { reshape_pos: usize },
    /// Layout that has the shape of an input, but not its strides.
    VirtualShape {
        original: FuseArg,
        shape: Shape,
        strides: Strides,
    },
    /// The layout is provided dynamically by the host at runtime.
    Runtime { pos: usize },
}

impl<R: Runtime> LaunchPlan<'_, R> {
    /// Creates a new `LaunchPlan` from a slice of fusion blocks.
    ///
    /// Initializes blocks with default "Searching" references and calculates
    /// the initial max rank.
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

/// Debugging information for aliased handles when `autotune-checks` is enabled.
#[cfg(feature = "autotune-checks")]
#[derive(Debug)]
pub struct HandleOutputAliasDebugInfo<R: Runtime> {
    pub handle: CubeFusionHandle<R>,
    pub relative_id: TensorId,
    pub global_shape: Shape,
}

/// Represents the output of a fused kernel execution.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum HandleOutput<R: Runtime> {
    /// An output that reuses the memory of an input tensor (In-place).
    Alias {
        /// Index of the input handle being aliased.
        input_pos: usize,
        /// Data type precision.
        precision: FuseType,
        #[cfg(feature = "autotune-checks")]
        debug_info: HandleOutputAliasDebugInfo<R>,
    },
    /// An output that requires a newly allocated memory buffer.
    Owned {
        global_id: TensorId,
        relative_id: TensorId,
        precision: FuseType,
        handle: CubeFusionHandle<R>,
        global_shape: Shape,
        vectorization: LineSize,
    },
}

/// A standard input handle with associated layout and vectorization metadata.
#[derive(Debug)]
pub struct NormalHandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_ir: TensorIr,
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub line_size: LineSize,
    pub broadcated: bool,
    /// Stores the original strides of the handle for restoration during plan rollback.
    pub orig_strides: Strides,
}

/// An input handle containing values for a quantized tensor.
#[derive(Debug)]
pub struct QuantValuesHandleInput<R: Runtime> {
    pub relative_id: TensorId,
    pub global_ir: TensorIr,
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub line_size: LineSize,
}

/// An input handle containing parameters (scales/offsets) for quantization.
#[derive(Debug)]
pub struct QuantParamsHandleInput<R: Runtime> {
    pub precision: FuseType,
    pub handle: CubeFusionHandle<R>,
    pub shape: Shape,
}

/// Different types of inputs that can be passed to a fused kernel.
#[derive(Debug)]
pub enum HandleInput<R: Runtime> {
    Normal(NormalHandleInput<R>),
    QuantValues(QuantValuesHandleInput<R>),
    QuantParams(QuantParamsHandleInput<R>),
}

impl<R: Runtime> HandleInput<R> {
    /// Returns a reference to the inner `NormalHandleInput` if the variant matches.
    pub fn as_normal(&self) -> Option<&NormalHandleInput<R>> {
        match self {
            HandleInput::Normal(normal) => Some(normal),
            _ => None,
        }
    }
}

impl<R: Runtime> NormalHandleInput<R> {
    /// Creates a new `NormalHandleInput` tracking original strides.
    pub fn new(
        tensor_global: TensorIr,
        tensor_relative: &TensorIr,
        precision: FuseType,
        mut handle: CubeFusionHandle<R>,
        mut strides: Strides,
    ) -> Self {
        // Swap current handle strides with provided strides to track the original state for rollback.
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

    /// Restores the handle's original strides and returns the handle.
    ///
    /// Used when a plan is invalidated or needs to be rolled back.
    pub fn handle_rollback(mut self) -> CubeFusionHandle<R> {
        core::mem::swap(&mut self.handle.strides, &mut self.orig_strides);
        self.handle
    }
}

/// A candidate for in-place optimization.
#[derive(Debug)]
pub struct PotentialInplace<'a> {
    /// Position of the input handle in the `handle_inputs` vector.
    pub input_pos: usize,
    /// Reference to the IR of the relative tensor.
    pub tensor_relative: &'a TensorIr,
    /// Current strides of the potential in-place candidate.
    pub strides: Strides,
}

impl ReferenceSelection {
    pub fn is_found(&self) -> bool {
        !matches!(self, Self::Searching)
    }

    pub fn compatible_strides_for_inplace(&self, strides_inplace: &[usize]) -> bool {
        match self {
            ReferenceSelection::Concrete { strides, .. } => &**strides == strides_inplace,
            _ => false,
        }
    }
}
