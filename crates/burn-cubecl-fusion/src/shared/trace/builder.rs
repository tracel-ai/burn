use crate::shared::trace::block::QuantInput;

use super::{
    super::{
        ir::{Arg, FuseOp, FusePrecision, LayoutInfo},
        settings::FuseSettings,
    },
    FuseResources, RegisterTensor,
    block::FuseBlockBuilder,
};
use super::{FuseTrace, RegisteredTensors};
use burn_fusion::stream::ScalarId;
use burn_ir::{ScalarIr, TensorId, TensorIr};
use burn_tensor::DType;

#[derive(Clone, Debug)]
/// It is responsible to create a [trace](FuseTrace) composed of multiple [blocks](super::block::FuseBlock).
///
/// It mostly handles the [resources](KernelResources) needed by the generated fused kernel, and
/// delegates most of the work to the [block builder](FuseBlockBuilder).
pub struct FuseTraceBuilder {
    settings: FuseSettings,
    pub bool_precision: FusePrecision,
    // The tensors returned by the block that don't need to be written to global memory.
    block_current: FuseBlockBuilder,
    blocks_previous: Vec<(FuseBlockBuilder, Vec<usize>)>,
    resources: FuseResources,
}

impl FuseTraceBuilder {
    /// Create a new trace builder with the given bool precision and [fuse settings](FuseSettings).
    pub fn new(bool_precision: FusePrecision, settings: FuseSettings) -> Self {
        Self {
            settings,
            bool_precision,
            block_current: FuseBlockBuilder::new(bool_precision, settings),
            blocks_previous: Default::default(),
            resources: Default::default(),
        }
    }

    /// Tag a tensor as dropped.
    pub fn register_dropped(&mut self, id: TensorId) {
        self.resources.dropped.insert(id);
    }

    /// Register an operation.
    pub fn register_operation(&mut self, op: FuseOp) {
        self.block_current.ops.push(op);
    }

    /// The number of operations fused.
    pub fn num_ops_fused(&self) -> u32 {
        let mut num_ops_fused = 0;

        for (block, _) in self.blocks_previous.iter() {
            num_ops_fused += block.ops.len();
        }

        num_ops_fused += self.block_current.ops.len();
        num_ops_fused as u32
    }

    /// Close the current block with the given reference shape and creates a new one with new [fusion settings](FuseSettings).
    pub fn next_block(&mut self, shape_ref: Vec<usize>, settings: FuseSettings) {
        let mut block_new = FuseBlockBuilder::new(self.bool_precision, settings);
        core::mem::swap(&mut self.block_current, &mut block_new);
        self.blocks_previous.push((block_new, shape_ref));
        self.settings = settings;
    }

    // Estimate how many bindings are in use right now. This can return more than the actual number
    // but should never return less.
    pub fn estimate_bindings(&self) -> u32 {
        let mut estimation = 1; // Metadata takes one.
        for b in self.blocks_previous.iter() {
            estimation += b.0.estimate_num_outputs(&self.resources);
        }
        estimation += self.block_current.estimate_num_outputs(&self.resources);
        estimation += self.resources.inputs.len() as u32;
        // One buffer per scalar type for now.
        estimation += self.resources.scalars.len() as u32;

        estimation
    }

    /// Tag the [tensor](TensorIr) to be the logical returned value of the current block.
    ///
    /// This will avoid the output to be written in global memory when not necessary.
    pub fn block_local_output(&mut self, tensor: &TensorIr) {
        self.block_current.local_outputs.push(tensor.id);
    }

    /// Register an output tensor that won't be automatically synced into global memory.
    ///
    /// It is therefore the responsibility of the operation to write the result to given tensor.
    pub fn output_unhandled(&mut self, tensor: &TensorIr) -> Arg {
        let arg = self
            .output(tensor)
            .expect("Can't add a new output that is already used in an index operation");
        self.resources.outputs_unhandled.push(arg.clone());
        self.block_current.outputs_unhandled.push(arg.clone());
        arg
    }

    /// Register an input tensor that won't be automatically read into a local variable.
    ///
    /// It is therefore the responsibility of the operation to read the given tensor.
    pub fn input_unhandled(&mut self, tensor: &TensorIr) -> Arg {
        if self.resources.indexed.contains_key(&tensor.id) {
            panic!("Can't add a new input that is already used in an index operation");
        }

        let precision = tensor.dtype.into();
        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FusePrecision::Bool => self.bool_precision,
            _ => precision,
        };
        let new_input = self
            .resources
            .inputs
            .insert(precision_input, tensor.clone());
        let arg = Arg::Input(new_input, precision_input, LayoutInfo::Unknown);

        self.resources.inputs_unhandled.push(tensor.id);
        arg
    }

    /// Register an input tensor.
    pub fn input(&mut self, tensor: &TensorIr) -> Option<Arg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        self.block_current.input(tensor, &mut self.resources)
    }

    /// Register an input tensor.
    pub fn input_quantized(&mut self, tensor: &TensorIr) -> Option<QuantInput> {
        self.block_current.input_quant(tensor, &mut self.resources)
    }

    /// Register an output tensor.
    pub fn output(&mut self, tensor: &TensorIr) -> Option<Arg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }
        self.block_current.output(tensor, &mut self.resources)
    }

    /// Register an input that will be accessed using custom indexing with no vectorization.
    pub fn input_indexed(&mut self, tensor: &TensorIr) -> Option<Arg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        if let Some(val) = self.resources.indexed.get(&tensor.id) {
            return Some(val.clone());
        };

        if self.resources.inputs.get(tensor.id).is_some() {
            return None;
        }

        if self.resources.outputs.get(tensor.id).is_some() {
            return None;
        }

        let input = self.input_unhandled(tensor);
        self.resources.indexed.insert(tensor.id, input.clone());

        Some(input)
    }

    /// Register an input with swapped dims.
    pub fn input_swap_dims(
        &mut self,
        tensor: &TensorIr,
        output: &TensorIr,
        dims: (u32, u32),
    ) -> Option<Arg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }
        self.block_current
            .input_swap_dims(tensor, output, dims, &mut self.resources)
    }

    /// Register an input that is reshaped.
    pub fn input_reshaped(&mut self, tensor: &TensorIr, output: &TensorIr) -> Option<Arg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        self.block_current
            .input_reshaped(tensor, output, &mut self.resources)
    }

    /// Register a scalar value.
    pub fn scalar(&mut self, elem: &ScalarIr, dtype: DType) -> Arg {
        let precision = dtype.into();
        let id = if let ScalarIr::U64(value) = elem {
            ScalarId { value: *value }
        } else {
            unreachable!() // should always be u64
        };

        // Bool scalars are encoded as bool_precision.
        let precision = match precision {
            FusePrecision::Bool => self.bool_precision,
            _ => precision,
        };
        let new_index = self.resources.scalars.len() as u32;

        self.resources.scalars.push((precision, id.value));
        Arg::Scalar(new_index, precision)
    }

    /// Build into a trace.
    pub fn build(&self, shape_ref: Vec<usize>) -> FuseTrace {
        let mut resources = self.resources.clone();
        let mut outputs = RegisteredTensors::default();
        let mut blocks = Vec::new();

        let mut register_block =
            |block: &FuseBlockBuilder, shape_ref: &Vec<usize>, offset: usize| {
                let (block, block_tensor_writes) =
                    block.build(&self.resources, shape_ref.clone(), offset);
                blocks.push(block);

                let num_outputs = block_tensor_writes.len();
                for entry in block_tensor_writes.into_iter() {
                    match entry {
                        RegisterTensor::Normal(ir, precision) => {
                            outputs.insert(precision, ir);
                        }
                        _ => {
                            panic!("Quantized tensor unsupported for output")
                        }
                    }
                }

                num_outputs
            };

        let mut offset = 0;

        for (block, shape_ref) in self.blocks_previous.iter() {
            offset += register_block(block, shape_ref, offset);
        }
        register_block(&self.block_current, &shape_ref, offset);

        // We update the output tensors registered to be the ones that are written to in global
        // memory.
        resources.outputs = outputs;

        FuseTrace { blocks, resources }
    }
}
