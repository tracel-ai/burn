use super::{
    super::{
        codegen::ir::{FuseArg, FuseOp, FuseType, LayoutInfo},
        settings::FuseSettings,
    },
    FuseResources,
    block::FuseBlockBuilder,
};
use super::{FuseTrace, RegisteredTensors};
use crate::engine::trace::block::QuantInput;
use burn_fusion::stream::ScalarId;
use burn_ir::{ScalarIr, TensorIr};
use burn_std::{DType, Shape};
use cubecl::quant::scheme::QuantParam;

#[derive(Clone, Debug)]
/// It is responsible to create a [trace](FuseTrace) composed of multiple [blocks](super::block::FuseBlock).
///
/// It mostly handles the [resources](KernelResources) needed by the generated fused kernel, and
/// delegates most of the work to the [block builder](FuseBlockBuilder).
pub struct TraceFuser {
    settings: FuseSettings,
    pub bool_precision: FuseType,
    // The tensors returned by the block that don't need to be written to global memory.
    block_current: FuseBlockBuilder,
    blocks_previous: Vec<FuseBlockBuilder>,
    resources: FuseResources,
}

impl TraceFuser {
    /// Create a new trace builder with the given bool precision and [fuse settings](FuseSettings).
    pub fn new(bool_precision: FuseType, settings: FuseSettings) -> Self {
        Self {
            settings,
            bool_precision,
            block_current: FuseBlockBuilder::new(bool_precision, settings),
            blocks_previous: Default::default(),
            resources: Default::default(),
        }
    }

    /// Get the number of blocks that are closed.
    pub fn num_previous_blocks(&self) -> usize {
        self.blocks_previous.len()
    }

    /// Tag a tensor as dropped.
    pub fn fuse_dropped(&mut self, tensor: &TensorIr) {
        self.resources.outputs.update(tensor);
        self.resources.inputs.update(tensor);
        self.resources.dropped.insert(tensor.id);
    }

    /// Register an operation.
    pub fn fuse_operation(&mut self, op: FuseOp) {
        self.block_current.ops.push(op);
    }

    /// The number of operations fused.
    pub fn num_ops_fused(&self) -> u32 {
        let mut num_ops_fused = 0;

        for block in self.blocks_previous.iter() {
            num_ops_fused += block.ops.len();
        }

        num_ops_fused += self.block_current.ops.len();
        num_ops_fused as u32
    }

    /// Close the current block with the given reference shape and creates a new one with new [fusion settings](FuseSettings).
    pub fn next_block(&mut self, shape_ref: Shape, settings: FuseSettings) {
        let mut block_new = FuseBlockBuilder::new(self.bool_precision, settings);
        core::mem::swap(&mut self.block_current, &mut block_new);
        block_new.shape_ref = shape_ref;
        self.blocks_previous.push(block_new);
        self.settings = settings;
    }

    // Estimate how many bindings are in use right now. This can return more than the actual number
    // but should never return less.
    pub fn estimate_bindings(&self) -> u32 {
        let mut buffers = Vec::new();
        let mut estimation = 1; // Metadata takes one.

        // We assume we are not going to write multiple times in the same output buffer.
        for b in self.blocks_previous.iter() {
            estimation += b.tensor_writes(&self.resources, &mut buffers).len() as u32;
        }

        estimation += self
            .block_current
            .tensor_writes(&self.resources, &mut buffers)
            .len() as u32;
        estimation += self.resources.inputs.len() as u32;
        // One buffer per scalar type for now.
        estimation += self.resources.scalars.len() as u32;

        estimation
    }

    /// Tag the [tensor](TensorIr) as received from a previous block.
    ///
    /// This will avoid reading the input again and instead use le local version when possible.
    pub fn block_local_input(
        &mut self,
        tensor: &TensorIr,
        block_pos: usize,
        global: bool,
    ) -> FuseArg {
        let block = &mut self.blocks_previous[block_pos];

        let src_arg = match block.multi_block_variable(block_pos, tensor, global) {
            Some(val) => val,
            None => {
                // We try to read the input if not present.
                block.input(tensor, &mut self.resources);
                block
                    .multi_block_variable(block_pos, tensor, global)
                    .unwrap()
            }
        };

        self.resources.outputs.update(tensor);

        if global {
            self.resources.registers.insert(tensor.id, src_arg.clone());
        }

        self.block_current
            .local_inputs
            .insert(tensor.id, src_arg.clone());
        src_arg
    }

    /// Register an output tensor that won't be automatically synced into global memory.
    ///
    /// It is therefore the responsibility of the operation to write the result to given tensor.
    pub fn output_unhandled(&mut self, tensor: &TensorIr) -> FuseArg {
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
    pub fn input_unhandled(&mut self, tensor: &TensorIr) -> FuseArg {
        if self.resources.indexed.contains_key(&tensor.id) {
            panic!("Can't add a new input that is already used in an index operation");
        }

        self.resources.outputs.update(tensor);

        let precision = tensor.dtype.into();
        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };
        let new_input = self
            .resources
            .inputs
            .insert(precision_input, tensor.clone());
        let arg = FuseArg::Input(new_input, precision_input, LayoutInfo::Unknown);

        self.resources.inputs_unhandled.push(tensor.id);
        arg
    }

    /// Register an input tensor.
    pub fn input_quantized_unhandled(&mut self, tensor: &TensorIr) -> Option<(FuseArg, FuseArg)> {
        if self.resources.indexed.contains_key(&tensor.id) {
            panic!("Can't add a new input that is already used in an index operation");
        }
        self.resources.outputs.update(tensor);

        let precision = tensor.dtype.into();
        let precision_scales = match tensor.dtype {
            DType::QFloat(scheme) => match scheme.param {
                QuantParam::F32 => FuseType::F32,
                QuantParam::F16 => FuseType::F16,
                QuantParam::BF16 => FuseType::BF16,
                QuantParam::UE8M0 | QuantParam::UE4M3 => {
                    unimplemented!("Unsupported fuse precision");
                }
            },
            _ => return None,
        };

        let (new_input, q_index) = self.resources.inputs.insert_quant(tensor.clone());
        let input = FuseArg::Input(new_input, precision, LayoutInfo::Unknown);
        let scales = FuseArg::Input(q_index, precision_scales, LayoutInfo::Unknown);

        self.resources.inputs_unhandled.push(tensor.id);
        Some((input, scales))
    }

    /// Register an input tensor.
    pub fn input(&mut self, tensor: &TensorIr) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        self.resources.outputs.update(tensor);

        self.block_current.input(tensor, &mut self.resources)
    }

    /// Register an input tensor.
    pub fn input_quantized(&mut self, tensor: &TensorIr) -> Option<QuantInput> {
        self.resources.outputs.update(tensor);
        self.block_current.input_quant(tensor, &mut self.resources)
    }

    /// Register an output tensor.
    pub fn output(&mut self, tensor: &TensorIr) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }
        self.block_current.output(tensor, &mut self.resources)
    }

    /// Register an input that will be accessed using custom indexing with no vectorization.
    pub fn input_indexed(&mut self, tensor: &TensorIr) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        if let Some(val) = self.resources.indexed.get(&tensor.id) {
            self.resources.outputs.update(tensor);
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
        dims: (usize, usize),
    ) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        self.resources.outputs.update(tensor);
        self.block_current
            .input_swap_dims(tensor, output, dims, &mut self.resources)
    }

    /// Register an input that is reshaped.
    pub fn input_reshaped(&mut self, tensor: &TensorIr, output: &TensorIr) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return None;
        }

        self.resources.outputs.update(tensor);
        self.block_current
            .input_reshaped(tensor, output, &mut self.resources)
    }

    /// Register a scalar value.
    pub fn scalar(&mut self, elem: &ScalarIr, dtype: DType) -> FuseArg {
        let precision = dtype.into();
        let id = if let ScalarIr::UInt(value) = elem {
            ScalarId { value: *value }
        } else {
            unreachable!() // should always be u64
        };

        // Bool scalars are encoded as bool_precision.
        let precision = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };
        let new_index = self.resources.scalars.len();

        self.resources.scalars.push((precision, id.value));
        FuseArg::Scalar(new_index, precision)
    }

    /// Finish fusing and returns the created trace.
    pub fn finish(&mut self, shape_ref: Shape) -> FuseTrace {
        let mut resources = self.resources.clone();
        let mut outputs = RegisteredTensors::default();
        let mut buffers = Vec::new();

        for tensor in resources.buffers.iter() {
            let (tensor, ty) = tensor.as_normal_tensor().unwrap();
            outputs.insert(*ty, tensor.clone());
        }

        let mut blocks = Vec::new();

        let mut register_block = |block: &FuseBlockBuilder| {
            let block = block.build(&self.resources, &mut outputs, &mut buffers);
            blocks.push(block);
        };

        for block in self.blocks_previous.iter() {
            register_block(block);
        }
        self.block_current.shape_ref = shape_ref;
        register_block(&self.block_current);

        // We update the output tensors registered to be the ones that are written to in global
        // memory.
        resources.outputs = outputs;

        FuseTrace { blocks, resources }
    }
}
