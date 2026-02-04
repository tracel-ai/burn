use super::{FuseResources, RegisteredTensors, TensorView};
use crate::engine::{
    codegen::ir::{FuseArg, FuseOp, FuseType, LayoutInfo, MultiBlockPos, UnaryFuseArgs},
    settings::FuseSettings,
};
use burn_ir::{TensorId, TensorIr, TensorStatus};
use burn_std::{DType, quantization::QuantParam};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, btree_map::Entry};

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A block containing all [operations](FuseOp) as well as reads and writes for each tensor along
/// with the [fusion settings](FuseSettings).
pub struct FuseBlock {
    /// Contains the [fusion settings](FuseSettings) associated to the current block.
    pub settings: FuseSettings,
    /// Contains all the [operations](FuseOp) registered in the current block.
    pub ops: Vec<FuseOp>,
    /// The reference shape of the current block.
    pub shape_ref: Vec<usize>,
    /// Contains all tensor inputs of the current block except for manually handled tensors.
    ///
    /// # Notes
    ///
    /// Some reads might not have read operations registered, such as dequantization, but it's
    /// important to be registered here for vectorization. Input tensors that are not
    /// registered here must be vectorized manually.
    pub reads: BTreeMap<TensorId, Vec<FuseOp>>,
    /// Contains all tensor outputs of the current block except for manually handled tensors.
    /// We can have multiple writes when the same variable is reused after in another block.
    pub writes: BTreeMap<TensorId, Vec<FuseOp>>,
}

#[derive(Clone, Debug)]
/// It is responsible to build a [trace](FuseBlock).
pub struct FuseBlockBuilder {
    pub settings: FuseSettings,
    locals: LocalVariablePool,
    pub ops: Vec<FuseOp>,
    reads: BTreeMap<TensorId, Vec<FuseOp>>,
    // Only for global registers.
    writes: BTreeMap<TensorId, Vec<FuseOp>>,
    bool_precision: FuseType,
    // Output declared in this block alone.
    outputs: RegisteredTensors,
    pub outputs_unhandled: Vec<FuseArg>,
    pub local_inputs: BTreeMap<TensorId, FuseArg>,
    /// The reference shape used by this block.
    pub shape_ref: Vec<usize>,
}

#[derive(Debug)]
/// How a quantized input can be read.
pub enum QuantInput {
    /// If already dequantized, we cache the dequantization and returns the local variable
    /// corresponding to the float value.
    AlreadyDequantized { local: FuseArg },
    /// Otherwise we return the information necessary to dequantize the tensor.
    Quantized { values: FuseArg, params: FuseArg },
}

impl FuseBlockBuilder {
    pub fn new(bool_precision: FuseType, settings: FuseSettings) -> Self {
        Self {
            bool_precision,
            settings,
            locals: Default::default(),
            ops: Default::default(),
            reads: Default::default(),
            writes: Default::default(),
            outputs: Default::default(),
            outputs_unhandled: Default::default(),
            local_inputs: Default::default(),
            shape_ref: Vec::new(),
        }
    }

    /// Register an output tensor.
    pub fn output(&mut self, tensor: &TensorIr, resources: &mut FuseResources) -> Option<FuseArg> {
        if resources.indexed.contains_key(&tensor.id) {
            return None;
        }
        if matches!(tensor.dtype, DType::QFloat(..)) {
            return None;
        }
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_output = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };

        let out = match self.locals.get(precision, tensor.id) {
            Some(local) => local,
            None => {
                let out = self.locals.create(precision, tensor.id);

                self.outputs.insert(precision_output, tensor.clone());
                resources.outputs.insert(precision_output, tensor.clone());

                out
            }
        };

        Some(out)
    }

    /// Register an input tensor.
    pub fn multi_block_variable(
        &mut self,
        block_pos: usize,
        tensor: &TensorIr,
        global: bool,
    ) -> Option<FuseArg> {
        let precision = tensor.dtype.into();

        if let Some(val) = self.local_inputs.get(&tensor.id) {
            return Some(val.clone());
        }

        let val = match self.locals.get(precision, tensor.id) {
            Some(val) => val,
            None => {
                return None;
            }
        };

        let arg = if global {
            FuseArg::MultiBlockGlobal(
                MultiBlockPos {
                    block_pos,
                    block_local_pos: self.writes.len(),
                },
                val.precision(),
            )
        } else {
            FuseArg::MultiBlockLocal(
                MultiBlockPos {
                    block_pos,
                    block_local_pos: self.writes.len(),
                },
                val.precision(),
            )
        };

        let ops = match self.writes.get_mut(&tensor.id) {
            Some(ops) => ops,
            None => {
                self.writes.insert(tensor.id, Vec::new());
                self.writes.get_mut(&tensor.id).unwrap()
            }
        };
        ops.push(FuseOp::Assign(UnaryFuseArgs {
            input: val,
            out: arg.clone(),
        }));

        Some(arg)
    }

    /// Register an input tensor.
    pub fn input(&mut self, tensor: &TensorIr, resources: &mut FuseResources) -> Option<FuseArg> {
        if resources.indexed.contains_key(&tensor.id) {
            return None;
        }

        if matches!(tensor.dtype, DType::QFloat(..)) {
            return None;
        }
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };

        if let Some(val) = self.local_inputs.get(&tensor.id) {
            return Some(val.clone());
        }

        let arg = match self.locals.get(precision, tensor.id) {
            Some(local) => {
                resources.inputs.update(tensor);

                local
            }
            None => {
                let input = if resources.outputs.get_index(tensor.id).is_some() {
                    if let Some(val) = resources.registers.get(&tensor.id) {
                        return Some(val.clone());
                    };

                    let pos = resources.buffers.insert(precision, tensor.clone());
                    FuseArg::Output(pos, precision_input, LayoutInfo::Unknown)
                } else {
                    let pos = resources.inputs.insert(precision_input, tensor.clone());
                    FuseArg::Input(pos, precision_input, LayoutInfo::Unknown)
                };

                let out = self.locals.create(precision, tensor.id);

                let reads = if let Entry::Vacant(e) = self.reads.entry(tensor.id) {
                    e.insert(Vec::with_capacity(1));
                    self.reads.get_mut(&tensor.id).unwrap()
                } else {
                    self.reads.get_mut(&tensor.id).unwrap()
                };

                reads.push(FuseOp::Assign(UnaryFuseArgs {
                    input,
                    out: out.clone(),
                }));

                out
            }
        };

        Some(arg)
    }

    /// Register an input quantized tensor.
    pub fn input_quant(
        &mut self,
        tensor: &TensorIr,
        resources: &mut FuseResources,
    ) -> Option<QuantInput> {
        if resources.indexed.contains_key(&tensor.id) {
            return None;
        }

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

        let arg = match self.locals.get(precision, tensor.id) {
            Some(local) => {
                resources.inputs.update(tensor);
                QuantInput::AlreadyDequantized { local }
            }
            None => {
                let (new_input, q_index) = resources.inputs.insert_quant(tensor.clone());
                let input = FuseArg::Input(new_input, precision, LayoutInfo::Unknown);
                let scales = FuseArg::Input(q_index, precision_scales, LayoutInfo::Unknown);

                // Important to flag that there is a read, even if no operation is registered.
                if let Entry::Vacant(e) = self.reads.entry(tensor.id) {
                    e.insert(Vec::new());
                };

                QuantInput::Quantized {
                    values: input,
                    params: scales,
                }
            }
        };

        Some(arg)
    }

    /// Register an input with swapped dims.
    pub fn input_swap_dims(
        &mut self,
        tensor: &TensorIr,
        output: &TensorIr,
        dims: (usize, usize),
        resources: &mut FuseResources,
    ) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(..)) {
            return None;
        }
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };

        let input_index = match self.locals.get(precision, tensor.id) {
            Some(_) => {
                // Can't fused an already fused input.
                if resources.outputs.get(tensor.id).is_some() {
                    return None;
                }

                match resources.inputs.get_index(tensor.id) {
                    Some(index) => {
                        resources.inputs.update(tensor);
                        index
                    }
                    None => {
                        return None;
                    }
                }
            }
            None => resources.inputs.insert(precision_input, tensor.clone()),
        };

        let out = self.output(output, resources)?;
        let original = FuseArg::Input(input_index, precision_input, LayoutInfo::Unknown);

        let broadcasted = output.shape[output.shape.rank() - 1] == 0;

        resources.views.push(TensorView::SwapDims {
            swapped: output.id,
            original: tensor.id,
            dims,
        });

        let input = FuseArg::InputSwapDims {
            original: Box::new(original),
            dims,
            broadcasted,
        };

        let reads = if let Entry::Vacant(e) = self.reads.entry(tensor.id) {
            e.insert(Vec::with_capacity(1));
            self.reads.get_mut(&tensor.id).unwrap()
        } else {
            self.reads.get_mut(&tensor.id).unwrap()
        };

        reads.push(FuseOp::Assign(UnaryFuseArgs {
            input,
            out: out.clone(),
        }));

        Some(out)
    }

    /// Register an input that is reshaped.
    pub fn input_reshaped(
        &mut self,
        tensor: &TensorIr,
        output: &TensorIr,
        resources: &mut FuseResources,
    ) -> Option<FuseArg> {
        if matches!(tensor.dtype, DType::QFloat(..)) {
            return None;
        }
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FuseType::Bool => self.bool_precision,
            _ => precision,
        };

        let input_index = match self.locals.get(precision, tensor.id) {
            Some(_) => {
                // Can't fused an already fused input.
                if resources.outputs.get(tensor.id).is_some() {
                    return None;
                }

                match resources.inputs.get_index(tensor.id) {
                    Some(index) => {
                        resources.inputs.update(tensor);
                        index
                    }
                    None => {
                        return None;
                    }
                }
            }
            None => resources.inputs.insert(precision_input, tensor.clone()),
        };

        let out = self.output(output, resources)?;
        let original = FuseArg::Input(input_index, precision_input, LayoutInfo::Unknown);

        let mut shape = Vec::new();

        let index = resources.num_reshaped;
        resources.num_reshaped += 1;

        let rank = output.shape.rank();

        for i in 0..output.shape.rank() {
            let id = index * rank + i;
            shape.push(FuseArg::ScalarShape(id));
        }

        resources.views.push(TensorView::Reshape {
            reshaped: output.id,
            original: tensor.id,
            reshape_pos: index,
            shape_relative: output.shape.dims.clone(),
        });

        let input = FuseArg::InputReshaped {
            original: Box::new(original),
            shape,
            broadcasted: output.shape[rank - 1] == 0,
        };

        let reads = if let Entry::Vacant(e) = self.reads.entry(tensor.id) {
            e.insert(Vec::with_capacity(1));
            self.reads.get_mut(&tensor.id).unwrap()
        } else {
            self.reads.get_mut(&tensor.id).unwrap()
        };

        reads.push(FuseOp::Assign(UnaryFuseArgs {
            input,
            out: out.clone(),
        }));

        Some(out)
    }

    /// Build into a fuse block.
    pub fn build(
        &self,
        resources: &FuseResources,
        outputs: &mut RegisteredTensors,
        buffers: &mut Vec<TensorId>,
    ) -> FuseBlock {
        let ops = self.ops.clone();
        let reads = self.reads.clone();
        let tensor_writes = self.tensor_writes(resources, buffers);

        let mut writes = self.writes.clone();

        for (tensor, precision) in tensor_writes
            .iter()
            .filter_map(|entry| entry.as_normal_tensor())
        {
            if let Some(local) = self.locals.get_any_precision(tensor.id) {
                let out_index = outputs.insert(*precision, tensor.clone());

                let ops = match writes.get_mut(&tensor.id) {
                    Some(ops) => ops,
                    None => {
                        writes.insert(tensor.id, Vec::new());
                        writes.get_mut(&tensor.id).unwrap()
                    }
                };

                ops.push(FuseOp::Assign(UnaryFuseArgs {
                    input: local,
                    out: FuseArg::Output(out_index, *precision, LayoutInfo::Unknown),
                }));
            }
        }

        FuseBlock {
            settings: self.settings,
            ops,
            shape_ref: self.shape_ref.clone(),
            reads,
            writes,
        }
    }

    /// Return the tensor that needs to be written to.
    ///
    /// # Notes
    ///
    /// The buffers vector passed as input is only to track the intermediary buffer writes needed
    /// during execution.
    pub fn tensor_writes(
        &self,
        resources: &FuseResources,
        buffers: &mut Vec<TensorId>,
    ) -> RegisteredTensors {
        let mut result = RegisteredTensors::default();

        // All tensors where their latest representation is not read write should be written to since they
        // are going to be used after the fused kernel by other operations.
        for output in self.outputs.iter() {
            if let Some((tensor, _precision)) = output.as_normal_tensor() {
                // We get the latest representation from the resources, not just this block.
                if let Some((tensor, precision)) = resources.outputs.get(tensor.id) {
                    if !matches!(tensor.status, TensorStatus::ReadWrite) {
                        result.insert(*precision, tensor.clone());
                    } else if resources.buffers.get(tensor.id).is_some()
                        && !buffers.contains(&tensor.id)
                    {
                        result.insert(*precision, tensor.clone());
                        // We make sure we don't write multiple time in the same buffer, only the
                        // earliest possible.
                        buffers.push(tensor.id);
                    }
                }
            }
        }

        result
    }
}

#[derive(Default, Clone, Debug)]
pub struct LocalVariablePool {
    values: BTreeMap<FuseType, BTreeMap<TensorId, usize>>,
}

impl LocalVariablePool {
    fn get(&self, precision: FuseType, tensor_id: TensorId) -> Option<FuseArg> {
        if let Some(indexes) = self.values.get(&precision)
            && let Some(index) = indexes.get(&tensor_id)
        {
            return Some(FuseArg::BlockLocal {
                pos: *index,
                ty: precision,
            });
        }

        None
    }

    fn get_any_precision(&self, tensor_id: TensorId) -> Option<FuseArg> {
        for (precision, indexes) in self.values.iter() {
            if let Some(index) = indexes.get(&tensor_id) {
                return Some(FuseArg::BlockLocal {
                    pos: *index,
                    ty: *precision,
                });
            }
        }

        None
    }

    fn create(&mut self, precision: FuseType, tensor_id: TensorId) -> FuseArg {
        if let Some(indexes) = self.values.get_mut(&precision) {
            let new_index = indexes.len();
            indexes.insert(tensor_id, new_index);
            return FuseArg::BlockLocal {
                pos: new_index,
                ty: precision,
            };
        }

        let new_index = 0;
        self.values
            .insert(precision, BTreeMap::from_iter([(tensor_id, new_index)]));

        FuseArg::BlockLocal {
            pos: new_index,
            ty: precision,
        }
    }
}
