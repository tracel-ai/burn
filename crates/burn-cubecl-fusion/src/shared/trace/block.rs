use crate::shared::{
    ir::{Arg, BinaryFuseArgs, FuseOp, FusePrecision, LayoutInfo, UnaryFuseArgs},
    settings::FuseSettings,
};
use burn_ir::{TensorId, TensorIr, TensorStatus};
use cubecl::prelude::Sequence;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, btree_map::Entry};

use super::{FuseResources, RegisteredTensors, TensorView};

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A block containing all [operations](FuseOp) as well as reads and writes for each tensor along
/// with the [fusion settings](FuseSettings).
pub struct FuseBlock {
    pub settings: FuseSettings,
    pub ops: Vec<FuseOp>,
    pub shape_ref: Vec<usize>,
    pub reads: BTreeMap<TensorId, Vec<FuseOp>>,
    pub writes: BTreeMap<TensorId, FuseOp>,
}

#[derive(Clone, Debug)]
/// It is responsible to build a [trace](FuseBlock).
pub struct FuseBlockBuilder {
    pub settings: FuseSettings,
    locals: LocalVariablePool,
    pub ops: Vec<FuseOp>,
    reads: BTreeMap<TensorId, Vec<FuseOp>>,
    bool_precision: FusePrecision,
    // Output declared in this block alone.
    outputs: RegisteredTensors,
    pub outputs_unhandled: Vec<Arg>,
    pub local_outputs: Vec<TensorId>,
}

impl FuseBlockBuilder {
    pub fn new(bool_precision: FusePrecision, settings: FuseSettings) -> Self {
        Self {
            bool_precision,
            settings,
            locals: Default::default(),
            ops: Default::default(),
            reads: Default::default(),
            outputs: Default::default(),
            outputs_unhandled: Default::default(),
            local_outputs: Default::default(),
        }
    }

    /// Register an output tensor.
    pub fn output(&mut self, tensor: &TensorIr, resources: &mut FuseResources) -> Option<Arg> {
        if resources.indexed.contains_key(&tensor.id) {
            return None;
        }
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_output = match precision {
            FusePrecision::Bool => self.bool_precision,
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
    pub fn input(&mut self, tensor: &TensorIr, resources: &mut FuseResources) -> Option<Arg> {
        if resources.indexed.contains_key(&tensor.id) {
            return None;
        }

        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FusePrecision::Bool => self.bool_precision,
            _ => precision,
        };

        let arg = match self.locals.get(precision, tensor.id) {
            Some(local) => {
                resources.inputs.update(tensor);
                // An input can be an output of a previously fused operation.
                // We need to flag the new status for the tensor.
                resources.outputs.update(tensor);
                self.outputs.update(tensor);

                local
            }
            None => {
                let new_input = resources.inputs.insert(precision_input, tensor.clone());
                let out = self.locals.create(precision, tensor.id);
                let input = Arg::Input(new_input, precision_input, LayoutInfo::Unknown);

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

    /// Register an input with swapped dims.
    pub fn input_swap_dims(
        &mut self,
        tensor: &TensorIr,
        output: &TensorIr,
        dims: (u32, u32),
        resources: &mut FuseResources,
    ) -> Option<Arg> {
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FusePrecision::Bool => self.bool_precision,
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
        let original = Arg::Input(input_index, precision_input, LayoutInfo::Unknown);

        let broadcasted = output.shape[output.shape.len() - 1] == 0;

        resources.views.push(TensorView::SwapDims {
            swapped: output.id,
            original: tensor.id,
            dims,
        });

        let input = Arg::InputSwapDims {
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
    ) -> Option<Arg> {
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as bool_precision.
        let precision_input = match precision {
            FusePrecision::Bool => self.bool_precision,
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
        let original = Arg::Input(input_index, precision_input, LayoutInfo::Unknown);

        let mut shape = Sequence::new();

        let index = resources.num_reshaped;
        resources.num_reshaped += 1;

        let rank = output.shape.len();

        for i in 0..output.shape.len() {
            let id = index * rank + i;
            shape.push(Arg::ScalarShape(id as u32));
        }

        resources.views.push(TensorView::Reshape {
            reshaped: output.id,
            original: tensor.id,
            reshape_pos: index as u32,
            shape_relative: output.shape.clone(),
        });

        let input = Arg::InputReshaped {
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
        shape_ref: Vec<usize>,
        offset: usize,
    ) -> (FuseBlock, RegisteredTensors) {
        let ops = self.ops.clone();
        let reads = self.reads.clone();
        let tensor_writes = self.tensor_writes(resources);

        let mut writes = BTreeMap::new();

        for (tensor, precision) in tensor_writes.iter() {
            if let Some(local) = self.locals.get_any_precision(tensor.id) {
                let out_index = tensor_writes.get_index(tensor.id).unwrap();

                writes.insert(
                    tensor.id,
                    FuseOp::Assign(UnaryFuseArgs {
                        input: local,
                        out: Arg::Output(
                            out_index + offset as u32,
                            *precision,
                            LayoutInfo::Unknown,
                        ),
                    }),
                );
            }
        }

        (
            FuseBlock {
                settings: self.settings,
                ops,
                shape_ref,
                reads,
                writes,
            },
            tensor_writes,
        )
    }

    pub fn estimate_num_outputs(&self, resources: &FuseResources) -> u32 {
        self.tensor_writes(resources).len() as u32
    }

    /// Return the tensor that needs to be written to.
    fn tensor_writes(&self, resources: &FuseResources) -> RegisteredTensors {
        let mut result = RegisteredTensors::default();

        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Arg, list: &mut Vec<(TensorId, FusePrecision)>| {
            if let Arg::Local(index, precision) = var
                && let Some(tensor_id) = self.locals.find_tensor_id(*precision, *index)
            {
                // Input and outputs tensors are using bool_precision for booleans.
                let precision = match precision {
                    FusePrecision::Bool => self.bool_precision,
                    _ => *precision,
                };

                let entry = (tensor_id, precision);
                if !list.contains(&entry) {
                    list.push(entry);
                }
            }
        };

        let mark_binary = |op: &BinaryFuseArgs,
                           inputs: &mut Vec<(TensorId, FusePrecision)>,
                           outputs: &mut Vec<(TensorId, FusePrecision)>| {
            mark(&op.lhs, inputs);
            mark(&op.rhs, inputs);
            mark(&op.out, outputs);
        };
        let mark_unary = |op: &UnaryFuseArgs,
                          inputs: &mut Vec<(TensorId, FusePrecision)>,
                          outputs: &mut Vec<(TensorId, FusePrecision)>| {
            mark(&op.input, inputs);
            mark(&op.out, outputs);
        };

        let mut mark_op = |op: &FuseOp| match op {
            FuseOp::Add(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),

            FuseOp::Sub(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Mul(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Div(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Powf(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Abs(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Exp(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Sqrt(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Log(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Log1p(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Cos(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Sin(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Tanh(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Erf(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Recip(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Assign(op) => mark_unary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => {
                mark(cond, &mut local_tensor_ids_input);
                mark(lhs, &mut local_tensor_ids_input);
                mark(rhs, &mut local_tensor_ids_input);
                mark(out, &mut local_tensor_ids_output);
            }
            FuseOp::Gather {
                input,
                indices,
                output,
                ..
            } => {
                mark(input, &mut local_tensor_ids_input);
                mark(indices, &mut local_tensor_ids_input);
                mark(output, &mut local_tensor_ids_output);
            }
            FuseOp::Select {
                input,
                indices,
                output,
                ..
            } => {
                mark(input, &mut local_tensor_ids_input);
                mark(indices, &mut local_tensor_ids_input);
                mark(output, &mut local_tensor_ids_output);
            }
            FuseOp::Equal(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Lower(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::Greater(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::LowerEqual(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
            FuseOp::GreaterEqual(op) => mark_binary(
                op,
                &mut local_tensor_ids_input,
                &mut local_tensor_ids_output,
            ),
        };

        // For all operators, mark their local tensor id in the proper set.
        for (_, ops) in self.reads.iter() {
            for op in ops {
                mark_op(op);
            }
        }

        for op in self.ops.iter() {
            mark_op(op);
        }

        for arg in self.outputs_unhandled.iter() {
            mark(arg, &mut local_tensor_ids_output);
        }

        // All output tensors that are never read by a following operation should be written to
        // since they are essentially the "logical" output of the shader.
        for entry in local_tensor_ids_output {
            let is_read = local_tensor_ids_input.contains(&entry);

            if !is_read
                && !self.local_outputs.contains(&entry.0)
                && !resources.dropped.contains(&entry.0)
            {
                let (tensor_id, precision) = entry;
                let (tensor, _) = resources.outputs.get(tensor_id).unwrap();
                result.insert(precision, tensor.clone());
            }
        }

        // All tensors where their latest representation is read only should be written to since they
        // are going to be used after the fused kernel by other operations.
        for (tensor, precision) in self.outputs.iter() {
            if let TensorStatus::ReadOnly = tensor.status
                && !resources.dropped.contains(&tensor.id)
            {
                result.insert(*precision, tensor.clone());
            }
        }

        result
    }
}

#[derive(Default, Clone, Debug)]
struct LocalVariablePool {
    values: BTreeMap<FusePrecision, BTreeMap<TensorId, u32>>,
}

impl LocalVariablePool {
    fn get(&self, precision: FusePrecision, tensor_id: TensorId) -> Option<Arg> {
        if let Some(indexes) = self.values.get(&precision)
            && let Some(index) = indexes.get(&tensor_id)
        {
            return Some(Arg::Local(*index, precision));
        }

        None
    }

    fn get_any_precision(&self, tensor_id: TensorId) -> Option<Arg> {
        for (precision, indexes) in self.values.iter() {
            if let Some(index) = indexes.get(&tensor_id) {
                return Some(Arg::Local(*index, *precision));
            }
        }

        None
    }

    fn find_tensor_id(&self, precision: FusePrecision, position: u32) -> Option<TensorId> {
        if let Some(indexes) = self.values.get(&precision) {
            indexes
                .iter()
                .find(|(_id, index)| **index == position)
                .map(|(id, _index)| *id)
        } else {
            None
        }
    }

    fn create(&mut self, precision: FusePrecision, tensor_id: TensorId) -> Arg {
        if let Some(indexes) = self.values.get_mut(&precision) {
            let new_index = indexes.len() as u32;
            indexes.insert(tensor_id, new_index);
            return Arg::Local(new_index, precision);
        }

        let new_index = 0;
        self.values
            .insert(precision, BTreeMap::from_iter([(tensor_id, new_index)]));

        Arg::Local(new_index, precision)
    }
}
