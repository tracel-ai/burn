use std::{collections::HashMap, ops::Range};

use burn_backend::{
    BoolDType, DType, DTypeUsageSet, DeviceId, ExecutionError, Scalar, Shape, TensorData,
    TensorMetadata,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_ir::{
    BackendIr, GraphBindings, GraphId, HandleKind, IrVisitorMut, LinearOpIr, ModuleOperationIr,
    NumericOperationIr, OperationIr, ReduceDimOpIr, ReduceOpIr, ScalarIr, ScalarOpIr, TensorId,
    TensorIr, TensorStatus,
};
use burn_router::{Graph, TensorInterpreter};
use burn_std::{device::Device, future::DynFut};

use super::{
    plan::{Execution, OpPlan},
    shard::ShardList,
};
use crate::{Chunks, EmbeddingRule, GroupPlacement, Redistribution};

/// A device group's interpreter, whatever backend its members run: what its client asks of it.
pub trait GroupInterpreter: Send {
    fn new_tensor_id(&mut self) -> TensorId;
    fn placement(&self, id: &TensorId) -> GroupPlacement;
    /// Every member holds the whole value.
    fn register_tensor_data(&mut self, data: TensorData) -> TensorId;
    fn register_op(&mut self, op: OperationIr);
    /// The same value at another placement, under a new id.
    ///
    /// # Panics
    ///
    /// When no collective produces `placement`: a partial sum out of a whole tensor, or a split
    /// along a dim shorter than the group.
    fn place(&mut self, tensor: TensorIr, placement: GroupPlacement) -> TensorId;
    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>>;
    fn register_alias(&mut self, new_id: TensorId, src_id: TensorId);
    fn register_graph(
        &mut self,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    );
    fn execute_graph(&mut self, graph_id: GraphId, bindings: GraphBindings);
    fn sync(&mut self) -> Result<(), ExecutionError>;
    fn flush(&mut self);
    fn seed(&self, seed: u64);
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet;
}

/// Runs every op of a device group on each member's interpreter, with each tensor at the
/// placement its op's rule asks for.
pub struct GroupExecutor<B: BackendIr> {
    members: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    placements: HashMap<TensorId, GroupPlacement>,
    graphs: HashMap<GraphId, Graph>,
    next_id: u64,
    flush_error: Option<ExecutionError>,
}

impl<B: BackendIr> GroupExecutor<B> {
    pub fn new(devices: Vec<B::Device>) -> Self {
        Self {
            members: devices
                .iter()
                .map(|device| TensorInterpreter::new(device.clone()))
                .collect(),
            devices,
            placements: HashMap::new(),
            graphs: HashMap::new(),
            next_id: 0,
            flush_error: None,
        }
    }

    /// The interpreter of a group whose members run `B` on `devices`.
    pub fn boxed(devices: &[DeviceId]) -> Box<dyn GroupInterpreter> {
        let devices = devices.iter().copied().map(B::Device::from_id).collect();
        Box::new(Self::new(devices))
    }
}

impl<B: BackendIr> GroupInterpreter for GroupExecutor<B> {
    fn new_tensor_id(&mut self) -> TensorId {
        self.next_id += 1;
        TensorId::new(self.next_id)
    }

    fn placement(&self, id: &TensorId) -> GroupPlacement {
        self.placements[id]
    }

    fn register_tensor_data(&mut self, data: TensorData) -> TensorId {
        let id = self.new_tensor_id();
        for member in &mut self.members {
            member.register_tensor_data_id(id, data.clone());
        }
        self.placements.insert(id, GroupPlacement::Replicated);
        id
    }

    fn register_op(&mut self, op: OperationIr) {
        match op {
            // The data was registered on every member with the tensor.
            OperationIr::Init(_) => {}
            OperationIr::Drop(tensor) => self.drop(tensor.id),
            OperationIr::Distributed(_) => {
                panic!("A device group runs its own collectives and takes none from outside")
            }
            op => self.run(op),
        }
    }

    fn place(&mut self, tensor: TensorIr, placement: GroupPlacement) -> TensorId {
        let current = self.placements[&tensor.id];
        let redistribution =
            Redistribution::new(current, placement, &tensor.shape, self.members.len())
                .unwrap_or_else(|| panic!("Cannot place a {current:?} tensor {placement:?}"));
        let shards = self.take_shards(&tensor, current).redistribute(
            redistribution,
            &tensor.shape,
            &self.devices,
        );
        let id = self.new_tensor_id();
        self.register_shards(id, shards, placement);
        id
    }

    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let placement = self.placements[&tensor.id];
        self.take_shards(&tensor, placement)
            .into_data(placement, &self.devices[0])
    }

    fn register_alias(&mut self, new_id: TensorId, src_id: TensorId) {
        for member in &mut self.members {
            member.register_alias(new_id, src_id);
        }
        self.placements.insert(new_id, self.placements[&src_id]);
    }

    fn register_graph(
        &mut self,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    ) {
        let graph = Graph::new(relative_graph);
        let operations = graph.bind(bindings).operations;
        self.graphs.insert(graph_id, graph);
        operations.into_iter().for_each(|op| self.register_op(op));
    }

    fn execute_graph(&mut self, graph_id: GraphId, bindings: GraphBindings) {
        let operations = self.graphs[&graph_id].bind(bindings).operations;
        operations.into_iter().for_each(|op| self.register_op(op));
    }

    fn sync(&mut self) -> Result<(), ExecutionError> {
        if let Some(error) = self.flush_error.take() {
            return Err(error);
        }
        self.members.iter().try_for_each(TensorInterpreter::sync)
    }

    fn flush(&mut self) {
        for device in &self.devices {
            if let Err(error) = B::flush(device) {
                self.flush_error.get_or_insert(error);
            }
        }
    }

    fn seed(&self, seed: u64) {
        self.members.iter().for_each(|member| member.seed(seed));
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        self.members[0].dtype_usage(dtype)
    }
}

impl<B: BackendIr> GroupExecutor<B> {
    fn run(&mut self, mut op: OperationIr) {
        let plan = OpPlan::new(&op, &self.placements, self.members.len());
        plan.take_split_dims_whole(&mut op);
        let temporaries = self.redistribute_inputs(&mut op, &plan);
        for (id, placement) in plan.outputs() {
            self.placements.insert(*id, *placement);
        }

        match plan.execution() {
            Execution::EveryMember => {
                (0..self.members.len()).for_each(|member| self.run_on(member, &op))
            }
            Execution::FirstMemberCopied => self.run_copied_from_first_member(&op),
            Execution::BiasOnFirstMember => self.run_with_bias_on_first_member(&op),
            Execution::SumThenDivide { count } => self.run_sum_then_divide(&op, count),
            Execution::VocabLookup { chunks } => self.run_vocab_lookup(&op, chunks),
            Execution::VocabBackward { chunks } => self.run_vocab_backward(&op, chunks),
        }

        for input in op.inputs() {
            if input.status == TensorStatus::ReadWrite {
                self.placements.remove(&input.id);
            }
        }
        temporaries.into_iter().for_each(|id| self.drop(id));
    }

    /// Moves every input whose placement differs from its target into a temporary, which the
    /// op reads instead.
    fn redistribute_inputs(&mut self, op: &mut OperationIr, plan: &OpPlan) -> Vec<TensorId> {
        let inputs: Vec<TensorIr> = op.inputs().cloned().collect();
        let mut renames = Renames::default();
        for input in &inputs {
            if renames.ids.contains_key(&input.id) {
                continue;
            }
            let (current, target) = (self.placements[&input.id], plan.input(&input.id));
            if current == target {
                continue;
            }
            let consumed = inputs
                .iter()
                .any(|other| other.id == input.id && other.status == TensorStatus::ReadWrite);
            let read = TensorIr {
                status: match consumed {
                    true => TensorStatus::ReadWrite,
                    false => TensorStatus::ReadOnly,
                },
                ..input.clone()
            };
            let redistribution =
                Redistribution::new(current, target, &input.shape, self.members.len())
                    .unwrap_or_else(|| panic!("A rule asked for {target:?} from {current:?}"));
            let shards = self.take_shards(&read, current).redistribute(
                redistribution,
                &input.shape,
                &self.devices,
            );
            let temporary = self.new_tensor_id();
            self.register_shards(temporary, shards, target);
            renames.ids.insert(input.id, temporary);
        }
        op.visit_mut(&mut renames);
        renames.ids.into_values().collect()
    }

    fn run_on(&mut self, member: usize, op: &OperationIr) {
        let mut op = op.clone();
        op.visit_mut(&mut MemberShapes {
            member,
            members: self.members.len(),
            placements: &self.placements,
        });
        self.members[member].register_op(op);
    }

    fn run_copied_from_first_member(&mut self, op: &OperationIr) {
        self.run_on(0, op);
        for output in op.outputs() {
            let read = TensorIr {
                status: TensorStatus::ReadOnly,
                ..output.clone()
            };
            let copies = ShardList::replicated(self.members[0].get_tensor(&read), &self.devices);
            for (member, copy) in self.members.iter_mut().zip(copies.into_handles()) {
                member.register_tensor_to_device(output.id, copy);
            }
        }
    }

    fn run_with_bias_on_first_member(&mut self, op: &OperationIr) {
        let OperationIr::Module(ModuleOperationIr::Linear(desc)) = op else {
            unreachable!("Only a linear adds a bias")
        };
        let bias = desc.bias.clone().expect("A linear with a bias");
        let without_bias = OperationIr::Module(ModuleOperationIr::Linear(LinearOpIr {
            bias: None,
            ..desc.clone()
        }));
        self.run_on(0, op);
        for member in 1..self.members.len() {
            self.run_on(member, &without_bias);
            if bias.status == TensorStatus::ReadWrite {
                self.members[member].register_op(OperationIr::Drop(bias.clone()));
            }
        }
    }

    /// The sum goes to a temporary of the output's placement, which the division consumes.
    fn run_sum_then_divide(&mut self, op: &OperationIr, count: usize) {
        let OperationIr::NumericFloat(dtype, mean) = op else {
            unreachable!("Only a float mean divides")
        };
        let out = op.outputs().next().expect("A mean has an output").clone();
        let sum_out = TensorIr {
            id: self.new_tensor_id(),
            ..out.clone()
        };
        let sum = match mean {
            NumericOperationIr::Mean(desc) => NumericOperationIr::Sum(ReduceOpIr {
                out: sum_out.clone(),
                ..desc.clone()
            }),
            NumericOperationIr::MeanDim(desc) => NumericOperationIr::SumDim(ReduceDimOpIr {
                out: sum_out.clone(),
                ..desc.clone()
            }),
            _ => unreachable!("Only a mean divides"),
        };
        let divide = NumericOperationIr::DivScalar(ScalarOpIr {
            lhs: TensorIr {
                status: TensorStatus::ReadWrite,
                ..sum_out.clone()
            },
            rhs: ScalarIr::new(count as f64, dtype),
            out,
        });
        self.placements.insert(sum_out.id, GroupPlacement::Partial);
        for member in 0..self.members.len() {
            self.run_on(member, &OperationIr::NumericFloat(*dtype, sum.clone()));
            self.run_on(member, &OperationIr::NumericFloat(*dtype, divide.clone()));
        }
        self.placements.remove(&sum_out.id);
    }

    fn run_vocab_lookup(&mut self, op: &OperationIr, chunks: Chunks) {
        let OperationIr::Module(ModuleOperationIr::Embedding(desc)) = op else {
            unreachable!("Only an embedding looks up the vocab")
        };
        for member in 0..self.members.len() {
            let chunk = VocabChunk {
                range: chunks.range(member),
                bool_dtype: self.members[member].device_settings().bool_dtype,
            };
            let weights = self.local_float(member, &desc.weights, EmbeddingRule::VOCAB_ROWS);
            let indices = self.local_int(member, &desc.indices);
            let looked_up = chunk.lookup::<B>(weights, indices, desc.out.shape.clone());
            self.members[member]
                .register_tensor_to_device(desc.out.id, HandleKind::Float(looked_up));
        }
    }

    fn run_vocab_backward(&mut self, op: &OperationIr, chunks: Chunks) {
        let OperationIr::Module(ModuleOperationIr::EmbeddingBackward(desc)) = op else {
            unreachable!("Only an embedding's backward scatters into the vocab")
        };
        for member in 0..self.members.len() {
            let chunk = VocabChunk {
                range: chunks.range(member),
                bool_dtype: self.members[member].device_settings().bool_dtype,
            };
            let weights = self.local_float(member, &desc.weights, EmbeddingRule::VOCAB_ROWS);
            let output_grad = self.local_float(member, &desc.out_grad, GroupPlacement::Replicated);
            let indices = self.local_int(member, &desc.indices);
            let grad = chunk.backward::<B>(weights, output_grad, indices);
            self.members[member].register_tensor_to_device(desc.out.id, HandleKind::Float(grad));
        }
    }

    fn local_float(
        &mut self,
        member: usize,
        tensor: &TensorIr,
        placement: GroupPlacement,
    ) -> FloatTensor<B> {
        let members = self.members.len();
        let local = TensorIr {
            shape: placement.local_shape(&tensor.shape, member, members),
            ..tensor.clone()
        };
        match self.members[member].get_tensor(&local) {
            HandleKind::Float(tensor) => tensor,
            _ => unreachable!("{} is a float tensor", tensor.id),
        }
    }

    fn local_int(&mut self, member: usize, tensor: &TensorIr) -> IntTensor<B> {
        match self.members[member].get_tensor(tensor) {
            HandleKind::Int(tensor) => tensor,
            _ => unreachable!("{} is an int tensor", tensor.id),
        }
    }

    fn take_shards(&mut self, tensor: &TensorIr, placement: GroupPlacement) -> ShardList<B> {
        let members = self.members.len();
        let shards = self
            .members
            .iter_mut()
            .enumerate()
            .map(|(member, interpreter)| {
                interpreter.get_tensor(&TensorIr {
                    shape: placement.local_shape(&tensor.shape, member, members),
                    ..tensor.clone()
                })
            })
            .collect();
        if tensor.status == TensorStatus::ReadWrite {
            self.placements.remove(&tensor.id);
        }
        ShardList::new(shards)
    }

    fn register_shards(&mut self, id: TensorId, shards: ShardList<B>, placement: GroupPlacement) {
        for (member, handle) in self.members.iter_mut().zip(shards.into_handles()) {
            member.register_tensor_to_device(id, handle);
        }
        self.placements.insert(id, placement);
    }

    fn drop(&mut self, id: TensorId) {
        if self.placements.remove(&id).is_some() {
            for member in &mut self.members {
                member.register_op(OperationIr::Drop(TensorIr {
                    id,
                    shape: Vec::<usize>::new().into(),
                    status: TensorStatus::ReadWrite,
                    dtype: DType::F32,
                }));
            }
        }
    }
}

/// The rows `range` of a vocab-split embedding, which one member holds.
struct VocabChunk {
    range: Range<usize>,
    bool_dtype: BoolDType,
}

/// Indices into one vocab chunk, with every token the chunk does not hold pointed at its first
/// row and flagged foreign.
struct ChunkIndices<B: BackendIr> {
    local: IntTensor<B>,
    foreign: BoolTensor<B>,
}

impl VocabChunk {
    /// Foreign rows are zeroed with a fill, not a multiply by a mask: a multiply turns an
    /// infinite weight into NaN.
    fn lookup<B: BackendIr>(
        &self,
        weights: FloatTensor<B>,
        indices: IntTensor<B>,
        out_shape: Shape,
    ) -> FloatTensor<B> {
        let ChunkIndices { local, foreign } = self.indices::<B>(indices);
        let looked_up = B::embedding(weights, local);
        Self::zero_foreign::<B>(looked_up, foreign, out_shape)
    }

    /// Each member sums the gradient of the tokens in its own chunk, so its weights' gradient is
    /// its chunk of the whole.
    fn backward<B: BackendIr>(
        &self,
        weights: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> FloatTensor<B> {
        let ChunkIndices { local, foreign } = self.indices::<B>(indices);
        let grad_shape = output_grad.shape();
        let own_grad = Self::zero_foreign::<B>(output_grad, foreign, grad_shape);
        B::embedding_backward(weights, own_grad, local)
    }

    fn indices<B: BackendIr>(&self, indices: IntTensor<B>) -> ChunkIndices<B> {
        let (start, end) = (self.range.start as i64, self.range.end as i64);
        let foreign = B::bool_or(
            B::int_lower_elem(indices.clone(), Scalar::Int(start), self.bool_dtype),
            B::int_greater_equal_elem(indices.clone(), Scalar::Int(end), self.bool_dtype),
        );
        let local = B::int_mask_fill(
            B::int_sub_scalar(indices, Scalar::Int(start)),
            foreign.clone(),
            Scalar::Int(0),
        );
        ChunkIndices { local, foreign }
    }

    /// Zeroes the hidden row of every foreign token in a `[batch, seq, hidden]` tensor.
    fn zero_foreign<B: BackendIr>(
        tensor: FloatTensor<B>,
        foreign: BoolTensor<B>,
        shape: Shape,
    ) -> FloatTensor<B> {
        let mut unsqueezed = foreign.shape();
        unsqueezed.push(1);
        let mask = B::bool_expand(B::bool_reshape(foreign, unsqueezed), shape);
        B::float_mask_fill(tensor, mask, Scalar::Float(0.0))
    }
}

/// Points an op's inputs at their redistributed temporaries, which outlive it.
#[derive(Default)]
struct Renames {
    ids: HashMap<TensorId, TensorId>,
}

impl IrVisitorMut for Renames {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        if tensor.status != TensorStatus::NotInit
            && let Some(id) = self.ids.get(&tensor.id)
        {
            tensor.id = *id;
            tensor.status = TensorStatus::ReadOnly;
        }
    }
}

/// Gives every tensor of an op the shape of the shard one member holds.
struct MemberShapes<'a> {
    member: usize,
    members: usize,
    placements: &'a HashMap<TensorId, GroupPlacement>,
}

impl IrVisitorMut for MemberShapes<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        let placement = self
            .placements
            .get(&tensor.id)
            .unwrap_or_else(|| panic!("{} has no placement on its device group", tensor.id));
        tensor.shape = placement.local_shape(&tensor.shape, self.member, self.members);
    }
}
