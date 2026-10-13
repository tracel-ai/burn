use std::collections::HashMap;

use burn_backend::{DType, DTypeUsageSet, ExecutionError, TensorData};
use burn_ir::{
    BackendIr, GraphBindings, GraphId, IrVisitorMut, LinearOpIr, ModuleOperationIr, OperationIr,
    TensorId, TensorIr, TensorStatus,
};
use burn_router::{Graph, TensorInterpreter};
use burn_std::future::DynFut;

use super::{
    placements::TensorPlacements,
    plan::{Execution, OpPlan},
    shard::ShardList,
};
use crate::{GroupPlacement, PlacementShapes, Redistribution};

/// A device group's [`GroupInterpreter`] as a trait object, so the [`GroupClient`] of every group
/// has one type whatever backend the members run.
///
/// An interpreter is burn-router's name for what executes the IR a
/// [`RouterClient`](burn_router::RouterClient) sends, as [`TensorInterpreter`] does for one
/// device. The group's client forwards each of its calls here, and the interpreter carries it out
/// on every member at once.
///
/// [`GroupClient`]: super::GroupClient
pub trait DynGroupInterpreter: Send {
    /// A fresh id for a tensor the client is about to create.
    fn new_tensor_id(&mut self) -> TensorId;

    /// Where the shards of the tensor `id` sit.
    ///
    /// # Panics
    ///
    /// When the tensor is not on the group.
    fn placement(&self, id: &TensorId) -> GroupPlacement;

    /// A new tensor holding `data`, replicated: every member holds the whole value.
    fn register_tensor_data(&mut self, data: TensorData) -> TensorId;

    /// Runs `op` on every member's shards, after moving each input to the placement the op's
    /// rule asks for.
    fn register_op(&mut self, op: OperationIr);

    /// The same value at another placement, under a new id.
    ///
    /// # Panics
    ///
    /// When no collective produces `placement`: a partial sum out of a tensor that is not one, or a
    /// split along a dim the tensor lacks or one shorter than the group.
    fn place(&mut self, tensor: TensorIr, placement: GroupPlacement) -> TensorId;

    /// The whole value of `tensor`, its shards gathered onto the first member and read there.
    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>>;

    /// Makes `new_id` a second handle on the shards of `src_id`, at the same placement.
    fn register_alias(&mut self, new_id: TensorId, src_id: TensorId);

    /// Keeps `relative_graph` under `graph_id` for replay, and runs it once with `bindings`.
    fn register_graph(
        &mut self,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    );

    /// Replays the graph kept under `graph_id` with `bindings`, each op planned at the placements
    /// its bound tensors have now.
    fn execute_graph(&mut self, graph_id: GraphId, bindings: GraphBindings);

    /// Waits for every member's work to finish, returning first any error a flush kept.
    fn sync(&mut self) -> Result<(), ExecutionError>;

    /// Submits every member's queued work without waiting, keeping the first error for
    /// [`sync`](Self::sync).
    fn flush(&mut self);

    /// Seeds every member's backend.
    fn seed(&self, seed: u64);

    /// What every member can do with `dtype`, since each op runs on all of them.
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet;
}

/// The interpreter of a group whose members run `B`: one [`TensorInterpreter`] per member, and
/// where each tensor's shards sit.
///
/// Each op is planned from its inputs' placements, its inputs are moved where the plan asks, and
/// every member's interpreter then runs the op on its own shards.
pub struct GroupInterpreter<B: BackendIr> {
    interpreters: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    placements: TensorPlacements,
    graphs: HashMap<GraphId, Graph>,
    next_id: u64,
    flush_error: Option<ExecutionError>,
}

impl<B: BackendIr> GroupInterpreter<B> {
    pub fn new(devices: Vec<B::Device>) -> Self {
        Self {
            interpreters: devices
                .iter()
                .map(|device| TensorInterpreter::new(device.clone()))
                .collect(),
            devices,
            placements: TensorPlacements::default(),
            graphs: HashMap::new(),
            next_id: 0,
            flush_error: None,
        }
    }
}

impl<B: BackendIr> DynGroupInterpreter for GroupInterpreter<B> {
    fn new_tensor_id(&mut self) -> TensorId {
        self.next_id += 1;
        TensorId::new(self.next_id)
    }

    fn placement(&self, id: &TensorId) -> GroupPlacement {
        self.placements.get(id)
    }

    fn register_tensor_data(&mut self, data: TensorData) -> TensorId {
        let id = self.new_tensor_id();
        for interpreter in &mut self.interpreters {
            interpreter.register_tensor_data_id(id, data.clone());
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
        let current = self.placements.get(&tensor.id);
        let members = self.interpreters.len();
        let redistribution = Redistribution::new(current, placement, &tensor.shape, members)
            .unwrap_or_else(|| {
                panic!(
                    "No collective places a {current:?} tensor of shape {:?} at {placement:?} over \
                     {members} members",
                    tensor.shape
                )
            });
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
        let placement = self.placements.get(&tensor.id);
        self.take_shards(&tensor, placement)
            .into_data(placement, &self.devices[0])
    }

    fn register_alias(&mut self, new_id: TensorId, src_id: TensorId) {
        for interpreter in &mut self.interpreters {
            interpreter.register_alias(new_id, src_id);
        }
        self.placements.insert(new_id, self.placements.get(&src_id));
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
        self.interpreters
            .iter()
            .try_for_each(TensorInterpreter::sync)
    }

    fn flush(&mut self) {
        for device in &self.devices {
            if let Err(error) = B::flush(device) {
                self.flush_error.get_or_insert(error);
            }
        }
    }

    fn seed(&self, seed: u64) {
        self.interpreters
            .iter()
            .for_each(|interpreter| interpreter.seed(seed));
    }

    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        self.interpreters
            .iter()
            .fold(DTypeUsageSet::all(), |usage, interpreter| {
                usage & interpreter.dtype_usage(dtype)
            })
    }
}

impl<B: BackendIr> GroupInterpreter<B> {
    fn run(&mut self, mut op: OperationIr) {
        let plan = OpPlan::new(&op, &self.placements);
        let temporaries = self.redistribute_inputs(&mut op, &plan);
        for (id, placement) in plan.outputs() {
            self.placements.insert(*id, *placement);
        }

        match plan.execution() {
            Execution::EveryMember => {
                (0..self.interpreters.len()).for_each(|member| self.run_on(member, &op))
            }
            Execution::FirstMemberCopied => self.run_copied_from_first_member(&op),
            Execution::BiasOnFirstMember => self.run_with_bias_on_first_member(&op),
        }

        for input in op.inputs() {
            if input.status == TensorStatus::ReadWrite {
                self.placements.remove(&input.id);
            }
        }
        temporaries.into_iter().for_each(|id| self.drop(id));
    }

    fn redistribute_inputs(&mut self, op: &mut OperationIr, plan: &OpPlan) -> Vec<TensorId> {
        let inputs: Vec<TensorIr> = op.inputs().cloned().collect();
        let mut renames = Renames::default();
        for input in &inputs {
            if renames.ids.contains_key(&input.id) {
                continue;
            }
            let (current, target) = (self.placements.get(&input.id), plan.input(&input.id));
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
                Redistribution::new(current, target, &input.shape, self.interpreters.len())
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
            members: self.interpreters.len(),
            placements: &self.placements,
        });
        self.interpreters[member].register_op(op);
    }

    fn run_copied_from_first_member(&mut self, op: &OperationIr) {
        self.run_on(0, op);
        for output in op.outputs() {
            let read = TensorIr {
                status: TensorStatus::ReadOnly,
                ..output.clone()
            };
            let copies =
                ShardList::replicated(self.interpreters[0].get_tensor(&read), &self.devices);
            for (interpreter, copy) in self.interpreters.iter_mut().zip(copies.into_handles()) {
                interpreter.register_tensor_to_device(output.id, copy);
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
        for member in 1..self.interpreters.len() {
            self.run_on(member, &without_bias);
            if bias.status == TensorStatus::ReadWrite {
                self.interpreters[member].register_op(OperationIr::Drop(bias.clone()));
            }
        }
    }

    fn take_shards(&mut self, tensor: &TensorIr, placement: GroupPlacement) -> ShardList<B> {
        let members = self.interpreters.len();
        let shards = self
            .interpreters
            .iter_mut()
            .enumerate()
            .map(|(member, interpreter)| {
                interpreter.get_tensor(&placement.local_tensor(tensor, member, members))
            })
            .collect();
        if tensor.status == TensorStatus::ReadWrite {
            self.placements.remove(&tensor.id);
        }
        ShardList::new(shards)
    }

    fn register_shards(&mut self, id: TensorId, shards: ShardList<B>, placement: GroupPlacement) {
        for (interpreter, handle) in self.interpreters.iter_mut().zip(shards.into_handles()) {
            interpreter.register_tensor_to_device(id, handle);
        }
        self.placements.insert(id, placement);
    }

    fn drop(&mut self, id: TensorId) {
        if self.placements.remove(&id).is_some() {
            for interpreter in &mut self.interpreters {
                interpreter.register_op(OperationIr::Drop(TensorIr {
                    id,
                    shape: Vec::<usize>::new().into(),
                    status: TensorStatus::ReadWrite,
                    dtype: DType::F32,
                }));
            }
        }
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
    placements: &'a TensorPlacements,
}

impl IrVisitorMut for MemberShapes<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        *tensor = self
            .placements
            .get(&tensor.id)
            .local_tensor(tensor, self.member, self.members);
    }
}
