use std::{collections::HashMap, fmt::Debug};

use burn_backend::{DType, DTypeUsageSet, DeviceOps, DeviceSettings, ExecutionError, TensorData};
use burn_ir::{BackendIr, IrVisitorMut, OperationIr, TensorId, TensorIr, TensorStatus};
use burn_router::TensorInterpreter;
use burn_std::future::DynFut;

use super::{
    plan::{Execution, OpPlan},
    shard::ShardList,
};
use crate::{GroupMember, Resharding, Sharding};

/// A device group's [`GroupInterpreter`] as a trait object, so the [`GroupClient`] of every group
/// has one type whatever backend the members run.
///
/// It interprets the group's IR in burn-router's sense, as [`TensorInterpreter`] does one device's.
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
    fn sharding(&self, id: &TensorId) -> Sharding;

    /// A new tensor holding `data`, replicated: every member holds the whole value.
    fn register_tensor_data(&mut self, data: TensorData) -> TensorId;

    /// Runs `op` on every member's shards, after resharding each input as the op's
    /// rule asks for.
    fn register_op(&mut self, op: OperationIr);

    /// The same value with another sharding, under a new id.
    ///
    /// # Panics
    ///
    /// When no resharding produces `sharding`: a partial sum out of a tensor that is not one, or a
    /// split along a dim the tensor lacks or one shorter than the group.
    fn reshard(&mut self, tensor: TensorIr, sharding: Sharding) -> TensorId;

    /// The whole value of `tensor`, its shards gathered onto the first member and read there.
    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>>;

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
pub struct GroupInterpreter<B: BackendIr> {
    interpreters: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    shardings: HashMap<TensorId, Sharding>,
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
            shardings: HashMap::new(),
            next_id: 0,
            flush_error: None,
        }
    }

    /// Each default dtype is the first member's own that every member supports, so a tensor
    /// made on the group runs on all of them.
    ///
    /// # Panics
    ///
    /// When no member's default for one of the dtypes runs on every member.
    pub fn default_settings(&self) -> DeviceSettings {
        let defaults: Vec<DeviceSettings> = self.devices.iter().map(DeviceOps::defaults).collect();
        DeviceSettings::new(
            self.first_supported_everywhere(defaults.iter().map(|it| it.float_dtype).collect()),
            self.first_supported_everywhere(defaults.iter().map(|it| it.int_dtype).collect()),
            self.first_supported_everywhere(defaults.iter().map(|it| it.bool_dtype).collect()),
            // Every device of one backend shares its quantization config.
            defaults[0].quantization,
        )
    }

    fn first_supported_everywhere<T: Copy + Debug + Into<DType>>(&self, candidates: Vec<T>) -> T {
        candidates
            .iter()
            .copied()
            .find(|dtype| {
                let dtype = (*dtype).into();
                self.devices
                    .iter()
                    .all(|device| B::supports_dtype(device, dtype))
            })
            .unwrap_or_else(|| {
                panic!("None of the members' defaults {candidates:?} runs on every member")
            })
    }
}

impl<B: BackendIr> DynGroupInterpreter for GroupInterpreter<B> {
    fn new_tensor_id(&mut self) -> TensorId {
        self.next_id += 1;
        TensorId::new(self.next_id)
    }

    fn sharding(&self, id: &TensorId) -> Sharding {
        self.shardings[id]
    }

    fn register_tensor_data(&mut self, data: TensorData) -> TensorId {
        let id = self.new_tensor_id();
        for interpreter in &mut self.interpreters {
            interpreter.register_tensor_data_id(id, data.clone());
        }
        self.shardings.insert(id, Sharding::Replicated);
        id
    }

    fn register_op(&mut self, op: OperationIr) {
        match op {
            // The data was registered on every member with the tensor.
            OperationIr::Init(_) => {}
            OperationIr::Drop(tensor) => self.drop_tensor(tensor.id),
            OperationIr::Distributed(_) => {
                panic!("A device group runs its own collectives and takes none from outside")
            }
            op => self.run(op),
        }
    }

    fn reshard(&mut self, tensor: TensorIr, sharding: Sharding) -> TensorId {
        let current = self.shardings[&tensor.id];
        let num_members = self.interpreters.len();
        let resharding = Resharding::new(current, sharding, &tensor.shape, num_members)
            .unwrap_or_else(|| {
                panic!(
                    "No resharding takes a {current:?} tensor of shape {:?} to {sharding:?} over \
                     {num_members} members",
                    tensor.shape
                )
            });
        let shards =
            self.take_shards(&tensor, current)
                .reshard(resharding, &tensor.shape, &self.devices);
        let id = self.new_tensor_id();
        self.register_shards(id, shards, sharding);
        id
    }

    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let sharding = self.shardings[&tensor.id];
        self.take_shards(&tensor, sharding)
            .into_data(sharding, &self.devices[0])
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
        let plan = OpPlan::new(&op, &self.shardings);
        let temporaries = self.reshard_inputs(&mut op, &plan);
        for (id, sharding) in plan.output_shardings() {
            self.shardings.insert(*id, *sharding);
        }

        match plan.execution() {
            Execution::EveryMember => {
                (0..self.interpreters.len()).for_each(|member| self.run_on(member, &op))
            }
            Execution::FirstMemberCopied => self.run_copied_from_first_member(&op),
        }

        for input in op.inputs() {
            if input.status == TensorStatus::ReadWrite {
                self.shardings.remove(&input.id);
            }
        }
        temporaries.into_iter().for_each(|id| self.drop_tensor(id));
    }

    fn reshard_inputs(&mut self, op: &mut OperationIr, plan: &OpPlan) -> Vec<TensorId> {
        let inputs: Vec<TensorIr> = op.inputs().cloned().collect();
        let mut resharded = ReshardedInputs::default();
        for input in &inputs {
            if resharded.temporaries.contains_key(&input.id) {
                continue;
            }
            let current = self.shardings[&input.id];
            let target = plan.input_sharding(&input.id);
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
            let resharding =
                Resharding::new(current, target, &input.shape, self.interpreters.len())
                    .unwrap_or_else(|| panic!("A rule asked for {target:?} from {current:?}"));
            let shards =
                self.take_shards(&read, current)
                    .reshard(resharding, &input.shape, &self.devices);
            let temporary = self.new_tensor_id();
            self.register_shards(temporary, shards, target);
            resharded.temporaries.insert(input.id, temporary);
        }
        op.visit_mut(&mut resharded);
        resharded.temporaries.into_values().collect()
    }

    fn run_on(&mut self, member: usize, op: &OperationIr) {
        let mut op = op.clone();
        op.visit_mut(&mut MemberShapes {
            member: GroupMember {
                index: member,
                num_members: self.interpreters.len(),
            },
            shardings: &self.shardings,
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

    fn take_shards(&mut self, tensor: &TensorIr, sharding: Sharding) -> ShardList<B> {
        let num_members = self.interpreters.len();
        let shards = self
            .interpreters
            .iter_mut()
            .enumerate()
            .map(|(index, interpreter)| {
                let member = GroupMember { index, num_members };
                interpreter.get_tensor(&member.local_tensor(tensor, sharding))
            })
            .collect();
        if tensor.status == TensorStatus::ReadWrite {
            self.shardings.remove(&tensor.id);
        }
        ShardList::new(shards)
    }

    fn register_shards(&mut self, id: TensorId, shards: ShardList<B>, sharding: Sharding) {
        for (interpreter, handle) in self.interpreters.iter_mut().zip(shards.into_handles()) {
            interpreter.register_tensor_to_device(id, handle);
        }
        self.shardings.insert(id, sharding);
    }

    fn drop_tensor(&mut self, id: TensorId) {
        if self.shardings.remove(&id).is_some() {
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

/// Points an op's inputs at their resharded temporaries, which outlive it.
#[derive(Default)]
struct ReshardedInputs {
    temporaries: HashMap<TensorId, TensorId>,
}

impl IrVisitorMut for ReshardedInputs {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        if tensor.status != TensorStatus::NotInit
            && let Some(id) = self.temporaries.get(&tensor.id)
        {
            tensor.id = *id;
            tensor.status = TensorStatus::ReadOnly;
        }
    }
}

/// Gives every tensor of an op the shape of the shard one member holds.
struct MemberShapes<'a> {
    member: GroupMember,
    shardings: &'a HashMap<TensorId, Sharding>,
}

impl IrVisitorMut for MemberShapes<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        *tensor = self.member.local_tensor(tensor, self.shardings[&tensor.id]);
    }
}
