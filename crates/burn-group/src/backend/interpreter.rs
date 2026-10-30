use std::{
    collections::{BTreeMap, HashMap},
    iter,
};

use burn_backend::{
    DType, DTypeUsageSet, DeviceSettings, ExecutionError, TensorData, ops::DimSplit,
};
use burn_ir::{BackendIr, IrVisitorMut, OperationIr, TensorId, TensorIr, TensorStatus};
use burn_router::TensorInterpreter;
use burn_std::{QuantConfig, future::DynFut};

use super::{GroupError, MemberDevice};
use super::{
    plan::{Execution, OpPlan},
    tensor::ShardedTensor,
};
use crate::{GroupMember, Resharding, Sharding, ShardingError};

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

    /// A new tensor holding `data`, replicated. It stays on the host until it is used, so a split
    /// by `with_sharding`, or by an op that consumes it, uploads each member only its chunk.
    fn register_tensor_data(&mut self, data: TensorData) -> TensorId;

    /// Runs `op` on every member's shards, after resharding each input as the op's
    /// rule asks for.
    ///
    /// An op its plan defers, such as a random one, waits until its output is needed, and runs at
    /// the sharding that need asks for.
    fn register_op(&mut self, op: OperationIr);

    /// The same value with another sharding, under a new id, or why no resharding produces it.
    fn reshard(&mut self, tensor: TensorIr, sharding: Sharding) -> Result<TensorId, ShardingError>;

    /// The whole value of `tensor`: the data itself while it is on the host, otherwise its shards
    /// gathered onto the first member and read there.
    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>>;

    /// Waits for every member's work to finish.
    fn sync(&mut self) -> Result<(), ExecutionError>;

    /// Submits every member's queued work without waiting, and returns the first member's error.
    fn flush(&mut self) -> Result<(), ExecutionError>;

    /// Seeds every member's backend, once every pending op has run on the seed before.
    fn seed(&mut self, seed: u64);

    /// What every member can do with `dtype`, since each op runs on all of them.
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet;
}

/// The interpreter of a group whose members run `B`: one [`TensorInterpreter`] per member, and
/// where each tensor's shards sit.
pub struct GroupInterpreter<B: BackendIr> {
    interpreters: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    shardings: HashMap<TensorId, Sharding>,
    on_host: HashMap<TensorId, TensorData>,
    // Ordered by id, the order the ops were issued, so running several at once is reproducible.
    pending_ops: BTreeMap<TensorId, OperationIr>,
    next_id: u64,
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
            on_host: HashMap::new(),
            pending_ops: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Each default dtype is the first of the `members`' own that every member supports, so a
    /// tensor made on the group runs on all of them.
    pub fn default_settings(
        &self,
        members: &[MemberDevice<B::Device>],
    ) -> Result<DeviceSettings, GroupError> {
        let settings = || members.iter().map(|it| it.settings);
        Ok(DeviceSettings::new(
            self.first_supported_everywhere(settings().map(|it| it.float_dtype))?,
            self.first_supported_everywhere(settings().map(|it| it.int_dtype))?,
            self.first_supported_everywhere(settings().map(|it| it.bool_dtype))?,
            Self::shared_quantization(settings().map(|it| it.quantization))?,
        ))
    }

    fn shared_quantization(
        configs: impl Iterator<Item = QuantConfig> + Clone,
    ) -> Result<QuantConfig, GroupError> {
        let mut rest = configs.clone();
        let first = rest.next().expect("A group has at least one member");
        match rest.all(|config| config == first) {
            true => Ok(first),
            false => Err(GroupError::MixedQuantization {
                configs: configs.collect(),
            }),
        }
    }

    fn first_supported_everywhere<T: Copy + Into<DType>>(
        &self,
        candidates: impl Iterator<Item = T> + Clone,
    ) -> Result<T, GroupError> {
        candidates
            .clone()
            .find(|&dtype| {
                let dtype = dtype.into();
                self.devices
                    .iter()
                    .all(|device| B::supports_dtype(device, dtype))
            })
            .ok_or_else(|| GroupError::NoSharedDefault {
                candidates: candidates.map(Into::into).collect(),
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
        self.on_host.insert(id, data);
        self.shardings.insert(id, Sharding::Replicated);
        id
    }

    fn register_op(&mut self, op: OperationIr) {
        match op {
            // The data was registered with the tensor.
            OperationIr::Init(_) => {}
            OperationIr::Drop(tensor) => self.drop_tensor(tensor.id),
            op => self.run(op),
        }
    }

    fn reshard(&mut self, tensor: TensorIr, sharding: Sharding) -> Result<TensorId, ShardingError> {
        self.run_pending(tensor.id, sharding);
        let current = self.shardings[&tensor.id];
        self.resharded(&tensor, current, sharding).inspect_err(|_| {
            // The caller gave the tensor up, so a refusal must not leave it on the members.
            if tensor.status == TensorStatus::ReadWrite {
                self.drop_tensor(tensor.id);
            }
        })
    }

    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        self.run_pending(tensor.id, Sharding::Replicated);
        if let Some(data) = self.take_host_data(&tensor) {
            return Box::pin(async move { Ok(data) });
        }
        let sharding = self.shardings[&tensor.id];
        self.take_shards(&tensor, sharding)
            .into_data(sharding, &self.devices[0])
    }

    fn sync(&mut self) -> Result<(), ExecutionError> {
        self.interpreters
            .iter()
            .try_for_each(TensorInterpreter::sync)
    }

    fn flush(&mut self) -> Result<(), ExecutionError> {
        // Every member is flushed, whichever fails first.
        let mut first_error = None;
        for device in &self.devices {
            if let Err(error) = B::flush(device) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn seed(&mut self, seed: u64) {
        while let Some(&id) = self.pending_ops.keys().next() {
            self.run_pending(id, Sharding::Replicated);
        }
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
        let plan = OpPlan::new(&op, &self.shardings, self.interpreters.len());
        let temporaries = self.shard_inputs(&mut op, &plan);
        for (id, sharding) in plan.output_shardings() {
            self.shardings.insert(*id, *sharding);
        }

        match plan.execution() {
            Execution::Eager => self.run_on_every_member(&op),
            Execution::Deferred => self.keep_pending(&op),
        }

        for input in op.inputs() {
            if input.status == TensorStatus::ReadWrite {
                self.shardings.remove(&input.id);
            }
        }
        temporaries.into_iter().for_each(|id| self.drop_tensor(id));
    }

    /// Gets every input of `op` onto the members as the plan shards it: uploaded whole, cut on the
    /// host, or resharded. Returns the temporaries to drop after the op.
    fn shard_inputs(&mut self, op: &mut OperationIr, plan: &OpPlan) -> Vec<TensorId> {
        let inputs: Vec<TensorIr> = op.inputs().cloned().collect();
        let mut resharded = ReshardedInputs::default();
        for input in &inputs {
            if resharded.temporaries.contains_key(&input.id) {
                continue;
            }
            let target = plan.input_sharding(&input.id);
            self.run_pending(input.id, target);
            let consumed = inputs
                .iter()
                .any(|other| other.id == input.id && other.status == TensorStatus::ReadWrite);
            // Only an input the op consumes is cut on the host. One it only reads, such as a
            // parameter, reaches the members whole once, and each later op slices it there.
            if !(consumed && matches!(target, Sharding::Split { .. })) {
                self.upload(input.id);
            }
            let current = self.shardings[&input.id];
            if current == target {
                continue;
            }
            let read = TensorIr {
                status: match consumed {
                    true => TensorStatus::ReadWrite,
                    false => TensorStatus::ReadOnly,
                },
                ..input.clone()
            };
            let temporary = self
                .resharded(&read, current, target)
                .unwrap_or_else(|error| {
                    panic!("A rule asked for {target:?} from {current:?}: {error}")
                });
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
                ShardedTensor::replicated(self.interpreters[0].get_tensor(&read), &self.devices);
            for (interpreter, copy) in self.interpreters.iter_mut().zip(copies.into_handles()) {
                interpreter.register_tensor_to_device(output.id, copy);
            }
        }
    }

    fn run_on_every_member(&mut self, op: &OperationIr) {
        (0..self.interpreters.len()).for_each(|member| self.run_on(member, op));
    }

    fn keep_pending(&mut self, op: &OperationIr) {
        let out = op.outputs().next().expect("A deferred op has one output");
        self.pending_ops.insert(out.id, op.clone());
    }

    /// Runs the pending op of tensor `id`, if it has one: on every member at its shard when
    /// `sharding` splits it, otherwise on member 0 and copied.
    fn run_pending(&mut self, id: TensorId, sharding: Sharding) {
        let Some(op) = self.pending_ops.remove(&id) else {
            return;
        };
        let out = op.outputs().next().expect("A deferred op has one output");
        // A split the tensor cannot take is left to the resharding after, which reports it.
        let split = matches!(sharding, Sharding::Split { .. })
            && Resharding::new(
                Sharding::Replicated,
                sharding,
                &out.shape,
                self.interpreters.len(),
            )
            .is_ok();
        match split {
            true => {
                self.shardings.insert(id, sharding);
                self.run_on_every_member(&op);
            }
            false => self.run_copied_from_first_member(&op),
        }
    }

    fn take_shards(&mut self, tensor: &TensorIr, sharding: Sharding) -> ShardedTensor<B> {
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
        ShardedTensor::new(shards)
    }

    /// The value of `tensor` resharded from `current` to `target`, under a new id.
    fn resharded(
        &mut self,
        tensor: &TensorIr,
        current: Sharding,
        target: Sharding,
    ) -> Result<TensorId, ShardingError> {
        let resharding = Resharding::new(current, target, &tensor.shape, self.interpreters.len())?;
        let id = self.new_tensor_id();
        match resharding {
            Resharding::Slice { dim } if self.on_host.contains_key(&tensor.id) => {
                self.upload_chunks(tensor, id, dim)?
            }
            resharding => self.reshard_on_members(tensor, current, resharding, id)?,
        }
        self.shardings.insert(id, target);
        Ok(id)
    }

    /// Reshards the shards of `tensor` on the members under `id`, uploading it whole first if it
    /// is still on the host.
    fn reshard_on_members(
        &mut self,
        tensor: &TensorIr,
        current: Sharding,
        resharding: Resharding,
        id: TensorId,
    ) -> Result<(), ShardingError> {
        self.upload(tensor.id);
        let shards = self
            .take_shards(tensor, current)
            .reshard(resharding, &tensor.shape)?;
        self.register_shards(id, shards);
        Ok(())
    }

    /// Cuts a tensor still on the host along `dim` and uploads each member only its chunk, under
    /// `id`. The data leaves the host when the tensor is given up.
    fn upload_chunks(
        &mut self,
        tensor: &TensorIr,
        id: TensorId,
        dim: usize,
    ) -> Result<(), ShardingError> {
        if matches!(tensor.dtype, DType::QFloat(_)) {
            return Err(ShardingError::Quantized);
        }
        let split = DimSplit {
            dim,
            len: tensor.shape[dim],
            members: self.interpreters.len(),
        };
        let chunks = split.data_chunks(&self.on_host[&tensor.id]);
        for (interpreter, chunk) in self.interpreters.iter_mut().zip(chunks) {
            interpreter.register_tensor_data_id(id, chunk);
        }
        if tensor.status == TensorStatus::ReadWrite {
            self.drop_tensor(tensor.id);
        }
        Ok(())
    }

    /// The data of a tensor still on the host, which leaves it when the tensor is given up.
    fn take_host_data(&mut self, tensor: &TensorIr) -> Option<TensorData> {
        match tensor.status {
            TensorStatus::ReadWrite => {
                let data = self.on_host.remove(&tensor.id)?;
                self.shardings.remove(&tensor.id);
                Some(data)
            }
            _ => self.on_host.get(&tensor.id).cloned(),
        }
    }

    /// Uploads a tensor still on the host to every member whole.
    fn upload(&mut self, id: TensorId) {
        if let Some(data) = self.on_host.remove(&id) {
            let copies = iter::repeat_n(data, self.interpreters.len());
            for (interpreter, data) in self.interpreters.iter_mut().zip(copies) {
                interpreter.register_tensor_data_id(id, data);
            }
        }
    }

    fn register_shards(&mut self, id: TensorId, shards: ShardedTensor<B>) {
        for (interpreter, handle) in self.interpreters.iter_mut().zip(shards.into_handles()) {
            interpreter.register_tensor_to_device(id, handle);
        }
    }

    fn drop_tensor(&mut self, id: TensorId) {
        if self.on_host.remove(&id).is_some() || self.pending_ops.remove(&id).is_some() {
            self.shardings.remove(&id);
            return;
        }
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

#[cfg(test)]
mod tests {
    use burn_flex::{Flex, FlexDevice};
    use burn_ir::{FloatOperationIr, MatmulOpIr};
    use burn_std::{Bytes, FileReadCounter, QuantScheme, QuantStore, QuantValue, future::block_on};

    use super::*;

    fn whole(id: TensorId, status: TensorStatus) -> TensorIr {
        TensorIr {
            id,
            shape: [4, 6].into(),
            status,
            dtype: DType::F32,
        }
    }

    #[test]
    fn a_split_of_data_uploads_no_member_the_whole_tensor() {
        let mut group = GroupInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice]);
        let id = group.register_tensor_data(TensorData::new(vec![0f32; 24], [4, 6]));

        let split = group
            .reshard(
                whole(id, TensorStatus::ReadOnly),
                Sharding::Split { dim: 1 },
            )
            .unwrap();

        for interpreter in &group.interpreters {
            assert!(interpreter.read_failure(&id).is_some());
            assert!(interpreter.read_failure(&split).is_none());
        }
    }

    #[test]
    fn data_an_op_consumes_split_is_cut_on_the_host() {
        let mut group = GroupInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice]);
        let x = group.register_tensor_data(TensorData::new(vec![1f32; 24], [4, 6]));
        let x = group
            .reshard(
                whole(x, TensorStatus::ReadWrite),
                Sharding::Split { dim: 1 },
            )
            .unwrap();
        let path = std::env::temp_dir().join(format!("burn-group-cut-{}", std::process::id()));
        std::fs::write(&path, TensorData::new(vec![1f32; 24], [6, 4]).as_bytes()).unwrap();
        let reads = FileReadCounter::new(&path);
        let w = group.register_tensor_data(TensorData::from_bytes(
            Bytes::from_file(&path, 96, 0),
            [6, 4],
            DType::F32,
        ));
        let out = group.new_tensor_id();

        // A lhs split along its columns asks for the rhs split along its rows.
        group.register_op(OperationIr::Float(
            DType::F32,
            FloatOperationIr::Matmul(MatmulOpIr {
                lhs: whole(x, TensorStatus::ReadWrite),
                rhs: TensorIr {
                    shape: [6, 4].into(),
                    ..whole(w, TensorStatus::ReadWrite)
                },
                out: TensorIr {
                    shape: [4, 4].into(),
                    ..whole(out, TensorStatus::NotInit)
                },
            }),
        ));

        // Each member reads only its rows of the file, where a whole upload reads it all on each.
        assert_eq!(reads.read(), 96);
        let read = block_on(group.read(TensorIr {
            shape: [4, 4].into(),
            ..whole(out, TensorStatus::ReadWrite)
        }))
        .unwrap();
        assert_eq!(read, TensorData::new(vec![6f32; 16], [4, 4]));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_split_of_quantized_data_is_refused_before_any_upload() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let data = TensorData::quantized(vec![1i8; 24], [4, 6], scheme, &[0.1], None);
        let dtype = data.dtype();
        let mut group = GroupInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice]);
        let id = group.register_tensor_data(data);

        let refused = group.reshard(
            TensorIr {
                dtype,
                ..whole(id, TensorStatus::ReadOnly)
            },
            Sharding::Split { dim: 0 },
        );

        assert_eq!(refused, Err(ShardingError::Quantized));
        for interpreter in &group.interpreters {
            assert!(interpreter.read_failure(&id).is_some());
        }
    }
}
