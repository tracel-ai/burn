use burn_backend::RandomKey;
use std::{
    collections::{BTreeMap, HashMap},
    iter,
};

use burn_backend::{
    DType, DTypeUsageSet, DeviceSettings, ExecutionError, TensorData, ops::DimSplit,
};
use burn_ir::{
    BackendIr, BaseOperationIr, IrVisitorMut, OperationIr, TensorError, TensorId, TensorIr,
    TensorStatus,
};
use burn_router::TensorInterpreter;
use burn_std::{QuantConfig, Slice, future::DynFut};

use super::{
    CollectiveMember, CollectiveOpRegistry, MemberDevice, ShardCustomOps, ShardError,
    custom_ops::{CustomOpRuleMap, MemberCustomOps},
};
use super::{
    plan::{Execution, OpPlan},
    tensor::ShardedTensor,
};
use crate::{Resharding, ShardMember, Sharding, ShardingError};

/// A sharded compute device's [`ShardInterpreter`] as a trait object, so the [`ShardClient`] of every sharded compute device
/// has one type whatever backend the members run.
///
/// It interprets the sharded compute device's IR in burn-router's sense, as [`TensorInterpreter`] does one device's.
///
/// [`ShardClient`]: super::ShardClient
pub trait DynShardInterpreter: Send {
    /// A fresh id for a tensor the client is about to create.
    fn new_tensor_id(&mut self) -> TensorId;

    /// Where the shards of the tensor `id` sit.
    ///
    /// # Panics
    ///
    /// When the tensor is not on the sharded compute device.
    fn sharding(&self, id: &TensorId) -> Sharding;

    /// How many members the tensors are split over.
    fn num_members(&self) -> usize;

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

    /// Seeds future draws on every member's backend without executing pending operations.
    fn seed(&mut self, seed: u64);

    /// Reserves a key while the random operation is being registered.
    fn next_random_key(&self) -> RandomKey;

    /// What every member can do with `dtype`, since each op runs on all of them.
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet;
}

/// The interpreter of a sharded compute device whose members run `B`: one [`TensorInterpreter`] per member, and
/// where each tensor's shards sit.
pub struct ShardInterpreter<B: BackendIr> {
    interpreters: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    shardings: HashMap<TensorId, Sharding>,
    on_host: HashMap<TensorId, TensorData>,
    // Ordered by id, the order the ops were issued, so running several at once is reproducible.
    pending_ops: BTreeMap<TensorId, OperationIr>,
    next_id: u64,
    collectives: CollectiveOpRegistry<B>,
    rules: CustomOpRuleMap,
}

impl<B: BackendIr> ShardInterpreter<B> {
    /// The members of `devices` run the custom operations of `custom_ops`.
    ///
    /// # Errors
    ///
    /// When `custom_ops` are another backend's.
    pub fn new(devices: Vec<B::Device>, custom_ops: ShardCustomOps) -> Result<Self, ShardError> {
        let MemberCustomOps {
            handlers,
            collectives,
            rules,
        } = MemberCustomOps::<B>::try_from(custom_ops)?;
        Ok(Self {
            interpreters: devices
                .iter()
                .map(|device| TensorInterpreter::with_custom_ops(device.clone(), handlers.clone()))
                .collect(),
            devices,
            shardings: HashMap::new(),
            on_host: HashMap::new(),
            pending_ops: BTreeMap::new(),
            next_id: 0,
            collectives,
            rules,
        })
    }

    /// Each default dtype is the first of the `members`' own that every member supports, so a
    /// tensor made on the sharded compute device runs on all of them.
    pub fn default_settings(
        &self,
        members: &[MemberDevice<B::Device>],
    ) -> Result<DeviceSettings, ShardError> {
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
    ) -> Result<QuantConfig, ShardError> {
        let mut rest = configs.clone();
        let first = rest
            .next()
            .expect("A sharded compute device has at least one member");
        match rest.all(|config| config == first) {
            true => Ok(first),
            false => Err(ShardError::MixedQuantization {
                configs: configs.collect(),
            }),
        }
    }

    fn first_supported_everywhere<T: Copy + Into<DType>>(
        &self,
        candidates: impl Iterator<Item = T> + Clone,
    ) -> Result<T, ShardError> {
        candidates
            .clone()
            .find(|&dtype| {
                let dtype = dtype.into();
                self.devices
                    .iter()
                    .all(|device| B::supports_dtype(device, dtype))
            })
            .ok_or_else(|| ShardError::NoSharedDefault {
                candidates: candidates.map(Into::into).collect(),
            })
    }
}

impl<B: BackendIr> DynShardInterpreter for ShardInterpreter<B> {
    fn new_tensor_id(&mut self) -> TensorId {
        self.next_id += 1;
        TensorId::new(self.next_id)
    }

    fn sharding(&self, id: &TensorId) -> Sharding {
        self.shardings[id]
    }

    fn num_members(&self) -> usize {
        self.interpreters.len()
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

    fn next_random_key(&self) -> RandomKey {
        B::next_random_key(&self.devices[0])
    }

    fn seed(&mut self, seed: u64) {
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

impl<B: BackendIr> ShardInterpreter<B> {
    fn run(&mut self, mut op: OperationIr) {
        let plan = OpPlan::new(&op, &self.shardings, &self.rules, self.interpreters.len());
        if let Some(error) = self.failure(&op, &plan) {
            return self.fail(&op, &error);
        }
        let temporaries = self.shard_inputs(&mut op, &plan);
        for (id, sharding) in plan.output_shardings() {
            self.shardings.insert(*id, *sharding);
        }

        match plan.execution() {
            Execution::Eager => self.run_on_every_member(&op),
            Execution::Deferred => self.keep_pending(&op),
            Execution::Collective => self.run_collective(&op),
            Execution::Refused(_) => unreachable!("A refused op fails before it runs"),
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
        let op = self.member_op(member, op);
        self.interpreters[member].register_op(op);
    }

    /// `op` as `member` runs it, on its own shards.
    fn member_op(&self, member: usize, op: &OperationIr) -> OperationIr {
        let mut op = op.clone();
        op.visit_mut(&mut MemberShapes {
            member: ShardMember {
                index: member,
                num_members: self.interpreters.len(),
            },
            shardings: &self.shardings,
        });
        self.localize_slice_ranges(&mut op);
        op
    }

    /// Why `op` fails on every member without running: an input a failure claimed, as a
    /// member's own interpreter fails an op, a rule its inputs cannot follow, or a collective
    /// op with no handler. Checked before any input is resharded, which reads every shard.
    fn failure(&mut self, op: &OperationIr, plan: &OpPlan) -> Option<TensorError> {
        let input_error = self
            .interpreters
            .iter_mut()
            .find_map(|interpreter| interpreter.handles_mut().input_error(op).cloned());
        if let Some(error) = input_error {
            return Some(error.propagated());
        }
        let custom = match op {
            OperationIr::Custom(custom) => custom,
            _ => return None,
        };
        let reason = match plan.execution() {
            Execution::Refused(error) => {
                format!(
                    "`{}` cannot run on the sharded compute device: {error}",
                    custom.id
                )
            }
            Execution::Collective if self.collectives.get(&custom.id).is_none() => format!(
                "No collective handler registered for `{}`. Register one in the \
                 `CollectiveOpRegistry` the sharded compute device is made with.",
                custom.id
            ),
            _ => return None,
        };
        Some(TensorError::new(ExecutionError::with_context(reason)))
    }

    /// Fails the outputs of `op` on every member with `error`, and gives up the inputs it
    /// consumes.
    fn fail(&mut self, op: &OperationIr, error: &TensorError) {
        for out in op.outputs() {
            self.shardings.insert(out.id, Sharding::Replicated);
        }
        for interpreter in &mut self.interpreters {
            interpreter.handles_mut().set_output_errors(op, error);
        }
        for input in op.inputs() {
            if input.status == TensorStatus::ReadWrite {
                self.drop_tensor(input.id);
            }
        }
    }

    /// Runs a custom op once over every member by its collective handler.
    fn run_collective(&mut self, op: &OperationIr) {
        let OperationIr::Custom(custom) = op else {
            unreachable!("Only a custom op is planned collective")
        };
        let ops: Vec<OperationIr> = (0..self.interpreters.len())
            .map(|member| self.member_op(member, op))
            .collect();
        let handler = self
            .collectives
            .get(&custom.id)
            .cloned()
            .expect("A collective op without a handler fails before it runs");
        let mut members: Vec<CollectiveMember<'_, B>> = self
            .interpreters
            .iter_mut()
            .zip(&ops)
            .zip(&self.devices)
            .map(|((interpreter, op), device)| {
                let OperationIr::Custom(op) = op else {
                    unreachable!("A member's op is the op it was cut from")
                };
                CollectiveMember {
                    handles: interpreter.handles_mut(),
                    op,
                    device,
                }
            })
            .collect();
        handler(&mut members);
    }

    /// Gives a slice the ranges of the shard one member holds, since a backend may reject the
    /// whole tensor's range on a shard. Its rule keeps a split only on a dim the slice takes whole,
    /// so on a member that dim's range is the whole shard.
    fn localize_slice_ranges(&self, op: &mut OperationIr) {
        let (tensor, ranges) = match op {
            OperationIr::BaseFloat(BaseOperationIr::Slice(desc))
            | OperationIr::BaseInt(BaseOperationIr::Slice(desc))
            | OperationIr::BaseBool(BaseOperationIr::Slice(desc)) => {
                (&desc.tensor, &mut desc.ranges)
            }
            OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc))
            | OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc))
            | OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc)) => {
                (&desc.tensor, &mut desc.ranges)
            }
            _ => return,
        };
        if let Sharding::Split { dim } = self.shardings[&tensor.id]
            && let Some(range) = ranges.get_mut(dim)
        {
            *range = Slice::full();
        }
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
                self.run_random_on_members(op);
            }
            false => self.run_copied_from_first_member(&op),
        }
    }

    fn run_random_on_members(&mut self, mut op: OperationIr) {
        use burn_ir::{FloatOperationIr, NumericOperationIr, RandomOpIr};

        fn random_desc(op: &mut OperationIr) -> &mut RandomOpIr {
            match op {
                OperationIr::Float(_, FloatOperationIr::Random(desc))
                | OperationIr::NumericInt(_, NumericOperationIr::IntRandom(desc)) => desc,
                _ => unreachable!("Only random operations are deferred"),
            }
        }

        // Derive shard keys from the captured operation key, independently of first use
        // and member scheduling. Replicated draws instead keep the original operation key.
        let keys = random_desc(&mut op).key().split();
        for (member, key) in (0..self.interpreters.len()).zip(keys) {
            random_desc(&mut op).key = key.into();
            self.run_on(member, &op);
        }
    }

    fn take_shards(&mut self, tensor: &TensorIr, sharding: Sharding) -> ShardedTensor<B> {
        let num_members = self.interpreters.len();
        let shards = self
            .interpreters
            .iter_mut()
            .enumerate()
            .map(|(index, interpreter)| {
                let member = ShardMember { index, num_members };
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
    member: ShardMember,
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
    use burn_ir::{CustomOpIr, FloatOperationIr, MatmulOpIr};
    use burn_std::{Bytes, FileReadCounter, QuantScheme, QuantStore, QuantValue, future::block_on};

    use super::*;
    use crate::{CustomOpPlan, CustomOpRule, OpKind};

    /// Runs its op collective, its one input taken as `input`.
    struct CollectiveRule {
        input: Sharding,
    }

    impl CustomOpRule for CollectiveRule {
        fn plan<'a>(&self, _op: &'a CustomOpIr) -> CustomOpPlan<'a> {
            CustomOpPlan::Collective {
                inputs: vec![self.input],
                outputs: vec![Sharding::Replicated],
            }
        }
    }

    /// Plans its op as a built-in op of `kind`.
    struct KindRule {
        kind: OpKind<'static>,
    }

    impl CustomOpRule for KindRule {
        fn plan<'a>(&self, _op: &'a CustomOpIr) -> CustomOpPlan<'a> {
            CustomOpPlan::Kind(self.kind)
        }
    }

    /// The custom op `id` reading `input` and writing `out`, of the input's shape.
    fn custom_op(id: &str, input: TensorIr, out: TensorId) -> OperationIr {
        OperationIr::Custom(CustomOpIr {
            id: id.into(),
            outputs: vec![TensorIr {
                id: out,
                status: TensorStatus::NotInit,
                ..input.clone()
            }],
            inputs: vec![input],
            scalars: Vec::new(),
        })
    }

    /// A Flex pair whose custom ops `ids` are each run collective, `input` as their input.
    fn collective_pair(
        ids: &[&str],
        input: Sharding,
        collectives: CollectiveOpRegistry<Flex>,
    ) -> ShardInterpreter<Flex> {
        let custom_ops = ids.iter().fold(
            ShardCustomOps::default().with_collectives(collectives),
            |custom_ops, id| custom_ops.with_rule(id, CollectiveRule { input }),
        );
        ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], custom_ops).unwrap()
    }

    fn assert_fails_on_every_member(
        interpreter: &ShardInterpreter<Flex>,
        id: &TensorId,
        why: &str,
    ) {
        for member in &interpreter.interpreters {
            let failure = member.read_failure(id).expect("The output failed");
            assert!(failure.contains(why), "{failure}");
        }
    }

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
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let id = interpreter.register_tensor_data(TensorData::new(vec![0f32; 24], [4, 6]));

        let split = interpreter
            .reshard(
                whole(id, TensorStatus::ReadOnly),
                Sharding::Split { dim: 1 },
            )
            .unwrap();

        for member in &interpreter.interpreters {
            assert!(member.read_failure(&id).is_some());
            assert!(member.read_failure(&split).is_none());
        }
    }

    #[test]
    fn data_an_op_consumes_split_is_cut_on_the_host() {
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let x = interpreter.register_tensor_data(TensorData::new(vec![1f32; 24], [4, 6]));
        let x = interpreter
            .reshard(
                whole(x, TensorStatus::ReadWrite),
                Sharding::Split { dim: 1 },
            )
            .unwrap();
        let path = std::env::temp_dir().join(format!("burn-shard-cut-{}", std::process::id()));
        std::fs::write(&path, TensorData::new(vec![1f32; 24], [6, 4]).as_bytes()).unwrap();
        let reads = FileReadCounter::new(&path);
        let w = interpreter.register_tensor_data(TensorData::from_bytes(
            Bytes::from_file(&path, 96, 0),
            [6, 4],
            DType::F32,
        ));
        let out = interpreter.new_tensor_id();

        // A lhs split along its columns asks for the rhs split along its rows.
        interpreter.register_op(OperationIr::Float(
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
        let read = block_on(interpreter.read(TensorIr {
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
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let id = interpreter.register_tensor_data(data);

        let refused = interpreter.reshard(
            TensorIr {
                dtype,
                ..whole(id, TensorStatus::ReadOnly)
            },
            Sharding::Split { dim: 0 },
        );

        assert_eq!(refused, Err(ShardingError::Quantized));
        for member in &interpreter.interpreters {
            assert!(member.read_failure(&id).is_some());
        }
    }

    #[test]
    fn a_collective_op_without_a_handler_fails_its_outputs() {
        let mut interpreter = collective_pair(
            &["gather"],
            Sharding::Replicated,
            CollectiveOpRegistry::new(),
        );
        let x = interpreter.register_tensor_data(TensorData::new(vec![1f32; 24], [4, 6]));
        let out = interpreter.new_tensor_id();

        interpreter.register_op(custom_op("gather", whole(x, TensorStatus::ReadOnly), out));

        assert_fails_on_every_member(&interpreter, &out, "No collective handler registered");
    }

    /// The failed input is replicated and the op asks for it split, so a resharding before the
    /// check would read the shards the failure never wrote.
    #[test]
    fn a_collective_op_reading_a_failed_input_fails_without_running() {
        let mut collectives = CollectiveOpRegistry::<Flex>::new();
        collectives.register("second", |_| panic!("The handler ran on a failed input"));
        let mut interpreter = collective_pair(
            &["first", "second"],
            Sharding::Split { dim: 0 },
            collectives,
        );
        let x = interpreter.register_tensor_data(TensorData::new(vec![1f32; 24], [4, 6]));
        let (failed, out) = (interpreter.new_tensor_id(), interpreter.new_tensor_id());

        interpreter.register_op(custom_op("first", whole(x, TensorStatus::ReadOnly), failed));
        interpreter.register_op(custom_op(
            "second",
            whole(failed, TensorStatus::ReadWrite),
            out,
        ));

        assert_fails_on_every_member(&interpreter, &out, "skipped 1 operation");
    }

    #[test]
    fn a_collective_op_whose_input_cannot_take_its_sharding_fails_its_outputs() {
        let mut interpreter = collective_pair(
            &["gather"],
            Sharding::Split { dim: 0 },
            CollectiveOpRegistry::new(),
        );
        let x = interpreter.register_tensor_data(TensorData::new(vec![1f32; 6], [1, 6]));
        let out = interpreter.new_tensor_id();
        let input = TensorIr {
            shape: [1, 6].into(),
            ..whole(x, TensorStatus::ReadOnly)
        };

        interpreter.register_op(custom_op("gather", input, out));

        assert_fails_on_every_member(&interpreter, &out, "shorter than");
    }

    /// A built-in reshape keeps this split, but a custom one reads its target shape from its own
    /// scalars, which no member's shard matches.
    #[test]
    fn a_custom_reshape_gathers_its_split_input() {
        let custom_ops = ShardCustomOps::default().with_rule(
            "reshape",
            KindRule {
                kind: OpKind::Reshape,
            },
        );
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], custom_ops).unwrap();
        let x = interpreter.register_tensor_data(TensorData::new(vec![1f32; 24], [4, 6]));
        let x = interpreter
            .reshard(
                whole(x, TensorStatus::ReadWrite),
                Sharding::Split { dim: 0 },
            )
            .unwrap();
        let out = interpreter.new_tensor_id();
        let OperationIr::Custom(mut op) =
            custom_op("reshape", whole(x, TensorStatus::ReadOnly), out)
        else {
            unreachable!()
        };
        op.outputs[0].shape = [4, 2, 3].into();

        interpreter.register_op(OperationIr::Custom(op));

        assert_eq!(interpreter.sharding(&out), Sharding::Replicated);
    }

    /// Splits its one input on dim 1 and writes two outputs split the same way.
    struct ShardedRule;

    impl CustomOpRule for ShardedRule {
        fn plan<'a>(&self, _op: &'a CustomOpIr) -> CustomOpPlan<'a> {
            let split = Sharding::Split { dim: 1 };
            CustomOpPlan::Sharded {
                inputs: vec![split],
                outputs: vec![split, split],
            }
        }
    }

    /// No kind describes an op of two outputs, so only its rule's shardings keep it split: each
    /// member runs its own handler on its half, and the halves make up the whole results.
    #[test]
    fn a_sharded_op_runs_on_each_members_shards() {
        use burn_backend::ops::FloatTensorOps;

        let mut handlers = burn_router::CustomOpRegistry::<Flex>::new();
        handlers.register("double_and_shift", |handles, ir, _device| {
            assert_eq!(
                ir.inputs[0].shape,
                [4, 3].into(),
                "a member reads its own half"
            );
            let x = handles.get_float_tensor::<Flex>(&ir.inputs[0]);
            let doubled = Flex::float_mul_scalar(x.clone(), 2.0f32.into());
            let shifted = Flex::float_add_scalar(x, 1.0f32.into());
            handles.register_float_tensor::<Flex>(&ir.outputs[0].id, doubled);
            handles.register_float_tensor::<Flex>(&ir.outputs[1].id, shifted);
        });
        let custom_ops = ShardCustomOps::default()
            .with_handlers(handlers)
            .with_rule("double_and_shift", ShardedRule);
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], custom_ops).unwrap();
        let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let x = interpreter.register_tensor_data(TensorData::new(values.clone(), [4, 6]));
        let (doubled, shifted) = (interpreter.new_tensor_id(), interpreter.new_tensor_id());
        let input = whole(x, TensorStatus::ReadOnly);
        let output = |id| TensorIr {
            id,
            status: TensorStatus::NotInit,
            ..input.clone()
        };

        interpreter.register_op(OperationIr::Custom(CustomOpIr {
            id: "double_and_shift".into(),
            outputs: vec![output(doubled), output(shifted)],
            inputs: vec![input.clone()],
            scalars: Vec::new(),
        }));

        let read = |interpreter: &mut ShardInterpreter<Flex>, id| {
            block_on(interpreter.read(whole(id, TensorStatus::ReadOnly)))
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
        };
        for id in [doubled, shifted] {
            assert_eq!(interpreter.sharding(&id), Sharding::Split { dim: 1 });
        }
        let expected = |f: fn(f32) -> f32| values.iter().copied().map(f).collect::<Vec<_>>();
        assert_eq!(read(&mut interpreter, doubled), expected(|x| 2.0 * x));
        assert_eq!(read(&mut interpreter, shifted), expected(|x| x + 1.0));
    }
}
