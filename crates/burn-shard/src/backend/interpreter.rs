use burn_backend::RandomKey;
use std::{any::Any, collections::HashMap};

use burn_backend::{DType, DTypeUsageSet, DeviceSettings, ExecutionError, Shape, TensorData};
use burn_ir::{
    BackendIr, BaseOperationIr, HandleKind, IrVisitorMut, OperationIr, TensorError, TensorId,
    TensorIr, TensorStatus,
};
use burn_router::TensorInterpreter;
use burn_std::{
    QuantConfig, Slice,
    distributed::{Shards, ShardsError},
    future::DynFut,
};

use super::{
    CollectiveMember, CollectiveOpRegistry, MemberDevice, ShardCustomOps, ShardError,
    custom_ops::{CustomOpRuleMap, MemberCustomOps},
    plan::{Execution, OpPlan},
    shardings::Shardings,
    shards::MemberShards,
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

    /// A new tensor holding `data`, uploaded whole to every member.
    fn register_tensor_data(&mut self, data: TensorData) -> TensorId;

    /// A new tensor of the shards a caller made, a `Shards<HandleKind<B>>` of the members'
    /// backend, with its whole shape and dtype.
    ///
    /// # Errors
    ///
    /// When the shards are of another backend than the members', or do not make up one tensor
    /// as their sharding says.
    fn register_shards(&mut self, shards: Box<dyn Any + Send>)
    -> Result<JoinedTensor, ShardsError>;

    /// The shards of `tensor`, as a `Shards<HandleKind<B>>`.
    fn take_shards(&mut self, tensor: TensorIr) -> Box<dyn Any + Send>;

    /// Runs `op` on every member's shards, after resharding each input as the op's
    /// rule asks for.
    fn register_op(&mut self, op: OperationIr);

    /// The whole value of `tensor`, its shards gathered onto the first member and read there.
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

/// The tensor a caller's shards make up once joined: its id on the sharded compute device, and
/// its whole shape and dtype.
pub struct JoinedTensor {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// The interpreter of a sharded compute device whose members run `B`: one [`TensorInterpreter`] per member, and
/// where each tensor's shards sit.
pub struct ShardInterpreter<B: BackendIr> {
    interpreters: Vec<TensorInterpreter<B>>,
    devices: Vec<B::Device>,
    shardings: Shardings,
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
            shardings: Shardings::default(),
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
        self.shardings.get(id)
    }

    fn num_members(&self) -> usize {
        self.interpreters.len()
    }

    fn register_tensor_data(&mut self, data: TensorData) -> TensorId {
        let id = self.new_tensor_id();
        for interpreter in &mut self.interpreters {
            interpreter.register_tensor_data_id(id, data.clone());
        }
        self.shardings.insert(id, Sharding::Replicated);
        id
    }

    fn register_shards(
        &mut self,
        shards: Box<dyn Any + Send>,
    ) -> Result<JoinedTensor, ShardsError> {
        // Shards of another backend than the members' are on no member's device.
        let Shards { shards, sharding } = *shards
            .downcast::<Shards<HandleKind<B>>>()
            .map_err(|_| ShardsError::Device { shard: 0 })?;
        let shards = MemberShards::new(shards);
        let shape = shards.whole_shape(sharding, &self.devices)?;
        let dtype = shards.dtype();
        let id = self.new_tensor_id();
        self.register_sharded(id, ShardedTensor::new(shards.into_handles()));
        self.shardings.insert(id, sharding);
        Ok(JoinedTensor { id, shape, dtype })
    }

    fn take_shards(&mut self, tensor: TensorIr) -> Box<dyn Any + Send> {
        let sharding = self.shardings.get(&tensor.id);
        let shards = self.sharded_tensor(&tensor, sharding).into_handles();
        Box::new(Shards { shards, sharding })
    }

    fn register_op(&mut self, op: OperationIr) {
        match op {
            // The data was registered with the tensor.
            OperationIr::Init(_) => {}
            OperationIr::Drop(tensor) => self.drop_tensor(tensor.id),
            op => self.run(op),
        }
    }

    fn read(&mut self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let sharding = self.shardings.get(&tensor.id);
        self.sharded_tensor(&tensor, sharding)
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
        let temporaries = match self.shard_inputs(&mut op, &plan) {
            Ok(temporaries) => temporaries,
            Err(error) => {
                let reason = format!("An input of the op cannot be resharded: {error}");
                return self.fail(&op, &TensorError::new(ExecutionError::with_context(reason)));
            }
        };
        for (id, sharding) in plan.output_shardings() {
            self.shardings.insert(*id, *sharding);
        }

        match plan.execution() {
            Execution::EveryMember => self.run_on_every_member(&op),
            Execution::FirstMemberCopied => self.run_copied_from_first_member(&op),
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

    /// Reshards every input of `op` as the plan asks for. Returns the temporaries to drop after
    /// the op.
    ///
    /// # Errors
    ///
    /// When an input cannot take the sharding the plan asks for, or its shards do not join,
    /// after dropping the temporaries made so far.
    fn shard_inputs(
        &mut self,
        op: &mut OperationIr,
        plan: &OpPlan,
    ) -> Result<Vec<TensorId>, ShardingError> {
        let inputs: Vec<TensorIr> = op.inputs().cloned().collect();
        let mut resharded = ReshardedInputs::default();
        for input in &inputs {
            if resharded.temporaries.contains_key(&input.id) {
                continue;
            }
            let target = plan.input_sharding(&input.id);
            let current = self.shardings.get(&input.id);
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
            match self.resharded(&read, current, target) {
                Ok(temporary) => resharded.temporaries.insert(input.id, temporary),
                Err(error) => {
                    let made: Vec<_> = resharded.temporaries.into_values().collect();
                    made.into_iter().for_each(|id| self.drop_tensor(id));
                    return Err(error);
                }
            };
        }
        op.visit_mut(&mut resharded);
        Ok(resharded.temporaries.into_values().collect())
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
        if let Sharding::Split { dim } = self.shardings.get(&tensor.id)
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

    fn sharded_tensor(&mut self, tensor: &TensorIr, sharding: Sharding) -> ShardedTensor<B> {
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
        let resharding = Resharding::of_tensor(current, target, tensor, self.interpreters.len())?;
        let shards = self
            .sharded_tensor(tensor, current)
            .reshard(resharding, &tensor.shape)?;
        let id = self.new_tensor_id();
        self.register_sharded(id, shards);
        self.shardings.insert(id, target);
        Ok(id)
    }

    fn register_sharded(&mut self, id: TensorId, shards: ShardedTensor<B>) {
        for (interpreter, handle) in self.interpreters.iter_mut().zip(shards.into_handles()) {
            interpreter.register_tensor_to_device(id, handle);
        }
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
    member: ShardMember,
    shardings: &'a Shardings,
}

impl IrVisitorMut for MemberShapes<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        *tensor = self
            .member
            .local_tensor(tensor, self.shardings.get(&tensor.id));
    }
}

#[cfg(test)]
mod tests {
    use burn_backend::ops::{DimSplit, FloatTensorOps, QTensorOps};
    use burn_flex::{Flex, FlexDevice};
    use burn_ir::CustomOpIr;
    use burn_std::{QuantScheme, QuantStore, QuantValue, ScaleDtype, future::block_on};

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

    /// The shards of `data` laid out as `sharding` says, and the value they make up: a partial
    /// sum is every member holding `data`, so the value is `data` times the members.
    fn joined(
        interpreter: &mut ShardInterpreter<Flex>,
        data: &TensorData,
        sharding: Sharding,
    ) -> (TensorId, TensorData) {
        let num_members = interpreter.num_members();
        let chunks: Vec<TensorData> = match sharding {
            Sharding::Split { dim } => DimSplit {
                dim,
                len: data.shape()[dim],
                members: num_members,
            }
            .data_chunks(data)
            .collect(),
            Sharding::Replicated | Sharding::PartialSum => vec![data.clone(); num_members],
        };
        let value = match sharding {
            Sharding::PartialSum => {
                let summed: Vec<f32> = data.iter::<f32>().map(|x| x * num_members as f32).collect();
                TensorData::new(summed, data.shape().clone())
            }
            _ => data.clone(),
        };
        let shards: Vec<HandleKind<Flex>> = chunks
            .into_iter()
            .map(|chunk| HandleKind::Float(Flex::float_from_data(chunk, &FlexDevice)))
            .collect();
        let id = interpreter
            .register_shards(Box::new(Shards { shards, sharding }))
            .unwrap()
            .id;
        (id, value)
    }

    #[test]
    fn every_sharding_reshards_to_every_other_and_keeps_its_value() {
        let data = TensorData::new((0..24).map(|i| i as f32).collect::<Vec<_>>(), [4, 6]);
        let shardings = [
            Sharding::Replicated,
            Sharding::PartialSum,
            Sharding::Split { dim: 0 },
            Sharding::Split { dim: 1 },
        ];
        for num_members in 1..=4 {
            let mut interpreter = ShardInterpreter::<Flex>::new(
                vec![FlexDevice; num_members],
                ShardCustomOps::default(),
            )
            .unwrap();
            for from in shardings {
                for to in shardings {
                    if to == Sharding::PartialSum && from != Sharding::PartialSum {
                        continue;
                    }
                    let (id, value) = joined(&mut interpreter, &data, from);

                    let moved = interpreter
                        .resharded(&whole(id, TensorStatus::ReadWrite), from, to)
                        .unwrap();

                    let case = format!("{num_members} members, {from:?} to {to:?}");
                    assert_eq!(interpreter.sharding(&moved), to, "{case}");
                    let read = block_on(interpreter.read(whole(moved, TensorStatus::ReadWrite)));
                    assert_eq!(read.unwrap(), value, "{case}");
                }
            }
        }
    }

    /// Its codes and scales are cut and joined as they are, so the value reads back exactly.
    #[test]
    fn a_quantized_split_reshards_to_another_dim_and_gathers_unchanged() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([2], ScaleDtype::F32);
        let codes: Vec<i8> = (0..32).collect();
        let scales: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let data = TensorData::quantized(codes, [4, 8], scheme, &scales, None);
        let tensor = |id| TensorIr {
            id,
            shape: [4, 8].into(),
            status: TensorStatus::ReadWrite,
            dtype: data.dtype(),
        };
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let rows = DimSplit {
            dim: 0,
            len: 4,
            members: 2,
        };
        let shards: Vec<HandleKind<Flex>> = rows
            .data_chunks(&data)
            .map(|chunk| HandleKind::Quantized(Flex::q_from_data(chunk, &FlexDevice)))
            .collect();
        let by_rows = interpreter
            .register_shards(Box::new(Shards {
                shards,
                sharding: Sharding::Split { dim: 0 },
            }))
            .unwrap()
            .id;

        let by_columns = interpreter
            .resharded(
                &tensor(by_rows),
                Sharding::Split { dim: 0 },
                Sharding::Split { dim: 1 },
            )
            .unwrap();
        let gathered = interpreter
            .resharded(
                &tensor(by_columns),
                Sharding::Split { dim: 1 },
                Sharding::Replicated,
            )
            .unwrap();

        assert_eq!(block_on(interpreter.read(tensor(gathered))).unwrap(), data);
    }

    /// Two shards quantized each on their own, split along the rows of a `[4, 2]` tensor.
    fn unjoinable_quantized(interpreter: &mut ShardInterpreter<Flex>) -> TensorIr {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let shard = |scale| {
            let data = TensorData::quantized(vec![0i8, 1, 2, 3], [2, 2], scheme, &[scale], None);
            HandleKind::<Flex>::Quantized(Flex::q_from_data(data, &FlexDevice))
        };
        let joined = interpreter
            .register_shards(Box::new(Shards {
                shards: vec![shard(0.5), shard(0.25)],
                sharding: Sharding::Split { dim: 0 },
            }))
            .unwrap();
        TensorIr {
            id: joined.id,
            shape: joined.shape,
            status: TensorStatus::ReadOnly,
            dtype: joined.dtype,
        }
    }

    #[test]
    fn quantized_shards_with_their_own_per_tensor_scales_are_never_joined() {
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let tensor = unjoinable_quantized(&mut interpreter);

        let read = block_on(interpreter.read(tensor.clone()));
        let gathered =
            interpreter.resharded(&tensor, Sharding::Split { dim: 0 }, Sharding::Replicated);

        assert!(read.is_err());
        assert_eq!(gathered, Err(ShardingError::QuantizedScales { shard: 1 }));
    }

    /// A custom op with no rule gathers its inputs: the split float input is gathered into a
    /// temporary before the quantized one fails to join.
    #[test]
    fn an_op_gathering_unjoinable_quantized_shards_fails_its_outputs() {
        let mut interpreter =
            ShardInterpreter::<Flex>::new(vec![FlexDevice, FlexDevice], ShardCustomOps::default())
                .unwrap();
        let (split, _) = joined(
            &mut interpreter,
            &TensorData::new(vec![1f32; 24], [4, 6]),
            Sharding::Split { dim: 0 },
        );
        let quantized = unjoinable_quantized(&mut interpreter);
        let out = interpreter.new_tensor_id();
        let OperationIr::Custom(mut op) =
            custom_op("plain", whole(split, TensorStatus::ReadOnly), out)
        else {
            unreachable!("custom_op builds a custom op")
        };
        op.inputs.push(quantized);

        interpreter.register_op(OperationIr::Custom(op));

        assert_fails_on_every_member(&interpreter, &out, "per-tensor or global scale");
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
        let data = TensorData::new(vec![1f32; 24], [4, 6]);
        let rows = DimSplit {
            dim: 0,
            len: 4,
            members: 2,
        };
        let shards: Vec<HandleKind<Flex>> = rows
            .data_chunks(&data)
            .map(|chunk| HandleKind::Float(Flex::float_from_data(chunk, &FlexDevice)))
            .collect();
        let x = interpreter
            .register_shards(Box::new(Shards {
                shards,
                sharding: Sharding::Split { dim: 0 },
            }))
            .unwrap()
            .id;
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
}
