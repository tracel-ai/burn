//! A member is one of the tensors a collective is given, as
//! [`CollectiveOps`](burn_backend::ops::CollectiveOps) defines it: member `i` is position `i` of
//! the input, on its own device, and receives position `i` of the output.

use burn_backend::{
    DType, DeviceOps, Shape,
    cubecl::{dtype_to_elem_type, reduce_operation_to_cubecl},
    distributed::ReduceOperation,
    ops::{CollectiveOps, EmulatedCollectives},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use core::{cmp::Ordering, fmt};
use cubecl::{
    client::{Client, ClientGroup, ClientGroupError, GroupMember, MemberBuffers},
    ir::ElemType,
};
use std::sync::{Mutex, PoisonError};

use crate::{
    CubeBackend, CubeDevice,
    kernel::{copy_contiguous, into_contiguous},
    ops::{numeric::empty_device_contiguous_dtype, permute},
    tensor::CubeTensor,
};

impl CollectiveOps<Self> for CubeBackend {
    fn float_all_reduce(
        mut tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        let order = DeviceOrder::sort(&mut tensors);
        let outputs = match CollectivePlan::new(tensors, Exchange::AllReduce { op }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_all_reduce(tensors, op)
            }
        };
        order.restored(outputs)
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::AllGather { dim }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_all_gather(tensors, dim)
            }
        }
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::ReduceScatter { dim, op }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::all_to_all(split_dim, concat_dim)) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn int_all_reduce(
        mut tensors: Vec<IntTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        let order = DeviceOrder::sort(&mut tensors);
        let outputs = match CollectivePlan::new(tensors, Exchange::AllReduce { op }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_all_reduce(tensors, op)
            }
        };
        order.restored(outputs)
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::AllGather { dim }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_all_gather(tensors, dim)
            }
        }
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::ReduceScatter { dim, op }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::all_to_all(split_dim, concat_dim)) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::AllGather { dim }) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::bool_all_gather(tensors, dim)
            }
        }
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        match CollectivePlan::new(tensors, Exchange::all_to_all(split_dim, concat_dim)) {
            CollectivePlan::Native(native) => native.run(),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::bool_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }
}

/// How a collective over the members runs.
enum CollectivePlan {
    /// Natively, by cubecl over a group of the members' devices: NCCL on CUDA.
    Native(NativeCollective),
    /// Emulated with transfers between devices, slices, concatenations and additions.
    Emulated(Vec<CubeTensor>),
}

impl CollectivePlan {
    /// Native when the members' tensors are alike and their devices form a group that runs
    /// `exchange`, emulated otherwise.
    fn new(tensors: Vec<CubeTensor>, exchange: Exchange) -> Self {
        match Self::native_group(&tensors, exchange) {
            Ok(group) => Self::Native(NativeCollective {
                group,
                exchange,
                tensors,
            }),
            Err(emulation) => {
                emulation.report(exchange);
                Self::Emulated(tensors)
            }
        }
    }

    /// The group that runs `exchange` natively over `tensors`, or why none does.
    fn native_group(tensors: &[CubeTensor], exchange: Exchange) -> Result<NativeGroup, Emulation> {
        let first = &tensors[0];
        let (shape, dtype) = (first.meta.shape().clone(), first.dtype);
        if tensors
            .iter()
            .any(|tensor| *tensor.meta.shape() != shape || tensor.dtype != dtype)
        {
            return Err(Emulation::UnlikeMembers);
        }
        let clients = tensors.iter().map(|tensor| tensor.client.clone()).collect();
        let group = NativeGroup::new(clients, shape, dtype)?;
        exchange.check(&group)?;
        Ok(group)
    }
}

/// How collectives between devices run, whatever tensors they are given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveTransport {
    /// Natively, by the transport between `runtime`'s devices, such as NCCL on CUDA. A reduction
    /// in a dtype outside `reduces` is still emulated.
    Native {
        /// The runtime's name, as cubecl gives it.
        runtime: &'static str,
        /// The dtypes the transport reduces.
        reduces: Vec<DType>,
    },
    /// Emulated, since cubecl forms no group of the devices.
    Emulated(ClientGroupError),
}

impl CollectiveTransport {
    /// The dtypes a reduction can be asked to run in.
    const REDUCIBLE: [DType; 13] = [
        DType::F64,
        DType::F32,
        DType::Flex32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
    ];

    /// The transport between `devices`, which a collective groups in ascending device order.
    pub fn new(devices: &[CubeDevice]) -> Self {
        let mut ranked = devices.to_vec();
        ranked.sort_by_key(|device| device.id());
        match ClientGroup::new(ranked.iter().map(CubeDevice::client).collect()) {
            Ok(group) => Self::Native {
                runtime: group.clients()[0].name(),
                reduces: Self::REDUCIBLE
                    .into_iter()
                    .filter(|dtype| group.reduces(dtype_to_elem_type(*dtype)))
                    .collect(),
            },
            Err(error) => Self::Emulated(error),
        }
    }
}

/// Why a collective is emulated rather than run by the transport between the members' devices.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Emulation {
    /// The members' tensors differ in shape or dtype.
    UnlikeMembers,
    /// cubecl forms no group of the members' devices.
    Ungrouped(ClientGroupError),
    /// The transport does not reduce the members' dtype.
    NotReduced(DType),
    /// The collective's dimension does not split evenly across the members.
    UnevenSplit,
    /// An all-to-all splits and concatenates along two dimensions, and cubecl's along one.
    TwoDimensions,
}

impl Emulation {
    /// Logs that `exchange` is emulated, once per process for each collective and reason.
    fn report(self, exchange: Exchange) {
        static REPORTED: Mutex<Vec<(&str, Emulation)>> = Mutex::new(Vec::new());
        let collective = exchange.name();
        let mut reported = REPORTED.lock().unwrap_or_else(PoisonError::into_inner);
        if reported
            .iter()
            .any(|(reported, emulation)| *reported == collective && *emulation == self)
        {
            return;
        }
        log::log!(
            self.level(),
            "{collective} is emulated {}, which is slower than natively: {self}",
            self.path()
        );
        reported.push((collective, self));
    }

    /// How the emulation moves data. Only devices without a transport between them stage
    /// through the host; otherwise the transfers stay between devices.
    fn path(&self) -> &'static str {
        match self {
            Self::Ungrouped(
                ClientGroupError::MixedRuntimes | ClientGroupError::NoDeviceTransport { .. },
            ) => "through the host",
            _ => "by separate transfers and kernels",
        }
    }

    fn level(&self) -> log::Level {
        match self {
            // A group that never forms is a choice of runtimes or of build, and cubecl warns on
            // its own when NCCL or RCCL is missing.
            Self::Ungrouped(
                ClientGroupError::MixedRuntimes
                | ClientGroupError::NoDeviceTransport { .. }
                | ClientGroupError::BlockingDeviceHandle
                | ClientGroupError::Empty,
            ) => log::Level::Info,
            Self::Ungrouped(ClientGroupError::NotAscending { .. })
            | Self::UnlikeMembers
            | Self::NotReduced(_)
            | Self::UnevenSplit
            | Self::TwoDimensions => log::Level::Warn,
        }
    }
}

impl fmt::Display for Emulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnlikeMembers => f.write_str("the members' tensors differ in shape or dtype"),
            Self::Ungrouped(error) => write!(f, "{error}"),
            Self::NotReduced(dtype) => write!(f, "the transport does not reduce {dtype:?}"),
            Self::UnevenSplit => {
                f.write_str("its dimension does not split evenly across the members")
            }
            Self::TwoDimensions => f.write_str("it splits and concatenates along two dimensions"),
        }
    }
}

impl From<ClientGroupError> for Emulation {
    fn from(error: ClientGroupError) -> Self {
        Self::Ungrouped(error)
    }
}

/// A cubecl group over the members' devices, in ascending order, with the shape and dtype every
/// member's tensor shares.
struct NativeGroup {
    clients: ClientGroup,
    shape: Shape,
    dtype: DType,
}

impl NativeGroup {
    /// The group over `clients`, or why cubecl cannot group them.
    fn new(clients: Vec<Client>, shape: Shape, dtype: DType) -> Result<Self, ClientGroupError> {
        ClientGroup::new(clients).map(|clients| Self {
            clients,
            shape,
            dtype,
        })
    }

    fn members(&self) -> usize {
        self.clients.clients().len()
    }

    fn elem(&self) -> ElemType {
        dtype_to_elem_type(self.dtype)
    }

    fn reduces(&self) -> bool {
        self.clients.reduces(self.elem())
    }

    fn splits_evenly(&self, dim: usize) -> bool {
        let (members, len) = (self.members(), self.shape[dim]);
        len >= members && len.is_multiple_of(members)
    }
}

/// A collective every member contributes a tensor to and receives one from, as cubecl runs it
/// natively: along one dimension, moved first, since cubecl cuts and joins whole buffers.
#[derive(Debug, Clone, Copy)]
enum Exchange {
    AllReduce {
        op: ReduceOperation,
    },
    AllGather {
        dim: usize,
    },
    ReduceScatter {
        dim: usize,
        op: ReduceOperation,
    },
    /// Split and concatenated along one dimension; along two, it is not native.
    AllToAll {
        dim: usize,
        concat_dim: usize,
    },
}

impl Exchange {
    fn all_to_all(split_dim: usize, concat_dim: usize) -> Self {
        Self::AllToAll {
            dim: split_dim,
            concat_dim,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::AllReduce { .. } => "all-reduce",
            Self::AllGather { .. } => "all-gather",
            Self::ReduceScatter { .. } => "reduce-scatter",
            Self::AllToAll { .. } => "all-to-all",
        }
    }

    /// Whether `group` runs this collective natively, or why it does not.
    fn check(self, group: &NativeGroup) -> Result<(), Emulation> {
        let reduced = || match group.reduces() {
            true => Ok(()),
            false => Err(Emulation::NotReduced(group.dtype)),
        };
        let split = |dim| match group.splits_evenly(dim) {
            true => Ok(()),
            false => Err(Emulation::UnevenSplit),
        };
        match self {
            Self::AllReduce { .. } => reduced(),
            Self::AllGather { .. } => Ok(()),
            Self::ReduceScatter { dim, .. } => reduced().and_then(|()| split(dim)),
            Self::AllToAll { dim, concat_dim } if dim != concat_dim => {
                Err(Emulation::TwoDimensions)
            }
            Self::AllToAll { dim, .. } => split(dim),
        }
    }

    fn leading(self) -> LeadingDim {
        LeadingDim::new(match self {
            Self::AllReduce { .. } => 0,
            Self::AllGather { dim }
            | Self::ReduceScatter { dim, .. }
            | Self::AllToAll { dim, .. } => dim,
        })
    }

    /// What a member contributes: `tensor` with the collective's dimension leading, dense.
    fn source(self, tensor: CubeTensor) -> CubeTensor {
        dense(self.leading().moved_ahead(tensor))
    }

    /// Where a member receives, for `source` over `members`. An all-reduce lands on its source
    /// when nothing else holds it.
    fn output(self, source: &CubeTensor, members: usize) -> CubeTensor {
        let len = |len: usize| match self {
            Self::AllReduce { .. } | Self::AllToAll { .. } => len,
            Self::AllGather { .. } => len * members,
            Self::ReduceScatter { .. } => len / members,
        };
        match self {
            Self::AllReduce { .. } if source.can_mut() => source.clone(),
            _ => output_like(source, len),
        }
    }

    /// The output with the collective's dimension back where it was.
    fn restored(self, output: CubeTensor) -> CubeTensor {
        self.leading().restored(output)
    }

    /// Queues every member's part over `clients`.
    fn run_group(self, clients: &ClientGroup, elem: ElemType, buffers: Vec<MemberBuffers>) {
        match self {
            Self::AllReduce { op } => {
                clients.all_reduce(buffers, elem, reduce_operation_to_cubecl(op))
            }
            Self::AllGather { .. } => clients.all_gather(buffers),
            Self::ReduceScatter { op, .. } => {
                clients.reduce_scatter(buffers, elem, reduce_operation_to_cubecl(op))
            }
            Self::AllToAll { .. } => clients.all_to_all(buffers),
        }
    }

    /// Queues `member`'s own part.
    fn run_member(self, member: &GroupMember, elem: ElemType, buffers: MemberBuffers) {
        match self {
            Self::AllReduce { op } => {
                member.all_reduce(buffers, elem, reduce_operation_to_cubecl(op))
            }
            Self::AllGather { .. } => member.all_gather(buffers),
            Self::ReduceScatter { op, .. } => {
                member.reduce_scatter(buffers, elem, reduce_operation_to_cubecl(op))
            }
            Self::AllToAll { .. } => member.all_to_all(buffers),
        }
    }
}

/// A collective the members' group runs natively, with each member's tensor.
struct NativeCollective {
    group: NativeGroup,
    exchange: Exchange,
    tensors: Vec<CubeTensor>,
}

impl NativeCollective {
    fn run(self) -> Vec<CubeTensor> {
        let Self {
            group,
            exchange,
            tensors,
        } = self;
        let members = group.members();
        let sources: Vec<CubeTensor> = tensors
            .into_iter()
            .map(|tensor| exchange.source(tensor))
            .collect();
        let outputs: Vec<CubeTensor> = sources
            .iter()
            .map(|source| exchange.output(source, members))
            .collect();
        let buffers = sources
            .iter()
            .zip(&outputs)
            .map(|(source, output)| member_buffers(source, output))
            .collect();
        exchange.run_group(&group.clients, group.elem(), buffers);
        outputs
            .into_iter()
            .map(|output| exchange.restored(output))
            .collect()
    }
}

fn member_buffers(source: &CubeTensor, output: &CubeTensor) -> MemberBuffers {
    MemberBuffers {
        src: source.handle.clone(),
        dst: output.handle.clone(),
    }
}

/// The tensor with its elements filling its buffer in row-major order, since cubecl's
/// collectives move whole buffers.
fn dense(tensor: CubeTensor) -> CubeTensor {
    let tensor = into_contiguous(tensor);
    match tensor.is_contiguous_buffer() {
        true => tensor,
        // `is_contiguous` ignores a dimension of size one, which can still hold a pitched row.
        false => copy_contiguous(tensor),
    }
}

/// An empty dense tensor like `source`, its leading length `len` of the source's.
fn output_like(source: &CubeTensor, len: impl Fn(usize) -> usize) -> CubeTensor {
    let mut shape = source.meta.shape().clone();
    shape[0] = len(shape[0]);
    empty_device_contiguous_dtype(
        source.client.clone(),
        source.device.clone(),
        shape,
        source.dtype,
    )
}

/// A collective's dimension moved first: cubecl cuts and joins whole buffers, which is cutting and
/// joining along the leading dimension of a dense tensor.
struct LeadingDim {
    dim: usize,
}

/// The ascending device order a native group takes its members in. An all-reduce gives every
/// member the same result, so it runs in that order and hands each output back to its member.
struct DeviceOrder {
    /// The caller's position of each member, in device order.
    positions: Vec<usize>,
}

impl LeadingDim {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn moved_ahead(&self, tensor: CubeTensor) -> CubeTensor {
        if self.dim == 0 {
            return tensor;
        }
        let axes: Vec<usize> = core::iter::once(self.dim)
            .chain((0..tensor.meta.shape().num_dims()).filter(|axis| *axis != self.dim))
            .collect();
        permute(tensor, &axes)
    }

    /// The output with the dimension back where it was.
    fn restored(&self, tensor: CubeTensor) -> CubeTensor {
        if self.dim == 0 {
            return tensor;
        }
        let axes: Vec<usize> = (0..tensor.meta.shape().num_dims())
            .map(|axis| match axis.cmp(&self.dim) {
                Ordering::Less => axis + 1,
                Ordering::Equal => 0,
                Ordering::Greater => axis,
            })
            .collect();
        permute(tensor, &axes)
    }
}

impl DeviceOrder {
    fn sort(tensors: &mut [CubeTensor]) -> Self {
        let mut positions: Vec<usize> = (0..tensors.len()).collect();
        positions.sort_by_key(|&position| tensors[position].device.id());
        // Both sorts are stable, so `positions` names where each sorted member came from.
        tensors.sort_by_key(|tensor| tensor.device.id());
        Self { positions }
    }

    /// The outputs back in the members' order.
    fn restored<T>(&self, outputs: Vec<T>) -> Vec<T> {
        if self.positions.is_sorted() {
            return outputs;
        }
        let mut outputs: Vec<_> = self.positions.iter().zip(outputs).collect();
        outputs.sort_by_key(|(position, _)| **position);
        outputs.into_iter().map(|(_, output)| output).collect()
    }
}

#[cfg(feature = "fusion")]
mod parts {
    //! Each member's part of a native collective, run by its own device inside Fusion: see
    //! [`FusionBackend::collective_parts`](burn_fusion::FusionBackend::collective_parts).

    use super::*;
    use crate::{fusion::CubeFusionHandle, ops::numeric::full_device_dtype};
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_fusion::{Collective, CollectivePart};
    use burn_ir::{BackendIr, TensorHandle};
    use cubecl::prelude::InputScalar;

    impl Exchange {
        fn from_fusion(collective: Collective) -> Self {
            match collective {
                Collective::AllReduce { op } => Self::AllReduce { op },
                Collective::AllGather { dim } => Self::AllGather { dim },
                Collective::ReduceScatter { dim, op } => Self::ReduceScatter { dim, op },
                Collective::AllToAll {
                    split_dim,
                    concat_dim,
                } => Self::all_to_all(split_dim, concat_dim),
            }
        }
    }

    /// Each member's part of `collective` over members of `shape` and `dtype` on `devices`, when
    /// their group runs it natively. An all-reduce runs in device order whatever the members'
    /// order; the other collectives place chunks by rank, so their members come in it.
    pub(crate) fn collective_parts(
        devices: &[CubeDevice],
        shape: &Shape,
        dtype: DType,
        collective: Collective,
    ) -> Option<Vec<Box<dyn CollectivePart<CubeBackend>>>> {
        let exchange = Exchange::from_fusion(collective);
        let mut ranked: Vec<CubeDevice> = devices.to_vec();
        let order = match exchange {
            Exchange::AllReduce { .. } => {
                let mut positions: Vec<usize> = (0..devices.len()).collect();
                positions.sort_by_key(|&position| devices[position].id());
                ranked.sort_by_key(|device| device.id());
                DeviceOrder { positions }
            }
            _ => DeviceOrder {
                positions: (0..devices.len()).collect(),
            },
        };
        let clients = ranked.iter().map(|device| device.client()).collect();
        let group = NativeGroup::new(clients, shape.clone(), dtype).ok()?;
        exchange.check(&group).ok()?;
        let (members, elem) = (group.members(), group.elem());
        let parts = group
            .clients
            .members()
            .into_iter()
            .zip(ranked)
            .map(|(member, device)| {
                Box::new(MemberPart {
                    member,
                    exchange,
                    members,
                    elem,
                    stand_in: StandIn {
                        device,
                        shape: shape.clone(),
                        dtype,
                    },
                }) as Box<dyn CollectivePart<CubeBackend>>
            })
            .collect();
        Some(order.restored(parts))
    }

    /// One member's part, run by its own device.
    struct MemberPart {
        member: GroupMember,
        exchange: Exchange,
        members: usize,
        elem: ElemType,
        stand_in: StandIn,
    }

    /// What a member contributes in place of a tensor that carries a failure: as large as the
    /// tensor, so the others' parts complete, and NaN where a float can hold it, so that what
    /// they receive shows it rather than passing for a result. Integers and booleans contribute
    /// zeros, which the others cannot tell from a member's values.
    struct StandIn {
        device: CubeDevice,
        shape: Shape,
        dtype: DType,
    }

    impl MemberPart {
        fn exchange(&self, tensor: CubeTensor) -> CubeTensor {
            let source = self.exchange.source(tensor);
            let output = self.exchange.output(&source, self.members);
            self.exchange
                .run_member(&self.member, self.elem, member_buffers(&source, &output));
            self.exchange.restored(output)
        }
    }

    impl CollectivePart<CubeBackend> for MemberPart {
        fn run(&self, tensor: TensorHandle<CubeFusionHandle>) -> CubeFusionHandle {
            let output = self.exchange(CubeBackend::float_tensor(tensor));
            CubeBackend::float_tensor_handle(output)
        }

        fn stand_in(&self) {
            let StandIn {
                device,
                shape,
                dtype,
            } = &self.stand_in;
            let fill = match dtype.is_float() {
                true => f32::NAN,
                false => 0.0,
            };
            let tensor = full_device_dtype(
                device.client(),
                shape.clone(),
                device.clone(),
                InputScalar::new(fill, dtype_to_storage_type(*dtype)),
                *dtype,
            );
            self.exchange(tensor);
        }
    }
}

#[cfg(feature = "fusion")]
pub(crate) use parts::collective_parts;

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::CubeDevice;
    use burn_backend::DType;
    use burn_std::Shape;
    use cubecl::cuda::CudaDevice;

    /// A collective runs natively only where a group forms, reduces the type and splits evenly;
    /// anything else must be emulated, and a plan that silently stopped holding would pass every
    /// value test.
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn each_collective_runs_natively_only_where_the_members_allow_it() {
        let devices = [0, 1].map(|index| CubeDevice::Cuda(CudaDevice { index }));
        let members = |order: [usize; 2], rows: [usize; 2], dtype: DType| -> Vec<CubeTensor> {
            order
                .iter()
                .zip(rows)
                .map(|(member, rows)| {
                    let device = devices[*member].clone();
                    let shape = Shape::new([rows, 2]);
                    empty_device_contiguous_dtype(device.client(), device, shape, dtype)
                })
                .collect()
        };
        let emulation = |tensors: Vec<CubeTensor>, exchange| {
            CollectivePlan::native_group(&tensors, exchange).err()
        };
        const SUM: Exchange = Exchange::AllReduce {
            op: ReduceOperation::Sum,
        };
        const SCATTER: Exchange = Exchange::ReduceScatter {
            dim: 0,
            op: ReduceOperation::Sum,
        };
        let gather = Exchange::AllGather { dim: 0 };

        let even = || members([0, 1], [2, 2], DType::F32);
        assert_eq!(emulation(even(), SUM), None);
        assert_eq!(emulation(even(), gather), None);
        assert_eq!(emulation(even(), SCATTER), None);
        assert_eq!(emulation(even(), Exchange::all_to_all(1, 1)), None);
        assert_eq!(
            emulation(even(), Exchange::all_to_all(0, 1)),
            Some(Emulation::TwoDimensions)
        );

        let uneven = || members([0, 1], [3, 3], DType::F32);
        assert_eq!(emulation(uneven(), gather), None);
        assert_eq!(emulation(uneven(), SCATTER), Some(Emulation::UnevenSplit));
        assert_eq!(
            emulation(uneven(), Exchange::all_to_all(0, 0)),
            Some(Emulation::UnevenSplit)
        );

        let unreduced = || members([0, 1], [2, 2], DType::I16);
        assert_eq!(emulation(unreduced(), gather), None);
        assert_eq!(
            emulation(unreduced(), SUM),
            Some(Emulation::NotReduced(DType::I16))
        );

        let descending = || members([1, 0], [2, 2], DType::F32);
        assert!(matches!(
            emulation(descending(), SUM),
            Some(Emulation::Ungrouped(ClientGroupError::NotAscending { .. }))
        ));
        let mut sorted = descending();
        let order = DeviceOrder::sort(&mut sorted);
        assert_eq!(emulation(sorted.clone(), SUM), None);
        let restored: Vec<_> = order
            .restored(sorted)
            .iter()
            .map(|tensor| tensor.device.clone())
            .collect();
        assert_eq!(restored, [devices[1].clone(), devices[0].clone()]);
        assert!(matches!(
            emulation(members([0, 0], [2, 2], DType::F32), gather),
            Some(Emulation::Ungrouped(ClientGroupError::NotAscending { .. }))
        ));
        assert_eq!(
            emulation(members([0, 1], [2, 3], DType::F32), gather),
            Some(Emulation::UnlikeMembers)
        );
    }

    /// The transport sorts the devices as a collective does, so their order cannot make it emulated.
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn devices_in_any_order_name_their_runtime_and_what_it_reduces() {
        let devices = [1, 0].map(|index| CubeDevice::Cuda(CudaDevice { index }));
        match CollectiveTransport::new(&devices) {
            CollectiveTransport::Native { runtime, reduces } => {
                assert_eq!(runtime, "cuda");
                assert!(reduces.contains(&DType::F32), "{reduces:?}");
                assert!(!reduces.contains(&DType::I16), "{reduces:?}");
            }
            emulated => panic!("{emulated:?}"),
        }
    }

    /// A member whose tensor carries a failure still joins its peers, so none of them waits
    /// forever, and what they receive shows the failure as NaN rather than passing for a sum.
    #[cfg(feature = "fusion")]
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn a_member_standing_in_for_a_failed_tensor_lets_its_peers_finish() {
        use crate::fusion::CubeFusionHandle;
        use burn_fusion::Collective;
        use burn_ir::TensorHandle;

        let devices = [0, 1].map(|index| CubeDevice::Cuda(CudaDevice { index }));
        let shape = Shape::new([4, 2]);
        let parts = super::collective_parts(
            &devices,
            &shape,
            DType::F32,
            Collective::AllReduce {
                op: ReduceOperation::Sum,
            },
        )
        .expect("two CUDA devices run an all-reduce natively");
        let mut parts = parts.into_iter();
        let (failed, peer) = (parts.next().unwrap(), parts.next().unwrap());
        let device = devices[1].clone();
        let tensor =
            empty_device_contiguous_dtype(device.client(), device, shape.clone(), DType::F32);

        // Each part from a thread of its own, as each member's stream runs it; a read on the
        // thread that queued the part waits for it.
        let peer = std::thread::spawn(move || {
            let handle: CubeFusionHandle = tensor.into();
            let received = peer.run(TensorHandle { handle, shape });
            received.client.read_one(received.handle).unwrap()
        });
        failed.stand_in();
        let bytes = peer.join().unwrap();

        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_ne_bytes(*word))
            .collect();
        assert!(values.iter().all(|value| value.is_nan()), "{values:?}");
    }
}
