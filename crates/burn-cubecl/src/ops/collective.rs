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
use core::cmp::Ordering;
use cubecl::{
    client::{Client, ClientGroup, GroupMember, MemberBuffers},
    ir::ElemType,
};

use crate::{
    CubeBackend,
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
        let first = &tensors[0];
        let (shape, dtype) = (first.meta.shape().clone(), first.dtype);
        let alike = tensors
            .iter()
            .all(|tensor| *tensor.meta.shape() == shape && tensor.dtype == dtype);
        let clients = tensors.iter().map(|tensor| tensor.client.clone()).collect();
        match alike
            .then(|| NativeGroup::new(clients, shape, dtype))
            .flatten()
            .filter(|group| exchange.runs_natively(group))
        {
            Some(group) => Self::Native(NativeCollective {
                group,
                exchange,
                tensors,
            }),
            None => Self::Emulated(tensors),
        }
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
    /// The group over `clients`, or `None` when cubecl cannot group them.
    fn new(clients: Vec<Client>, shape: Shape, dtype: DType) -> Option<Self> {
        ClientGroup::new(clients).ok().map(|clients| Self {
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

    fn runs_natively(self, group: &NativeGroup) -> bool {
        match self {
            Self::AllReduce { .. } => group.reduces(),
            Self::AllGather { .. } => true,
            Self::ReduceScatter { dim, .. } => group.reduces() && group.splits_evenly(dim),
            // cubecl splits and concatenates along one dimension, so two different ones are
            // emulated.
            Self::AllToAll { dim, concat_dim } => dim == concat_dim && group.splits_evenly(dim),
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
    use crate::{CubeDevice, fusion::CubeFusionHandle, ops::numeric::full_device_dtype};
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
        let group = NativeGroup::new(clients, shape.clone(), dtype)
            .filter(|group| exchange.runs_natively(group))?;
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
        let native = |plan: CollectivePlan| matches!(plan, CollectivePlan::Native(_));
        const SUM: Exchange = Exchange::AllReduce {
            op: ReduceOperation::Sum,
        };

        let even = || members([0, 1], [2, 2], DType::F32);
        assert!(native(CollectivePlan::new(even(), SUM)));
        assert!(native(CollectivePlan::new(
            even(),
            Exchange::AllGather { dim: 0 }
        )));
        assert!(native(CollectivePlan::new(
            even(),
            Exchange::ReduceScatter {
                dim: 0,
                op: ReduceOperation::Sum
            }
        )));
        assert!(native(CollectivePlan::new(
            even(),
            Exchange::all_to_all(1, 1)
        )));
        assert!(!native(CollectivePlan::new(
            even(),
            Exchange::all_to_all(0, 1)
        )));

        let uneven = || members([0, 1], [3, 3], DType::F32);
        assert!(native(CollectivePlan::new(
            uneven(),
            Exchange::AllGather { dim: 0 }
        )));
        assert!(!native(CollectivePlan::new(
            uneven(),
            Exchange::ReduceScatter {
                dim: 0,
                op: ReduceOperation::Sum
            }
        )));
        assert!(!native(CollectivePlan::new(
            uneven(),
            Exchange::all_to_all(0, 0)
        )));

        let unreduced = || members([0, 1], [2, 2], DType::I16);
        assert!(native(CollectivePlan::new(
            unreduced(),
            Exchange::AllGather { dim: 0 }
        )));
        assert!(!native(CollectivePlan::new(unreduced(), SUM)));

        let descending = || members([1, 0], [2, 2], DType::F32);
        assert!(!native(CollectivePlan::new(descending(), SUM)));
        let mut sorted = descending();
        let order = DeviceOrder::sort(&mut sorted);
        assert!(native(CollectivePlan::new(sorted.clone(), SUM)));
        let restored: Vec<_> = order
            .restored(sorted)
            .iter()
            .map(|tensor| tensor.device.clone())
            .collect();
        assert_eq!(restored, [devices[1].clone(), devices[0].clone()]);
        let gather = Exchange::AllGather { dim: 0 };
        assert!(!native(CollectivePlan::new(
            members([0, 0], [2, 2], DType::F32),
            gather
        )));
        assert!(!native(CollectivePlan::new(
            members([0, 1], [2, 3], DType::F32),
            gather
        )));
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
