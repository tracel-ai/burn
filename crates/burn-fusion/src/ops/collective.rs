//! A member is one of the tensors a collective is given, as
//! [`CollectiveOps`](burn_backend::ops::CollectiveOps) defines it: member `i` is position `i` of
//! the input, on its own device, and receives position `i` of the output.

use burn_backend::{
    DType, ExecutionError, Shape, TensorMetadata,
    distributed::ReduceOperation,
    ops::{CollectiveOps, EmulatedCollectives},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorError, TensorHandle, TensorIr};
use burn_std::device_handle::SpawnedCall;
use core::marker::PhantomData;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::{
    Client, Fusion, FusionBackend, FusionRuntime, FusionTensor,
    backend::server_error,
    get_client,
    stream::{Operation, StreamId},
};

impl<B: FusionBackend> CollectiveOps<Self> for Fusion<B> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
            Collective::AllReduce { op },
            |inner| B::float_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self>::float_all_reduce(tensors, op),
        )
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
            Collective::AllGather { dim },
            |inner| B::float_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self>::float_all_gather(tensors, dim),
        )
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
            Collective::ReduceScatter { dim, op },
            |inner| B::float_reduce_scatter(inner, dim, op),
            |tensors| EmulatedCollectives::<Self>::float_reduce_scatter(tensors, dim, op),
        )
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
            Collective::AllToAll {
                split_dim,
                concat_dim,
            },
            |inner| B::float_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::float_all_to_all(tensors, split_dim, concat_dim),
        )
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
            Collective::AllReduce { op },
            |inner| B::int_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self>::int_all_reduce(tensors, op),
        )
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
            Collective::AllGather { dim },
            |inner| B::int_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self>::int_all_gather(tensors, dim),
        )
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
            Collective::ReduceScatter { dim, op },
            |inner| B::int_reduce_scatter(inner, dim, op),
            |tensors| EmulatedCollectives::<Self>::int_reduce_scatter(tensors, dim, op),
        )
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
            Collective::AllToAll {
                split_dim,
                concat_dim,
            },
            |inner| B::int_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::int_all_to_all(tensors, split_dim, concat_dim),
        )
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        InnerCollectives::<B>::bool(
            tensors,
            Collective::AllGather { dim },
            |inner| B::bool_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self>::bool_all_gather(tensors, dim),
        )
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        InnerCollectives::<B>::bool(
            tensors,
            Collective::AllToAll {
                split_dim,
                concat_dim,
            },
            |inner| B::bool_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::bool_all_to_all(tensors, split_dim, concat_dim),
        )
    }
}

/// The collectives run on Fusion's inner backend, whose native collective Fusion would otherwise
/// hide behind transfers: queued as one part per member on each member's own stream when the
/// backend runs it so, or else run on the caller, each member resolved from Fusion's tensors and
/// the outputs registered back.
struct InnerCollectives<B> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> InnerCollectives<B> {
    fn float<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        collective: Collective,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            collective,
            |tensor| tensor.client.clone().resolve_carrying_float::<B>(tensor),
            B::float_tensor_handle,
            inner,
            emulated,
        )
    }

    fn int<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        collective: Collective,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<IntTensor<B>>) -> Vec<IntTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            collective,
            |tensor| tensor.client.clone().resolve_carrying_int::<B>(tensor),
            B::int_tensor_handle,
            inner,
            emulated,
        )
    }

    fn bool<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        collective: Collective,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<BoolTensor<B>>) -> Vec<BoolTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            collective,
            |tensor| tensor.client.clone().resolve_carrying_bool::<B>(tensor),
            B::bool_tensor_handle,
            inner,
            emulated,
        )
    }

    /// Queues `collective`'s parts on the members' streams when the backend runs them so.
    /// Otherwise runs `inner` on what every member's stream resolves to, every member's server
    /// resolving at once; when a member's tensor carries a failure, `emulated` runs on Fusion's
    /// own tensors instead, so every output carries it.
    fn run<T, R, H, I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        collective: Collective,
        resolve: R,
        handle: H,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        T: TensorMetadata<Device = Device<B>>,
        R: Fn(FusionTensor<B::FusionRuntime>) -> SpawnedCall<Result<T, TensorError>>,
        H: Fn(T) -> B::Handle,
        I: FnOnce(Vec<T>) -> Vec<T>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        let tensors = match MemberParts::<B>::queue(tensors, collective) {
            Ok(outputs) => return outputs,
            Err(tensors) => tensors,
        };
        let resolving: Vec<_> = tensors
            .into_iter()
            .map(|tensor| {
                let metadata = MemberMetadata {
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    client: tensor.client.clone(),
                };
                (resolve(tensor), metadata)
            })
            .collect();
        let resolved: Vec<_> = resolving
            .into_iter()
            .map(|(call, metadata)| {
                let resolved = call
                    .join()
                    .unwrap_or_else(|err| Err(TensorError::new(server_error(err))));
                (resolved, metadata)
            })
            .collect();
        if resolved.iter().all(|(resolved, _)| resolved.is_ok()) {
            let inner_tensors = resolved
                .into_iter()
                .map(|(resolved, _)| resolved.expect("every member resolved"))
                .collect();
            return inner(inner_tensors)
                .into_iter()
                .map(|tensor| Self::registered(tensor, &handle))
                .collect();
        }
        let tensors = resolved
            .into_iter()
            .map(|(resolved, metadata)| match resolved {
                Ok(tensor) => Self::registered(tensor, &handle),
                Err(error) => metadata.carrying(error),
            })
            .collect();
        emulated(tensors)
    }

    fn registered<T, H>(tensor: T, handle: &H) -> FusionTensor<B::FusionRuntime>
    where
        T: TensorMetadata<Device = Device<B>>,
        H: Fn(T) -> B::Handle,
    {
        let client = get_client::<B>(&tensor.device());
        let (shape, dtype) = (tensor.shape(), tensor.dtype());
        let id = client.register_tensor_handle(handle(tensor));
        FusionTensor::new(id, shape, dtype, client, StreamId::current())
    }
}

/// What a member's tensor was, so a failure resolving it can stand in for it.
struct MemberMetadata<R: FusionRuntime> {
    shape: Shape,
    dtype: DType,
    client: Client<R>,
}

impl<R: FusionRuntime> MemberMetadata<R> {
    fn carrying(self, error: TensorError) -> FusionTensor<R> {
        let id = self.client.register_error(error.propagated());
        FusionTensor::new(id, self.shape, self.dtype, self.client, StreamId::current())
    }
}

/// A collective every member contributes one tensor to and receives one tensor from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective {
    /// See [`CollectiveOps::float_all_reduce`].
    AllReduce {
        /// How the members' tensors combine.
        op: ReduceOperation,
    },
    /// See [`CollectiveOps::float_all_gather`].
    AllGather {
        /// The dimension the members' tensors are concatenated along.
        dim: usize,
    },
    /// See [`CollectiveOps::float_reduce_scatter`].
    ReduceScatter {
        /// The dimension the reduced tensor is chunked along.
        dim: usize,
        /// How the members' tensors combine.
        op: ReduceOperation,
    },
    /// See [`CollectiveOps::float_all_to_all`].
    AllToAll {
        /// The dimension each member's tensor is chunked along.
        split_dim: usize,
        /// The dimension the received chunks are concatenated along.
        concat_dim: usize,
    },
}

impl Collective {
    /// The shape each of `members` members receives from members of `shape`, whose chunked
    /// dimension divides evenly.
    fn output_shape(self, shape: &Shape, members: usize) -> Shape {
        let mut output = shape.clone();
        match self {
            Self::AllReduce { .. } => {}
            Self::AllGather { dim } => output[dim] *= members,
            Self::ReduceScatter { dim, .. } => output[dim] /= members,
            Self::AllToAll {
                split_dim,
                concat_dim,
            } => {
                output[split_dim] /= members;
                output[concat_dim] *= members;
            }
        }
        output
    }

    /// The identifier a member's part carries in its stream's IR.
    fn part_id(self) -> &'static str {
        match self {
            Self::AllReduce { .. } => "collective::all_reduce",
            Self::AllGather { .. } => "collective::all_gather",
            Self::ReduceScatter { .. } => "collective::reduce_scatter",
            Self::AllToAll { .. } => "collective::all_to_all",
        }
    }
}

/// One member's part of a [`Collective`], run by that member's own device, from
/// [`FusionBackend::collective_parts`].
///
/// The members wait for each other on their devices, so every member's part runs: its own, or in
/// its place [`stand_in`](Self::stand_in) when its tensor carries a failure.
pub trait CollectivePart<B: FusionBackend>: Send + Sync {
    /// Runs this member's part on its tensor and returns the member's output.
    fn run(&self, tensor: TensorHandle<B::Handle>) -> B::Handle;

    /// Joins the other members without this member's tensor, which carries a failure; what it
    /// receives is discarded.
    fn stand_in(&self);
}

/// Each member's part of a collective queued on its own stream, so the caller never waits for the
/// members.
struct MemberParts<B> {
    _backend: PhantomData<B>,
}

/// Held while one collective's parts are queued, so that every member's stream receives the
/// collectives in one order: members running two collectives in different orders wait on each
/// other forever.
static PART_ORDER: Mutex<()> = Mutex::new(());

impl<B: FusionBackend> MemberParts<B> {
    /// Each member's output of `collective`, queued as a part on the member's stream and started
    /// at once; the tensors back when the backend does not run the collective so over them.
    #[allow(clippy::type_complexity)]
    fn queue(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        collective: Collective,
    ) -> Result<Vec<FusionTensor<B::FusionRuntime>>, Vec<FusionTensor<B::FusionRuntime>>> {
        let (shape, dtype) = (tensors[0].shape.clone(), tensors[0].dtype);
        if tensors
            .iter()
            .any(|tensor| tensor.shape != shape || tensor.dtype != dtype)
        {
            return Err(tensors);
        }
        let devices: Vec<_> = tensors
            .iter()
            .map(|tensor| tensor.client.device().clone())
            .collect();
        let Some(parts) = B::collective_parts(&devices, &shape, dtype, collective) else {
            return Err(tensors);
        };
        let output = collective.output_shape(&shape, tensors.len());
        let stream = StreamId::current();
        PartsInFlight::enter(tensors.len());
        let _order = PART_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(tensors
            .into_iter()
            .zip(parts)
            .map(|(tensor, part)| {
                let client = tensor.client.clone();
                let output = TensorIr::uninit(client.create_empty_handle(), output.clone(), dtype);
                let desc = CustomOpIr::new(collective.part_id(), &[tensor.into_ir()], &[output]);
                let operation = MemberPart::<B> {
                    desc: desc.clone(),
                    part,
                    in_flight: InFlight::default(),
                };
                let output = client
                    .register(stream, OperationIr::Custom(desc), operation)
                    .pop()
                    .expect("a part has one output");
                // A stream holds an operation back for one queued after it to fuse with, and the
                // other members wait on this one until it runs.
                client.execute_queued(stream);
                output
            })
            .collect())
    }
}

/// One member's part of a collective, as an operation on the member's stream.
struct MemberPart<B: FusionBackend> {
    desc: CustomOpIr,
    part: Box<dyn CollectivePart<B>>,
    in_flight: InFlight,
}

impl<B: FusionBackend> core::fmt::Debug for MemberPart<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemberPart")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl<B: FusionBackend> Operation<B::FusionRuntime> for MemberPart<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        let _reached = self.in_flight.leave_on_drop();
        let ([input], [output]) = self.desc.as_fixed();
        let tensor = handles.get_tensor_handle(input);
        handles.register_handle(output.id, self.part.run(tensor));
        Ok(())
    }

    fn skipped(&self) {
        let _reached = self.in_flight.leave_on_drop();
        self.part.stand_in();
    }
}

/// The collective parts queued on their members' streams that have not reached their devices yet.
///
/// A transfer between devices first waits until none is left. Setting a transfer up can wait on a
/// device whose part already runs, and so waits in the collective for a peer whose own part may
/// still be queued behind the transfer; with every part on its device first, the collective
/// completes, then the transfer.
pub(crate) struct PartsInFlight;

static IN_FLIGHT: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());

impl PartsInFlight {
    fn enter(parts: usize) {
        *IN_FLIGHT
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) += parts;
    }

    fn leave() {
        let mut parts = IN_FLIGHT
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *parts -= 1;
        if *parts == 0 {
            IN_FLIGHT.1.notify_all();
        }
    }

    /// Waits until every queued part has reached its device. Called off the members' streams,
    /// whose own threads run the parts.
    pub(crate) fn wait_until_none() {
        let parts = IN_FLIGHT
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _none = IN_FLIGHT
            .1
            .wait_while(parts, |parts| *parts > 0)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
}

/// One part's place in [`PartsInFlight`], left once: when the part reached its device, or else when
/// the operation is dropped without running.
#[derive(Default)]
struct InFlight {
    left: AtomicBool,
}

impl InFlight {
    fn leave(&self) {
        if !self.left.swap(true, Ordering::AcqRel) {
            PartsInFlight::leave();
        }
    }

    /// Leaves when the returned guard drops, however the part's queueing ends.
    fn leave_on_drop(&self) -> impl Drop + '_ {
        struct Leave<'a>(&'a InFlight);
        impl Drop for Leave<'_> {
            fn drop(&mut self) {
                self.0.leave();
            }
        }
        Leave(self)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.leave();
    }
}
