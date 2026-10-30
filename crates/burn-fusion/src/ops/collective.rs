//! A member is one of the tensors a collective is given, as
//! [`CollectiveOps`](burn_backend::ops::CollectiveOps) defines it: member `i` is position `i` of
//! the input, on its own device, and receives position `i` of the output.

use burn_backend::{
    DType, Shape, TensorMetadata,
    distributed::ReduceOperation,
    ops::{CollectiveOps, EmulatedCollectives},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_ir::TensorError;
use core::marker::PhantomData;

use crate::{
    Client, Fusion, FusionBackend, FusionRuntime, FusionTensor, client::Resolving, get_client,
    stream::StreamId,
};

impl<B: FusionBackend> CollectiveOps<Self> for Fusion<B> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
            |inner| B::float_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self>::float_all_reduce(tensors, op),
        )
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B>::float(
            tensors,
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
            |inner| B::float_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::float_all_to_all(tensors, split_dim, concat_dim),
        )
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
            |inner| B::int_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self>::int_all_reduce(tensors, op),
        )
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B>::int(
            tensors,
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
            |inner| B::int_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::int_all_to_all(tensors, split_dim, concat_dim),
        )
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        InnerCollectives::<B>::bool(
            tensors,
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
            |inner| B::bool_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self>::bool_all_to_all(tensors, split_dim, concat_dim),
        )
    }
}

/// The collectives run on Fusion's inner backend, whose native collective Fusion would otherwise
/// hide behind transfers: each member resolved from Fusion's tensors, the outputs registered back.
struct InnerCollectives<B> {
    _backend: PhantomData<B>,
}

impl<B: FusionBackend> InnerCollectives<B> {
    fn float<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<FloatTensor<B>>) -> Vec<FloatTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            |tensor| tensor.client.clone().resolve_carrying_float::<B>(tensor),
            B::float_tensor_handle,
            inner,
            emulated,
        )
    }

    fn int<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<IntTensor<B>>) -> Vec<IntTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            |tensor| tensor.client.clone().resolve_carrying_int::<B>(tensor),
            B::int_tensor_handle,
            inner,
            emulated,
        )
    }

    fn bool<I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        I: FnOnce(Vec<BoolTensor<B>>) -> Vec<BoolTensor<B>>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
        Self::run(
            tensors,
            |tensor| tensor.client.clone().resolve_carrying_bool::<B>(tensor),
            B::bool_tensor_handle,
            inner,
            emulated,
        )
    }

    /// Runs `inner` on what every member's stream resolves to, every member's server resolving at
    /// once. When a member's tensor carries a failure, `emulated` runs on Fusion's own tensors
    /// instead, so every output carries it.
    fn run<T, R, H, I, E>(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        resolve: R,
        handle: H,
        inner: I,
        emulated: E,
    ) -> Vec<FusionTensor<B::FusionRuntime>>
    where
        T: TensorMetadata<Device = Device<B>>,
        R: Fn(FusionTensor<B::FusionRuntime>) -> Resolving<T>,
        H: Fn(T) -> B::Handle,
        I: FnOnce(Vec<T>) -> Vec<T>,
        E: FnOnce(Vec<FusionTensor<B::FusionRuntime>>) -> Vec<FusionTensor<B::FusionRuntime>>,
    {
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
            .map(|(resolving, metadata)| (resolving.wait(), metadata))
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
