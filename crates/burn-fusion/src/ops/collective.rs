use burn_backend::{
    DType, Shape, TensorMetadata,
    distributed::ReduceOperation,
    ops::{CollectiveKind, CollectiveOps, EmulatedCollectives},
    tensor::{Bool, BoolTensor, Float, FloatTensor, Int, IntTensor},
};
use burn_ir::TensorError;
use core::marker::PhantomData;

use crate::{
    Client, Fusion, FusionBackend, FusionRuntime, FusionTensor, get_client, stream::StreamId,
};

impl<B: FusionBackend> CollectiveOps<Self> for Fusion<B> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B, Float>::run(
            tensors,
            |inner| B::float_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self, Float>::all_reduce(tensors, op),
        )
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B, Float>::run(
            tensors,
            |inner| B::float_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self, Float>::all_gather(tensors, dim),
        )
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B, Float>::run(
            tensors,
            |inner| B::float_reduce_scatter(inner, dim, op),
            |tensors| EmulatedCollectives::<Self, Float>::reduce_scatter(tensors, dim, op),
        )
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        InnerCollectives::<B, Float>::run(
            tensors,
            |inner| B::float_all_to_all(inner, split_dim, concat_dim),
            |tensors| {
                EmulatedCollectives::<Self, Float>::all_to_all(tensors, split_dim, concat_dim)
            },
        )
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B, Int>::run(
            tensors,
            |inner| B::int_all_reduce(inner, op),
            |tensors| EmulatedCollectives::<Self, Int>::all_reduce(tensors, op),
        )
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B, Int>::run(
            tensors,
            |inner| B::int_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self, Int>::all_gather(tensors, dim),
        )
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B, Int>::run(
            tensors,
            |inner| B::int_reduce_scatter(inner, dim, op),
            |tensors| EmulatedCollectives::<Self, Int>::reduce_scatter(tensors, dim, op),
        )
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        InnerCollectives::<B, Int>::run(
            tensors,
            |inner| B::int_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self, Int>::all_to_all(tensors, split_dim, concat_dim),
        )
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        InnerCollectives::<B, Bool>::run(
            tensors,
            |inner| B::bool_all_gather(inner, dim),
            |tensors| EmulatedCollectives::<Self, Bool>::all_gather(tensors, dim),
        )
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        InnerCollectives::<B, Bool>::run(
            tensors,
            |inner| B::bool_all_to_all(inner, split_dim, concat_dim),
            |tensors| EmulatedCollectives::<Self, Bool>::all_to_all(tensors, split_dim, concat_dim),
        )
    }
}

/// A tensor kind Fusion hands to its inner backend's collectives, whose transport Fusion would
/// otherwise hide behind transfers: resolved from Fusion's tensors, then registered back.
trait ResolveKind<B: FusionBackend>: CollectiveKind<B> {
    fn resolve(tensor: FusionTensor<B::FusionRuntime>) -> Result<Self::Tensor, TensorError>;

    fn handle(tensor: Self::Tensor) -> B::Handle;
}

/// The collectives of tensor kind `K` run on Fusion's inner backend.
struct InnerCollectives<B, K> {
    _kind: PhantomData<(B, K)>,
}

impl<B: FusionBackend> ResolveKind<B> for Float {
    fn resolve(tensor: FusionTensor<B::FusionRuntime>) -> Result<FloatTensor<B>, TensorError> {
        tensor.client.clone().resolve_carrying_float::<B>(tensor)
    }

    fn handle(tensor: FloatTensor<B>) -> B::Handle {
        B::float_tensor_handle(tensor)
    }
}

impl<B: FusionBackend> ResolveKind<B> for Int {
    fn resolve(tensor: FusionTensor<B::FusionRuntime>) -> Result<IntTensor<B>, TensorError> {
        tensor.client.clone().resolve_carrying_int::<B>(tensor)
    }

    fn handle(tensor: IntTensor<B>) -> B::Handle {
        B::int_tensor_handle(tensor)
    }
}

impl<B: FusionBackend> ResolveKind<B> for Bool {
    fn resolve(tensor: FusionTensor<B::FusionRuntime>) -> Result<BoolTensor<B>, TensorError> {
        tensor.client.clone().resolve_carrying_bool::<B>(tensor)
    }

    fn handle(tensor: BoolTensor<B>) -> B::Handle {
        B::bool_tensor_handle(tensor)
    }
}

impl<B: FusionBackend, K: ResolveKind<B>> InnerCollectives<B, K> {
    /// Runs `inner` on what every member's stream resolves to. When a member's tensor carries a
    /// failure, `emulated` runs on Fusion's own tensors instead, so every output carries it.
    fn run(
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        inner: impl FnOnce(Vec<K::Tensor>) -> Vec<K::Tensor>,
        emulated: impl FnOnce(
            Vec<FusionTensor<B::FusionRuntime>>,
        ) -> Vec<FusionTensor<B::FusionRuntime>>,
    ) -> Vec<FusionTensor<B::FusionRuntime>> {
        let resolved: Vec<_> = tensors
            .into_iter()
            .map(|tensor| {
                let metadata = MemberMetadata {
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    client: tensor.client.clone(),
                };
                (K::resolve(tensor), metadata)
            })
            .collect();
        if resolved.iter().all(|(resolved, _)| resolved.is_ok()) {
            let inner_tensors = resolved
                .into_iter()
                .map(|(resolved, _)| resolved.expect("every member resolved"))
                .collect();
            return inner(inner_tensors)
                .into_iter()
                .map(Self::registered)
                .collect();
        }
        let tensors = resolved
            .into_iter()
            .map(|(resolved, metadata)| match resolved {
                Ok(tensor) => Self::registered(tensor),
                Err(error) => metadata.carrying(error),
            })
            .collect();
        emulated(tensors)
    }

    fn registered(tensor: K::Tensor) -> FusionTensor<B::FusionRuntime> {
        let client = get_client::<B>(&tensor.device());
        let (shape, dtype) = (tensor.shape(), tensor.dtype());
        let id = client.register_tensor_handle(K::handle(tensor));
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
