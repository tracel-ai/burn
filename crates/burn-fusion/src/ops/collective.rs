use burn_backend::{
    DType, Shape, TensorMetadata,
    distributed::ReduceOperation,
    ops::{CollectiveOps, ComposedCollectives, ComposedReductions},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_ir::TensorError;

use crate::{
    Client, Fusion, FusionBackend, FusionRuntime, FusionTensor, get_client, stream::StreamId,
};

impl<B: FusionBackend> CollectiveOps<Self> for Fusion<B> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerKind::<B, _>::float().run(
            tensors,
            |inner| B::float_all_reduce(inner, op),
            |tensors| ComposedReductions::<Self, _>::float().all_reduce(tensors, op),
        )
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        InnerKind::<B, _>::float().run(
            tensors,
            |inner| B::float_all_gather(inner, dim),
            |tensors| ComposedCollectives::<Self, _>::float().all_gather(tensors, dim),
        )
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        InnerKind::<B, _>::float().run(
            tensors,
            |inner| B::float_reduce_scatter(inner, dim, op),
            |tensors| ComposedReductions::<Self, _>::float().reduce_scatter(tensors, dim, op),
        )
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        InnerKind::<B, _>::float().run(
            tensors,
            |inner| B::float_all_to_all(inner, split_dim, concat_dim),
            |tensors| {
                ComposedCollectives::<Self, _>::float().all_to_all(tensors, split_dim, concat_dim)
            },
        )
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        InnerKind::<B, _>::int().run(
            tensors,
            |inner| B::int_all_reduce(inner, op),
            |tensors| ComposedReductions::<Self, _>::int().all_reduce(tensors, op),
        )
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        InnerKind::<B, _>::int().run(
            tensors,
            |inner| B::int_all_gather(inner, dim),
            |tensors| ComposedCollectives::<Self, _>::int().all_gather(tensors, dim),
        )
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        InnerKind::<B, _>::int().run(
            tensors,
            |inner| B::int_reduce_scatter(inner, dim, op),
            |tensors| ComposedReductions::<Self, _>::int().reduce_scatter(tensors, dim, op),
        )
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        InnerKind::<B, _>::int().run(
            tensors,
            |inner| B::int_all_to_all(inner, split_dim, concat_dim),
            |tensors| {
                ComposedCollectives::<Self, _>::int().all_to_all(tensors, split_dim, concat_dim)
            },
        )
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        InnerKind::<B, _>::bool().run(
            tensors,
            |inner| B::bool_all_gather(inner, dim),
            |tensors| ComposedCollectives::<Self, _>::bool().all_gather(tensors, dim),
        )
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        InnerKind::<B, _>::bool().run(
            tensors,
            |inner| B::bool_all_to_all(inner, split_dim, concat_dim),
            |tensors| {
                ComposedCollectives::<Self, _>::bool().all_to_all(tensors, split_dim, concat_dim)
            },
        )
    }
}

/// How one tensor kind passes between Fusion's tensors and its inner backend's, so a collective
/// runs on the inner backend, whose transport fusion would otherwise hide behind transfers.
struct InnerKind<B: FusionBackend, T> {
    resolve: fn(FusionTensor<B::FusionRuntime>) -> Result<T, TensorError>,
    handle: fn(T) -> B::Handle,
}

impl<B: FusionBackend> InnerKind<B, FloatTensor<B>> {
    fn float() -> Self {
        Self {
            resolve: |tensor| tensor.client.clone().resolve_carrying_float::<B>(tensor),
            handle: B::float_tensor_handle,
        }
    }
}

impl<B: FusionBackend> InnerKind<B, IntTensor<B>> {
    fn int() -> Self {
        Self {
            resolve: |tensor| tensor.client.clone().resolve_carrying_int::<B>(tensor),
            handle: B::int_tensor_handle,
        }
    }
}

impl<B: FusionBackend> InnerKind<B, BoolTensor<B>> {
    fn bool() -> Self {
        Self {
            resolve: |tensor| tensor.client.clone().resolve_carrying_bool::<B>(tensor),
            handle: B::bool_tensor_handle,
        }
    }
}

impl<B: FusionBackend, T: TensorMetadata<Device = Device<B>>> InnerKind<B, T> {
    /// Runs `inner` on what every member's stream resolves to. When a member's tensor carries a
    /// failure, `composed` runs on Fusion's own tensors instead, so every output carries it.
    fn run(
        &self,
        tensors: Vec<FusionTensor<B::FusionRuntime>>,
        inner: impl FnOnce(Vec<T>) -> Vec<T>,
        composed: impl FnOnce(
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
                ((self.resolve)(tensor), metadata)
            })
            .collect();
        if resolved.iter().all(|(resolved, _)| resolved.is_ok()) {
            let inner_tensors = resolved
                .into_iter()
                .map(|(resolved, _)| resolved.expect("every member resolved"))
                .collect();
            return inner(inner_tensors)
                .into_iter()
                .map(|tensor| self.registered(tensor))
                .collect();
        }
        let tensors = resolved
            .into_iter()
            .map(|(resolved, metadata)| match resolved {
                Ok(tensor) => self.registered(tensor),
                Err(error) => metadata.carrying(error),
            })
            .collect();
        composed(tensors)
    }

    fn registered(&self, tensor: T) -> FusionTensor<B::FusionRuntime> {
        let client = get_client::<B>(&tensor.device());
        let (shape, dtype) = (tensor.shape(), tensor.dtype());
        let id = client.register_tensor_handle((self.handle)(tensor));
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
