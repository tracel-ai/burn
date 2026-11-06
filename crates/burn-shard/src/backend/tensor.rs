use burn_backend::{
    ExecutionError, Shape, Slice, TensorData, TensorMetadata, distributed::ReduceOperation,
    ops::DimSplit,
};
use burn_ir::{BackendIr, HandleKind};
use burn_std::future::{DynFut, block_on};

use crate::{Resharding, Sharding, ShardingError};

/// One tensor as the backend holds it over the sharded compute device: its shard on every member, in member order.
pub enum ShardedTensor<B: BackendIr> {
    Float(Vec<B::FloatTensorPrimitive>),
    Int(Vec<B::IntTensorPrimitive>),
    Bool(Vec<B::BoolTensorPrimitive>),
    Quantized(Vec<B::QuantizedTensorPrimitive>),
}

impl<B: BackendIr> ShardedTensor<B> {
    /// # Panics
    ///
    /// When `handles` is empty, or when they are not all of one kind.
    pub fn new(handles: Vec<HandleKind<B>>) -> Self {
        let mut handles = handles.into_iter();
        let mut sharded = match handles
            .next()
            .expect("A sharded compute device has a member")
        {
            HandleKind::Float(tensor) => Self::Float(vec![tensor]),
            HandleKind::Int(tensor) => Self::Int(vec![tensor]),
            HandleKind::Bool(tensor) => Self::Bool(vec![tensor]),
            HandleKind::Quantized(tensor) => Self::Quantized(vec![tensor]),
        };
        handles.for_each(|handle| sharded.push(handle));
        sharded
    }

    /// `handle` on each of `devices`, one member each, through the backend's broadcast.
    pub fn replicated(handle: HandleKind<B>, devices: &[B::Device]) -> Self {
        match handle {
            HandleKind::Float(tensor) => Self::Float(B::float_broadcast(tensor, devices)),
            HandleKind::Int(tensor) => Self::Int(B::int_broadcast(tensor, devices)),
            HandleKind::Bool(tensor) => Self::Bool(B::bool_broadcast(tensor, devices)),
            // A quantized tensor has no broadcast, and a sharded compute device only ever replicates one.
            HandleKind::Quantized(tensor) => Self::Quantized(
                devices
                    .iter()
                    .map(|device| B::q_to_device(tensor.clone(), device))
                    .collect(),
            ),
        }
    }

    pub fn into_handles(self) -> Vec<HandleKind<B>> {
        match self {
            Self::Float(shards) => shards.into_iter().map(HandleKind::Float).collect(),
            Self::Int(shards) => shards.into_iter().map(HandleKind::Int).collect(),
            Self::Bool(shards) => shards.into_iter().map(HandleKind::Bool).collect(),
            Self::Quantized(shards) => shards.into_iter().map(HandleKind::Quantized).collect(),
        }
    }

    /// The shards once resharded, through the backend's collectives: native between its own
    /// devices where it has them, composed from transfers where it does not.
    ///
    /// # Errors
    ///
    /// When quantized shards joined on the host differ in a per-tensor or global scale.
    pub fn reshard(self, resharding: Resharding, shape: &Shape) -> Result<Self, ShardingError> {
        Ok(match resharding {
            Resharding::Keep => self,
            Resharding::Slice { dim } => self.slice(dim, shape),
            Resharding::AllGather { dim } => self.all_gather(dim)?,
            Resharding::AllReduce => self.all_reduce(),
            Resharding::ReduceScatter { dim } => self.reduce_scatter(dim),
            Resharding::AllToAll { from_dim, to_dim } => self.all_to_all(from_dim, to_dim)?,
        })
    }

    /// The whole value on `device`, read without changing the shards of any other member.
    pub fn into_data(
        self,
        sharding: Sharding,
        device: &B::Device,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        match (self, sharding) {
            (Self::Quantized(shards), Sharding::Split { dim }) => {
                let reads: Vec<_> = shards.into_iter().map(B::q_into_data).collect();
                Box::pin(async move {
                    let mut chunks = Vec::with_capacity(reads.len());
                    for read in reads {
                        chunks.push(read.await?);
                    }
                    QuantizedOnHost::<B>::joined(&chunks, dim)
                        .map_err(|error| ExecutionError::with_context(error.to_string()))
                })
            }
            (shards, sharding) => read(shards.whole_on(device, sharding)),
        }
    }

    /// The whole value as one tensor on `device`.
    fn whole_on(self, device: &B::Device, sharding: Sharding) -> HandleKind<B> {
        // Every member holds the whole of a replicated tensor, so only the first one's moves.
        let shards = match sharding {
            Sharding::Replicated => self.first_member_only(),
            Sharding::Split { .. } | Sharding::PartialSum => self,
        };
        shards.moved_to(device).assembled(sharding)
    }

    fn first_member_only(self) -> Self {
        match self {
            Self::Float(shards) => Self::Float(first(shards)),
            Self::Int(shards) => Self::Int(first(shards)),
            Self::Bool(shards) => Self::Bool(first(shards)),
            Self::Quantized(shards) => Self::Quantized(first(shards)),
        }
    }

    fn moved_to(self, device: &B::Device) -> Self {
        match self {
            Self::Float(shards) => Self::Float(each(shards, |t| B::float_to_device(t, device))),
            Self::Int(shards) => Self::Int(each(shards, |t| B::int_to_device(t, device))),
            Self::Bool(shards) => Self::Bool(each(shards, |t| B::bool_to_device(t, device))),
            Self::Quantized(shards) => Self::Quantized(each(shards, |t| B::q_to_device(t, device))),
        }
    }

    /// The one tensor the shards make up as `sharding` says, once they all sit on one device:
    /// any one of them when replicated, their concatenation when split, their sum when partial.
    fn assembled(self, sharding: Sharding) -> HandleKind<B> {
        match (self, sharding) {
            (shards, Sharding::Replicated) => shards
                .into_handles()
                .into_iter()
                .next()
                .expect("A sharded compute device has a member"),
            (Self::Float(shards), Sharding::Split { dim }) => {
                HandleKind::Float(B::float_cat(shards, dim))
            }
            (Self::Int(shards), Sharding::Split { dim }) => {
                HandleKind::Int(B::int_cat(shards, dim))
            }
            (Self::Bool(shards), Sharding::Split { dim }) => {
                HandleKind::Bool(B::bool_cat(shards, dim))
            }
            (Self::Float(shards), Sharding::PartialSum) => {
                HandleKind::Float(sum(shards, B::float_add))
            }
            (Self::Int(shards), Sharding::PartialSum) => HandleKind::Int(sum(shards, B::int_add)),
            (Self::Quantized(_), Sharding::Split { .. }) => {
                unreachable!("A split quantized tensor is joined on the host")
            }
            (Self::Bool(_) | Self::Quantized(_), Sharding::PartialSum) => not_a_partial_sum(),
        }
    }

    fn push(&mut self, handle: HandleKind<B>) {
        match (self, handle) {
            (Self::Float(shards), HandleKind::Float(tensor)) => shards.push(tensor),
            (Self::Int(shards), HandleKind::Int(tensor)) => shards.push(tensor),
            (Self::Bool(shards), HandleKind::Bool(tensor)) => shards.push(tensor),
            (Self::Quantized(shards), HandleKind::Quantized(tensor)) => shards.push(tensor),
            _ => panic!("The shards of one tensor share a kind"),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Float(shards) => shards.len(),
            Self::Int(shards) => shards.len(),
            Self::Bool(shards) => shards.len(),
            Self::Quantized(shards) => shards.len(),
        }
    }

    /// Each member's chunk of `dim`, the member keeping only its own.
    fn slice(self, dim: usize, shape: &Shape) -> Self {
        let split = DimSplit {
            dim,
            len: shape[dim],
            members: self.len(),
        };
        let chunk = |member| split.slices(member);
        match self {
            Self::Float(shards) => Self::Float(each_member(shards, |t, m| {
                owned_float_slice::<B>(t, &chunk(m))
            })),
            Self::Int(shards) => Self::Int(each_member(shards, |t, m| {
                owned_int_slice::<B>(t, &chunk(m))
            })),
            Self::Bool(shards) => Self::Bool(each_member(shards, |t, m| {
                owned_bool_slice::<B>(t, &chunk(m))
            })),
            Self::Quantized(shards) => Self::Quantized(
                QuantizedOnHost::<B>::from_replicated(shards).split_across_members(dim),
            ),
        }
    }

    fn all_gather(self, dim: usize) -> Result<Self, ShardingError> {
        Ok(match self {
            Self::Float(shards) => Self::Float(B::float_all_gather(shards, dim)),
            Self::Int(shards) => Self::Int(B::int_all_gather(shards, dim)),
            Self::Bool(shards) => Self::Bool(B::bool_all_gather(shards, dim)),
            Self::Quantized(shards) => Self::Quantized(
                QuantizedOnHost::<B>::from_split(shards, dim)?.replicated_across_members(),
            ),
        })
    }

    fn all_to_all(self, from_dim: usize, to_dim: usize) -> Result<Self, ShardingError> {
        // A member keeps its own chunk of `to_dim`, so that is the dim the shards split on, and
        // what it receives from every member makes up the whole of `from_dim`.
        let split_dim = to_dim;
        let concat_dim = from_dim;
        Ok(match self {
            Self::Float(shards) => Self::Float(B::float_all_to_all(shards, split_dim, concat_dim)),
            Self::Int(shards) => Self::Int(B::int_all_to_all(shards, split_dim, concat_dim)),
            Self::Bool(shards) => Self::Bool(B::bool_all_to_all(shards, split_dim, concat_dim)),
            Self::Quantized(shards) => Self::Quantized(
                QuantizedOnHost::<B>::from_split(shards, concat_dim)?
                    .split_across_members(split_dim),
            ),
        })
    }

    fn all_reduce(self) -> Self {
        match self {
            Self::Float(shards) => Self::Float(B::float_all_reduce(shards, ReduceOperation::Sum)),
            Self::Int(shards) => Self::Int(B::int_all_reduce(shards, ReduceOperation::Sum)),
            Self::Bool(_) | Self::Quantized(_) => not_a_partial_sum(),
        }
    }

    fn reduce_scatter(self, dim: usize) -> Self {
        match self {
            Self::Float(shards) => {
                Self::Float(B::float_reduce_scatter(shards, dim, ReduceOperation::Sum))
            }
            Self::Int(shards) => {
                Self::Int(B::int_reduce_scatter(shards, dim, ReduceOperation::Sum))
            }
            Self::Bool(_) | Self::Quantized(_) => not_a_partial_sum(),
        }
    }
}

/// A quantized tensor's value read to the host, where its codes and scales cut and join along a
/// dim as they are: a backend's quantized slice and concatenation requantize.
struct QuantizedOnHost<B: BackendIr> {
    /// Each member's device, in member order.
    devices: Vec<B::Device>,
    whole: TensorData,
}

impl<B: BackendIr> QuantizedOnHost<B> {
    /// Reads a replicated tensor from its first member alone.
    fn from_replicated(shards: Vec<B::QuantizedTensorPrimitive>) -> Self {
        let devices = shards.iter().map(TensorMetadata::device).collect();
        let first = shards
            .into_iter()
            .next()
            .expect("A sharded compute device has a member");
        Self {
            devices,
            whole: Self::read(first),
        }
    }

    /// Reads every member's chunk of a tensor split along `dim` and joins them.
    fn from_split(
        shards: Vec<B::QuantizedTensorPrimitive>,
        dim: usize,
    ) -> Result<Self, ShardingError> {
        let devices = shards.iter().map(TensorMetadata::device).collect();
        let chunks: Vec<_> = shards.into_iter().map(Self::read).collect();
        Ok(Self {
            devices,
            whole: Self::joined(&chunks, dim)?,
        })
    }

    /// The whole value of `chunks`, each member's along `dim`.
    fn joined(chunks: &[TensorData], dim: usize) -> Result<TensorData, ShardingError> {
        let split = DimSplit {
            dim,
            len: chunks.iter().map(|chunk| chunk.shape()[dim]).sum(),
            members: chunks.len(),
        };
        split.whole_data(chunks)
    }

    /// Each member's chunk along `dim`, uploaded to it.
    fn split_across_members(self, dim: usize) -> Vec<B::QuantizedTensorPrimitive> {
        let split = DimSplit {
            dim,
            len: self.whole.shape()[dim],
            members: self.devices.len(),
        };
        split
            .data_chunks(&self.whole)
            .zip(&self.devices)
            .map(|(chunk, device)| B::q_from_data(chunk, device))
            .collect()
    }

    /// The whole value, uploaded to every member.
    fn replicated_across_members(self) -> Vec<B::QuantizedTensorPrimitive> {
        self.devices
            .iter()
            .map(|device| B::q_from_data(self.whole.clone(), device))
            .collect()
    }

    fn read(shard: B::QuantizedTensorPrimitive) -> TensorData {
        block_on(B::q_into_data(shard))
            .unwrap_or_else(|error| panic!("Reading a quantized shard failed: {error}"))
    }
}

fn read<B: BackendIr>(tensor: HandleKind<B>) -> DynFut<Result<TensorData, ExecutionError>> {
    match tensor {
        HandleKind::Float(tensor) => Box::pin(B::float_into_data(tensor)),
        HandleKind::Int(tensor) => Box::pin(B::int_into_data(tensor)),
        HandleKind::Bool(tensor) => Box::pin(B::bool_into_data(tensor)),
        HandleKind::Quantized(tensor) => Box::pin(B::q_into_data(tensor)),
    }
}

/// A slice can be a view that keeps the whole buffer alive, so a shard copies into its own.
fn owned_float_slice<B: BackendIr>(
    tensor: B::FloatTensorPrimitive,
    slices: &[Slice],
) -> B::FloatTensorPrimitive {
    let view = B::float_slice(tensor, slices);
    let owned = B::float_empty(view.shape(), &view.device(), view.dtype().into());
    B::float_slice_assign(owned, &whole(&view.shape()), view)
}

fn owned_int_slice<B: BackendIr>(
    tensor: B::IntTensorPrimitive,
    slices: &[Slice],
) -> B::IntTensorPrimitive {
    let view = B::int_slice(tensor, slices);
    let owned = B::int_empty(view.shape(), &view.device(), view.dtype().into());
    B::int_slice_assign(owned, &whole(&view.shape()), view)
}

fn owned_bool_slice<B: BackendIr>(
    tensor: B::BoolTensorPrimitive,
    slices: &[Slice],
) -> B::BoolTensorPrimitive {
    let view = B::bool_slice(tensor, slices);
    let owned = B::bool_empty(view.shape(), &view.device(), view.dtype().into());
    B::bool_slice_assign(owned, &whole(&view.shape()), view)
}

fn whole(shape: &Shape) -> Vec<Slice> {
    vec![Slice::full(); shape.num_dims()]
}

fn first<T>(mut shards: Vec<T>) -> Vec<T> {
    shards.truncate(1);
    shards
}

fn each<T>(shards: Vec<T>, f: impl FnMut(T) -> T) -> Vec<T> {
    shards.into_iter().map(f).collect()
}

fn each_member<T>(shards: Vec<T>, mut f: impl FnMut(T, usize) -> T) -> Vec<T> {
    shards
        .into_iter()
        .enumerate()
        .map(|(member, shard)| f(shard, member))
        .collect()
}

fn sum<T>(shards: Vec<T>, add: impl FnMut(T, T) -> T) -> T {
    shards
        .into_iter()
        .reduce(add)
        .expect("A sharded compute device has a member")
}

/// No rule makes a bool tensor a partial sum, so one here is a planner bug.
fn not_a_partial_sum() -> ! {
    panic!("Only a float or int tensor can be a partial sum")
}

#[cfg(test)]
mod tests {
    use burn_backend::ops::FloatTensorOps;
    use burn_flex::{Flex, FlexDevice};

    use super::*;

    #[test]
    fn a_split_shard_owns_only_its_own_elements() {
        let devices = [FlexDevice, FlexDevice];
        let shape = Shape::new([4, 6]);
        let whole =
            Flex::float_from_data(TensorData::new(vec![0f32; 24], shape.clone()), &FlexDevice);

        let shards = ShardedTensor::<Flex>::replicated(HandleKind::Float(whole), &devices)
            .reshard(Resharding::Slice { dim: 0 }, &shape)
            .unwrap();

        let ShardedTensor::Float(shards) = shards else {
            unreachable!("A float tensor's shards are float")
        };
        for shard in shards {
            assert_eq!(shard.data_arc().len(), 2 * 6 * size_of::<f32>());
        }
    }
}
