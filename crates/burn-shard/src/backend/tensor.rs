use burn_backend::{
    ExecutionError, Shape, Slice, TensorData, TensorMetadata, distributed::ReduceOperation,
    ops::DimSplit,
};
use burn_ir::{BackendIr, HandleKind};
use burn_std::future::{DynFut, block_on};

use crate::{Resharding, ShardMember, Sharding, ShardingError};

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
            Resharding::Slice { dim, chunks } => self.slice(dim, chunks, shape),
            Resharding::AllGather { dim, chunks } => self.all_gather(dim, chunks)?,
            Resharding::AllReduce => self.all_reduce(),
            Resharding::ReduceScatter { dim } => self.reduce_scatter(dim),
            Resharding::AllToAll { from_dim, to_dim } => self.all_to_all(from_dim, to_dim)?,
            Resharding::GatherThenSlice {
                from_dim,
                from_chunks,
                to_dim,
                to_chunks,
            } => self
                .all_gather(from_dim, from_chunks)?
                .slice(to_dim, to_chunks, shape),
            Resharding::ReduceThenSlice { dim, chunks } => {
                self.all_reduce().slice(dim, chunks, shape)
            }
        })
    }

    /// The whole value on `device`, read without changing the shards of any other member.
    pub fn into_data(
        self,
        sharding: Sharding,
        device: &B::Device,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        let len = self.len();
        match (self, sharding.split_dim(), sharding.num_chunks(len)) {
            (Self::Quantized(shards), Some(dim), Some(chunks)) => {
                let reads: Vec<_> = one_per_chunk(shards, chunks)
                    .into_iter()
                    .map(B::q_into_data)
                    .collect();
                Box::pin(async move {
                    let mut chunks = Vec::with_capacity(reads.len());
                    for read in reads {
                        chunks.push(read.await?);
                    }
                    QuantizedOnHost::<B>::joined(&chunks, dim)
                        .map_err(|error| ExecutionError::with_context(error.to_string()))
                })
            }
            (shards, ..) => read(shards.whole_on(device, sharding)),
        }
    }

    /// The whole value as one tensor on `device`.
    fn whole_on(self, device: &B::Device, sharding: Sharding) -> HandleKind<B> {
        // Every member of a group holds the same chunk, so only the group's first one moves.
        let (shards, sharding) = match sharding {
            Sharding::Replicated => (self.one_per_chunk(1), sharding),
            Sharding::ReplicatedSplit { dim, groups } => {
                (self.one_per_chunk(groups), Sharding::Split { dim })
            }
            Sharding::Split { .. } | Sharding::PartialSum => (self, sharding),
        };
        shards.moved_to(device).assembled(sharding)
    }

    fn one_per_chunk(self, chunks: usize) -> Self {
        match self {
            Self::Float(shards) => Self::Float(one_per_chunk(shards, chunks)),
            Self::Int(shards) => Self::Int(one_per_chunk(shards, chunks)),
            Self::Bool(shards) => Self::Bool(one_per_chunk(shards, chunks)),
            Self::Quantized(shards) => Self::Quantized(one_per_chunk(shards, chunks)),
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
            (_, Sharding::ReplicatedSplit { .. }) => {
                unreachable!("A replicated split is assembled from one member per group")
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

    /// Each member's chunk of `dim` cut into `chunks`, the member keeping only its own.
    fn slice(self, dim: usize, chunks: usize, shape: &Shape) -> Self {
        let split = DimSplit {
            dim,
            len: shape[dim],
            members: chunks,
        };
        let num_members = self.len();
        let chunk = |index| split.slices(ShardMember { index, num_members }.chunk(chunks));
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
                QuantizedOnHost::<B>::from_replicated(shards).split_across_members(dim, chunks),
            ),
        }
    }

    /// The whole of a split cut into `chunks` along `dim` on every member: through the backend's
    /// all-gather when each member holds its own chunk, and otherwise each member keeping its own
    /// and receiving every other group's from that group's first member.
    fn all_gather(self, dim: usize, chunks: usize) -> Result<Self, ShardingError> {
        if chunks < self.len() {
            return self.gathered_groups(dim, chunks);
        }
        Ok(match self {
            Self::Float(shards) => Self::Float(B::float_all_gather(shards, dim)),
            Self::Int(shards) => Self::Int(B::int_all_gather(shards, dim)),
            Self::Bool(shards) => Self::Bool(B::bool_all_gather(shards, dim)),
            Self::Quantized(shards) => Self::Quantized(
                QuantizedOnHost::<B>::from_split(shards, dim, chunks)?.replicated_across_members(),
            ),
        })
    }

    fn gathered_groups(self, dim: usize, chunks: usize) -> Result<Self, ShardingError> {
        Ok(match self {
            Self::Float(shards) => Self::Float(gathered_groups(
                shards,
                chunks,
                |t, device| B::float_to_device(t, device),
                |ts| B::float_cat(ts, dim),
            )),
            Self::Int(shards) => Self::Int(gathered_groups(
                shards,
                chunks,
                |t, device| B::int_to_device(t, device),
                |ts| B::int_cat(ts, dim),
            )),
            Self::Bool(shards) => Self::Bool(gathered_groups(
                shards,
                chunks,
                |t, device| B::bool_to_device(t, device),
                |ts| B::bool_cat(ts, dim),
            )),
            Self::Quantized(shards) => Self::Quantized(
                QuantizedOnHost::<B>::from_split(shards, dim, chunks)?.replicated_across_members(),
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
            Self::Quantized(shards) => {
                let chunks = shards.len();
                Self::Quantized(
                    QuantizedOnHost::<B>::from_split(shards, concat_dim, chunks)?
                        .split_across_members(split_dim, chunks),
                )
            }
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

    /// Reads the chunks of a tensor split into `chunks` along `dim`, from the first member of
    /// each group, and joins them.
    fn from_split(
        shards: Vec<B::QuantizedTensorPrimitive>,
        dim: usize,
        chunks: usize,
    ) -> Result<Self, ShardingError> {
        let devices = shards.iter().map(TensorMetadata::device).collect();
        let chunks: Vec<_> = one_per_chunk(shards, chunks)
            .into_iter()
            .map(Self::read)
            .collect();
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

    /// Each member's chunk along `dim` cut into `chunks`, uploaded to it.
    fn split_across_members(self, dim: usize, chunks: usize) -> Vec<B::QuantizedTensorPrimitive> {
        let split = DimSplit {
            dim,
            len: self.whole.shape()[dim],
            members: chunks,
        };
        let num_members = self.devices.len();
        let data: Vec<_> = split.data_chunks(&self.whole).collect();
        self.devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let chunk = ShardMember { index, num_members }.chunk(chunks);
                B::q_from_data(data[chunk].clone(), device)
            })
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

/// The first member's shard of each of `chunks` contiguous groups.
fn one_per_chunk<T>(shards: Vec<T>, chunks: usize) -> Vec<T> {
    let num_members = shards.len();
    shards
        .into_iter()
        .enumerate()
        .filter(|(index, _)| {
            ShardMember {
                index: *index,
                num_members,
            }
            .is_first_of_chunk(chunks)
        })
        .map(|(_, shard)| shard)
        .collect()
}

/// The whole on every member, joined by `cat`: its own group's chunk from its own shard, and
/// every other group's from that group's first member, moved there by `to_device`.
fn gathered_groups<T: Clone + TensorMetadata>(
    shards: Vec<T>,
    chunks: usize,
    to_device: impl Fn(T, &T::Device) -> T,
    cat: impl Fn(Vec<T>) -> T,
) -> Vec<T> {
    let num_members = shards.len();
    let firsts = one_per_chunk(shards.clone(), chunks);
    shards
        .into_iter()
        .enumerate()
        .map(|(index, own)| {
            let device = own.device();
            let own_chunk = ShardMember { index, num_members }.chunk(chunks);
            let mut own = Some(own);
            cat(firsts
                .iter()
                .enumerate()
                .map(|(chunk, first)| match chunk == own_chunk {
                    true => own.take().expect("A member holds one chunk"),
                    false => to_device(first.clone(), &device),
                })
                .collect())
        })
        .collect()
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
            .reshard(Resharding::Slice { dim: 0, chunks: 2 }, &shape)
            .unwrap();

        let ShardedTensor::Float(shards) = shards else {
            unreachable!("A float tensor's shards are float")
        };
        for shard in shards {
            assert_eq!(shard.data_arc().len(), 2 * 6 * size_of::<f32>());
        }
    }
}
