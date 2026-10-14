use std::ops::Range;

use burn_backend::{ExecutionError, Shape, Slice, TensorData, TensorMetadata};
use burn_ir::{BackendIr, HandleKind};
use burn_std::future::DynFut;

use crate::{DimSplit, Resharding, Sharding};

/// The shard of one tensor on every member, in member order, as the backend holds them.
pub struct ShardList<B: BackendIr> {
    shards: Vec<Shard<B>>,
}

impl<B: BackendIr> ShardList<B> {
    pub fn new(shards: Vec<HandleKind<B>>) -> Self {
        Self {
            shards: shards.into_iter().map(Shard).collect(),
        }
    }

    /// `handle` copied onto each of `devices`, one member each.
    pub fn replicated(handle: HandleKind<B>, devices: &[B::Device]) -> Self {
        let shard = Shard(handle);
        Self {
            shards: devices
                .iter()
                .map(|device| shard.clone().moved_to(device))
                .collect(),
        }
    }

    pub fn into_handles(self) -> impl Iterator<Item = HandleKind<B>> {
        self.shards.into_iter().map(|shard| shard.0)
    }

    /// Collectives emulated by moving shards with `to_device`: correct on any backend, fast on
    /// none.
    pub fn reshard(self, resharding: Resharding, shape: &Shape, devices: &[B::Device]) -> Self {
        match resharding {
            Resharding::Keep => self,
            Resharding::Slice { dim } => self.slice(dim, shape),
            Resharding::AllGather { dim } => self.all_gather(dim, devices),
            Resharding::AllReduce => self.all_reduce(devices),
            Resharding::ReduceScatter { dim } => self.all_reduce(devices).slice(dim, shape),
            Resharding::AllToAll { from_dim, to_dim } => {
                self.all_gather(from_dim, devices).slice(to_dim, shape)
            }
        }
    }

    /// The whole value on `device`, read without changing the shards of any other member.
    pub fn into_data(
        self,
        sharding: Sharding,
        device: &B::Device,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        let mut moved = self.shards.into_iter().map(|shard| shard.moved_to(device));
        let whole = match sharding {
            Sharding::Replicated => moved.next().expect("A group has a member"),
            Sharding::Split { dim } => Shard::concat(moved.collect(), dim),
            Sharding::PartialSum => Shard::sum(moved.collect()),
        };
        whole.into_data()
    }

    fn slice(self, dim: usize, shape: &Shape) -> Self {
        let split = DimSplit {
            len: shape[dim],
            num_members: self.shards.len(),
        };
        Self {
            shards: self
                .shards
                .into_iter()
                .zip(split.ranges())
                .map(|(shard, range)| shard.slice(dim, range))
                .collect(),
        }
    }

    fn all_gather(self, dim: usize, devices: &[B::Device]) -> Self {
        self.combine_on_every_member(devices, |shards| Shard::concat(shards, dim))
    }

    fn all_reduce(self, devices: &[B::Device]) -> Self {
        self.combine_on_every_member(devices, Shard::sum)
    }

    fn combine_on_every_member(
        self,
        devices: &[B::Device],
        combine: impl Fn(Vec<Shard<B>>) -> Shard<B>,
    ) -> Self {
        let shards = devices
            .iter()
            .map(|device| {
                let moved = self
                    .shards
                    .iter()
                    .map(|shard| shard.clone().moved_to(device))
                    .collect();
                combine(moved)
            })
            .collect();
        Self { shards }
    }
}

#[derive(Clone)]
struct Shard<B: BackendIr>(HandleKind<B>);

impl<B: BackendIr> Shard<B> {
    fn slice(self, dim: usize, range: Range<usize>) -> Self {
        let slices: Vec<Slice> = (0..=dim)
            .map(|axis| match axis == dim {
                true => Slice::new(range.start as isize, Some(range.end as isize), 1),
                false => Slice::new(0, None, 1),
            })
            .collect();
        Self(match self.0 {
            HandleKind::Float(tensor) => HandleKind::Float(B::float_slice(tensor, &slices)),
            HandleKind::Int(tensor) => HandleKind::Int(B::int_slice(tensor, &slices)),
            HandleKind::Bool(tensor) => HandleKind::Bool(B::bool_slice(tensor, &slices)),
            HandleKind::Quantized(_) => Self::quantized_unsupported(),
        })
        .owned()
    }

    /// A slice can be a view that keeps the whole buffer alive, so a shard copies into its own.
    fn owned(self) -> Self {
        Self(match self.0 {
            HandleKind::Float(view) => {
                let whole = Self::whole(&view.shape());
                let owned = B::float_empty(view.shape(), &view.device(), view.dtype().into());
                HandleKind::Float(B::float_slice_assign(owned, &whole, view))
            }
            HandleKind::Int(view) => {
                let whole = Self::whole(&view.shape());
                let owned = B::int_empty(view.shape(), &view.device(), view.dtype().into());
                HandleKind::Int(B::int_slice_assign(owned, &whole, view))
            }
            HandleKind::Bool(view) => {
                let whole = Self::whole(&view.shape());
                let owned = B::bool_empty(view.shape(), &view.device(), view.dtype().into());
                HandleKind::Bool(B::bool_slice_assign(owned, &whole, view))
            }
            HandleKind::Quantized(_) => Self::quantized_unsupported(),
        })
    }

    fn whole(shape: &Shape) -> Vec<Slice> {
        vec![Slice::full(); shape.num_dims()]
    }

    fn concat(shards: Vec<Self>, dim: usize) -> Self {
        let mut shards = shards.into_iter().peekable();
        Self(match shards.peek().map(|shard| &shard.0) {
            Some(HandleKind::Float(_)) => {
                HandleKind::Float(B::float_cat(shards.map(Shard::into_float).collect(), dim))
            }
            Some(HandleKind::Int(_)) => {
                HandleKind::Int(B::int_cat(shards.map(Shard::into_int).collect(), dim))
            }
            Some(HandleKind::Bool(_)) => {
                HandleKind::Bool(B::bool_cat(shards.map(Shard::into_bool).collect(), dim))
            }
            Some(HandleKind::Quantized(_)) => Self::quantized_unsupported(),
            None => panic!("A group has a member"),
        })
    }

    fn sum(shards: Vec<Self>) -> Self {
        shards
            .into_iter()
            .reduce(|total, shard| {
                Self(match (total.0, shard.0) {
                    (HandleKind::Float(lhs), HandleKind::Float(rhs)) => {
                        HandleKind::Float(B::float_add(lhs, rhs))
                    }
                    (HandleKind::Int(lhs), HandleKind::Int(rhs)) => {
                        HandleKind::Int(B::int_add(lhs, rhs))
                    }
                    _ => panic!("Only a float or int tensor can be a partial sum"),
                })
            })
            .expect("A group has a member")
    }

    fn moved_to(self, device: &B::Device) -> Self {
        Self(match self.0 {
            HandleKind::Float(tensor) => HandleKind::Float(B::float_to_device(tensor, device)),
            HandleKind::Int(tensor) => HandleKind::Int(B::int_to_device(tensor, device)),
            HandleKind::Bool(tensor) => HandleKind::Bool(B::bool_to_device(tensor, device)),
            HandleKind::Quantized(_) => Self::quantized_unsupported(),
        })
    }

    fn into_data(self) -> DynFut<Result<TensorData, ExecutionError>> {
        match self.0 {
            HandleKind::Float(tensor) => Box::pin(B::float_into_data(tensor)),
            HandleKind::Int(tensor) => Box::pin(B::int_into_data(tensor)),
            HandleKind::Bool(tensor) => Box::pin(B::bool_into_data(tensor)),
            HandleKind::Quantized(_) => Self::quantized_unsupported(),
        }
    }

    fn into_float(self) -> B::FloatTensorPrimitive {
        match self.0 {
            HandleKind::Float(tensor) => tensor,
            _ => panic!("ShardList of one tensor share a kind"),
        }
    }

    fn into_int(self) -> B::IntTensorPrimitive {
        match self.0 {
            HandleKind::Int(tensor) => tensor,
            _ => panic!("ShardList of one tensor share a kind"),
        }
    }

    fn into_bool(self) -> B::BoolTensorPrimitive {
        match self.0 {
            HandleKind::Bool(tensor) => tensor,
            _ => panic!("ShardList of one tensor share a kind"),
        }
    }

    fn quantized_unsupported() -> ! {
        unimplemented!("Quantized tensors are not split over a device group")
    }
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

        let shards = ShardList::<Flex>::replicated(HandleKind::Float(whole), &devices).reshard(
            Resharding::Slice { dim: 0 },
            &shape,
            &devices,
        );

        for handle in shards.into_handles() {
            let HandleKind::Float(shard) = handle else {
                unreachable!("A float tensor's shards are float")
            };
            assert_eq!(shard.data_arc().len(), 2 * 6 * size_of::<f32>());
        }
    }
}
