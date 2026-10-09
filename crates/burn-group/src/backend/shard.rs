use std::ops::Range;

use burn_backend::{ExecutionError, Shape, Slice, TensorData};
use burn_ir::{BackendIr, HandleKind};
use burn_std::future::DynFut;

use crate::{Chunks, Placement, Redistribution};

/// The shard of one tensor on every rank, in rank order, as the backend holds them.
pub struct ShardList<B: BackendIr> {
    shards: Vec<Shard<B>>,
}

impl<B: BackendIr> ShardList<B> {
    pub fn new(shards: Vec<HandleKind<B>>) -> Self {
        Self {
            shards: shards.into_iter().map(Shard).collect(),
        }
    }

    /// `handle` copied onto each of `devices`, one rank each.
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
    pub fn redistribute(
        self,
        redistribution: Redistribution,
        shape: &Shape,
        devices: &[B::Device],
    ) -> Self {
        match redistribution {
            Redistribution::Keep => self,
            Redistribution::Slice { dim } => self.slice(dim, shape[dim]),
            Redistribution::AllGather { dim } => self.all_gather(dim, devices),
            Redistribution::AllReduce => self.all_reduce(devices),
            Redistribution::ReduceScatter { dim } => {
                self.all_reduce(devices).slice(dim, shape[dim])
            }
            Redistribution::AllToAll { from_dim, to_dim } => self
                .all_gather(from_dim, devices)
                .slice(to_dim, shape[to_dim]),
        }
    }

    /// The whole value on `device`, read without changing the shards of any other rank.
    pub fn into_data(
        self,
        placement: Placement,
        device: &B::Device,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        let moved = self.shards.into_iter().map(|shard| shard.moved_to(device));
        let whole = match placement {
            Placement::Replicated => moved.take(1).next().expect("A group has a rank"),
            Placement::Sharded { dim } => Shard::concat(moved.collect(), dim),
            Placement::Partial => Shard::sum(moved.collect()),
        };
        whole.into_data()
    }

    fn slice(self, dim: usize, len: usize) -> Self {
        let chunks = Chunks::new(len, self.shards.len());
        Self {
            shards: self
                .shards
                .into_iter()
                .zip(chunks.ranges())
                .map(|(shard, range)| shard.slice(dim, range))
                .collect(),
        }
    }

    fn all_gather(self, dim: usize, devices: &[B::Device]) -> Self {
        self.each_rank(devices, |shards| Shard::concat(shards, dim))
    }

    fn all_reduce(self, devices: &[B::Device]) -> Self {
        self.each_rank(devices, Shard::sum)
    }

    /// Every rank combines all the shards, moved onto its own device.
    fn each_rank(self, devices: &[B::Device], combine: impl Fn(Vec<Shard<B>>) -> Shard<B>) -> Self {
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
            HandleKind::Quantized(_) => unsupported(),
        })
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
            Some(HandleKind::Quantized(_)) => unsupported(),
            None => panic!("A group has a rank"),
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
            .expect("A group has a rank")
    }

    fn moved_to(self, device: &B::Device) -> Self {
        Self(match self.0 {
            HandleKind::Float(tensor) => HandleKind::Float(B::float_to_device(tensor, device)),
            HandleKind::Int(tensor) => HandleKind::Int(B::int_to_device(tensor, device)),
            HandleKind::Bool(tensor) => HandleKind::Bool(B::bool_to_device(tensor, device)),
            HandleKind::Quantized(_) => unsupported(),
        })
    }

    fn into_data(self) -> DynFut<Result<TensorData, ExecutionError>> {
        match self.0 {
            HandleKind::Float(tensor) => Box::pin(B::float_into_data(tensor)),
            HandleKind::Int(tensor) => Box::pin(B::int_into_data(tensor)),
            HandleKind::Bool(tensor) => Box::pin(B::bool_into_data(tensor)),
            HandleKind::Quantized(_) => unsupported(),
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
}

fn unsupported() -> ! {
    unimplemented!("Quantized tensors are not split over a device group")
}
