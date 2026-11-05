use burn_backend::{DType, Shape, TensorMetadata};
use burn_ir::{BackendIr, HandleKind, TensorId, TensorIr, TensorStatus};
use burn_std::distributed::ShardsError;

use crate::{Resharding, ShardMember, Sharding};

/// The shards a caller made of one tensor, one per member in member order, before they join the
/// sharded compute device.
pub struct MemberShards<B: BackendIr> {
    handles: Vec<HandleKind<B>>,
}

impl<B: BackendIr> MemberShards<B> {
    pub fn new(handles: Vec<HandleKind<B>>) -> Self {
        Self { handles }
    }

    /// The shape of the tensor the shards make up as `sharding` says, or why they do not make one
    /// over the members on `devices`.
    ///
    /// # Errors
    ///
    /// When there is not one shard per member, when a shard is not on its member's device, when
    /// the shards differ in dtype, when a partial sum is not of floats or ints, when the split
    /// cannot be taken, or when a shard's shape is not its member's part.
    pub fn whole_shape(
        &self,
        sharding: Sharding,
        devices: &[B::Device],
    ) -> Result<Shape, ShardsError> {
        let num_members = devices.len();
        if self.handles.len() != num_members {
            return Err(ShardsError::Count {
                shards: self.handles.len(),
                num_members,
            });
        }
        let shards: Vec<ShardMetadata<B::Device>> =
            self.handles.iter().map(ShardMetadata::of).collect();
        if let Some(shard) = shards
            .iter()
            .zip(devices)
            .position(|(shard, device)| shard.device != *device)
        {
            return Err(ShardsError::Device { shard });
        }
        let dtype = shards[0].dtype;
        if let Some((shard, found)) = shards.iter().enumerate().find(|(_, it)| it.dtype != dtype) {
            return Err(ShardsError::DType {
                shard,
                expected: dtype,
                found: found.dtype,
            });
        }
        let whole = Self::whole_of(&shards, sharding, dtype)?;
        for (index, shard) in shards.iter().enumerate() {
            let expected = ShardMember { index, num_members }.local_shape(&whole, sharding);
            if shard.shape != expected {
                return Err(ShardsError::Shape {
                    shard: index,
                    expected,
                    found: shard.shape.clone(),
                });
            }
        }
        Ok(whole)
    }

    pub fn dtype(&self) -> DType {
        ShardMetadata::<B::Device>::of(&self.handles[0]).dtype
    }

    pub fn into_handles(self) -> Vec<HandleKind<B>> {
        self.handles
    }

    /// The whole shape `sharding` reads from the shards, before each is checked against it.
    fn whole_of(
        shards: &[ShardMetadata<B::Device>],
        sharding: Sharding,
        dtype: DType,
    ) -> Result<Shape, ShardsError> {
        let first = &shards[0].shape;
        match sharding {
            Sharding::Replicated => Ok(first.clone()),
            Sharding::PartialSum => match dtype.is_float() || dtype.is_int() || dtype.is_uint() {
                true => Ok(first.clone()),
                false => Err(ShardsError::PartialSumOf { dtype }),
            },
            Sharding::Split { dim } | Sharding::ReplicatedSplit { dim, .. } => {
                // Each group's first member holds its chunk; the shape check covers the others.
                let members_per_chunk = sharding
                    .num_chunks(shards.len())
                    .filter(|chunks| *chunks > 0 && shards.len().is_multiple_of(*chunks))
                    .map_or(1, |chunks| shards.len() / chunks);
                let mut whole = first.clone();
                if dim < whole.num_dims() {
                    whole[dim] = shards
                        .iter()
                        .step_by(members_per_chunk)
                        .map(|shard| shard.shape.get(dim).copied().unwrap_or(0))
                        .sum();
                }
                let tensor = TensorIr {
                    id: TensorId::new(0),
                    shape: whole,
                    status: TensorStatus::ReadOnly,
                    dtype,
                };
                Resharding::of_tensor(Sharding::Replicated, sharding, &tensor, shards.len())?;
                Ok(tensor.shape)
            }
        }
    }
}

/// What the checks read of one shard.
struct ShardMetadata<D> {
    device: D,
    dtype: DType,
    shape: Shape,
}

impl<D> ShardMetadata<D> {
    fn of<B: BackendIr<Device = D>>(handle: &HandleKind<B>) -> Self {
        match handle {
            HandleKind::Float(tensor) => Self::read(tensor),
            HandleKind::Int(tensor) => Self::read(tensor),
            HandleKind::Bool(tensor) => Self::read(tensor),
            HandleKind::Quantized(tensor) => Self::read(tensor),
        }
    }

    fn read(tensor: &impl TensorMetadata<Device = D>) -> Self {
        Self {
            device: tensor.device(),
            dtype: tensor.dtype(),
            shape: tensor.shape(),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_backend::{
        TensorData,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps, QTensorOps},
    };
    use burn_flex::{Flex, FlexDevice};
    use burn_std::{QuantScheme, QuantStore, QuantValue, ScaleDtype, distributed::ShardingError};

    use super::*;

    const MEMBERS: [FlexDevice; 2] = [FlexDevice, FlexDevice];

    fn floats(shapes: [[usize; 2]; 2]) -> MemberShards<Flex> {
        MemberShards::new(
            shapes
                .map(|shape| {
                    let data = TensorData::new(vec![0f32; shape.iter().product()], shape);
                    HandleKind::Float(Flex::float_from_data(data, &FlexDevice))
                })
                .into(),
        )
    }

    #[test]
    fn a_split_takes_its_whole_shape_from_its_shards() {
        let shards = floats([[3, 4], [2, 4]]);

        assert_eq!(
            shards.whole_shape(Sharding::Split { dim: 0 }, &MEMBERS),
            Ok([5, 4].into())
        );
    }

    fn four_floats(shapes: [[usize; 2]; 4]) -> MemberShards<Flex> {
        MemberShards::new(
            shapes
                .map(|shape| {
                    let data = TensorData::new(vec![0f32; shape.iter().product()], shape);
                    HandleKind::Float(Flex::float_from_data(data, &FlexDevice))
                })
                .into(),
        )
    }

    #[test]
    fn a_replicated_split_takes_its_whole_shape_from_one_member_per_group() {
        let shards = four_floats([[2, 4], [2, 4], [1, 4], [1, 4]]);

        assert_eq!(
            shards.whole_shape(
                Sharding::ReplicatedSplit { dim: 0, groups: 2 },
                &[FlexDevice; 4]
            ),
            Ok([3, 4].into())
        );
    }

    #[test]
    fn a_copy_of_another_shape_in_a_group_is_refused() {
        let shards = four_floats([[2, 4], [1, 4], [1, 4], [1, 4]]);

        assert_eq!(
            shards.whole_shape(
                Sharding::ReplicatedSplit { dim: 0, groups: 2 },
                &[FlexDevice; 4]
            ),
            Err(ShardsError::Shape {
                shard: 1,
                expected: [2, 4].into(),
                found: [1, 4].into(),
            })
        );
    }

    #[test]
    fn groups_that_do_not_divide_the_members_are_refused() {
        let shards = four_floats([[2, 4]; 4]);

        assert_eq!(
            shards.whole_shape(
                Sharding::ReplicatedSplit { dim: 0, groups: 3 },
                &[FlexDevice; 4]
            ),
            Err(ShardsError::Sharding(ShardingError::Groups {
                groups: 3,
                num_members: 4,
            }))
        );
    }

    #[test]
    fn a_split_whose_first_member_holds_the_shorter_chunk_is_refused() {
        let shards = floats([[2, 4], [3, 4]]);

        assert_eq!(
            shards.whole_shape(Sharding::Split { dim: 0 }, &MEMBERS),
            Err(ShardsError::Shape {
                shard: 0,
                expected: [3, 4].into(),
                found: [2, 4].into(),
            })
        );
    }

    #[test]
    fn replicated_shards_of_two_shapes_are_refused() {
        let shards = floats([[2, 4], [2, 3]]);

        assert_eq!(
            shards.whole_shape(Sharding::Replicated, &MEMBERS),
            Err(ShardsError::Shape {
                shard: 1,
                expected: [2, 4].into(),
                found: [2, 3].into(),
            })
        );
    }

    #[test]
    fn one_shard_short_of_the_members_is_refused() {
        let shards = MemberShards::<Flex>::new(floats([[2, 4], [2, 4]]).into_handles()[..1].into());

        assert_eq!(
            shards.whole_shape(Sharding::Replicated, &MEMBERS),
            Err(ShardsError::Count {
                shards: 1,
                num_members: 2,
            })
        );
    }

    #[test]
    fn shards_of_two_dtypes_are_refused() {
        let float = Flex::float_from_data(TensorData::new(vec![0f32; 4], [4]), &FlexDevice);
        let int = Flex::int_from_data(TensorData::new(vec![0i64; 4], [4]), &FlexDevice);
        let shards =
            MemberShards::<Flex>::new(vec![HandleKind::Float(float), HandleKind::Int(int)]);

        assert!(matches!(
            shards.whole_shape(Sharding::Replicated, &MEMBERS),
            Err(ShardsError::DType { shard: 1, .. })
        ));
    }

    #[test]
    fn bool_shards_are_not_a_partial_sum() {
        let shards = MemberShards::<Flex>::new(
            MEMBERS
                .map(|device| {
                    let data = TensorData::new(vec![true; 4], [4]);
                    HandleKind::Bool(Flex::bool_from_data(data, &device))
                })
                .into(),
        );

        assert!(matches!(
            shards.whole_shape(Sharding::PartialSum, &MEMBERS),
            Err(ShardsError::PartialSumOf { .. })
        ));
    }

    /// Every chunk is three wide, and the tensor's blocks are two wide, so each boundary between
    /// members cuts a block.
    #[test]
    fn quantized_shards_cut_inside_a_block_of_the_whole_are_refused() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([2], ScaleDtype::F32);
        let members = [FlexDevice; 3];
        let shards = MemberShards::<Flex>::new(
            members
                .map(|device| {
                    let data = TensorData::quantized(vec![1i8; 6], [2, 3], scheme, &[0.1; 4], None);
                    HandleKind::Quantized(Flex::q_from_data(data, &device))
                })
                .into(),
        );

        assert!(matches!(
            shards.whole_shape(Sharding::Split { dim: 1 }, &members),
            Err(ShardsError::Sharding(ShardingError::QuantizedSplit {
                dim: 1,
                ..
            }))
        ));
    }
}
