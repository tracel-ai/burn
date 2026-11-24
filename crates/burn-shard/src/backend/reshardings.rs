use std::{cell::OnceCell, collections::HashMap};

use burn_backend::DType;
use burn_ir::{OperationIr, TensorIr};
use burn_std::{distributed::ReshardingTally, quantized_data_len};

use crate::Sharding;

/// Tallies every resharding that sends part of a tensor between members, by the operation that
/// asked for it.
#[derive(Default)]
pub struct ReshardingRecorder {
    /// Arrival and Position: read in the order each tally was first recorded, and bumped at the
    /// position `positions` keeps. Never shrinks, so a tally keeps its position.
    tallies: Vec<ReshardingTally>,
    /// Keyed: where the tally of each operation and pair of shardings sits in `tallies`.
    positions: HashMap<TallyKey, usize>,
}

/// What a tally counts reshardings under.
#[derive(Clone, PartialEq, Eq, Hash)]
struct TallyKey {
    operation: String,
    from: Sharding,
    to: Sharding,
}

/// An op's [name](OperationIr::name), made the first time it is asked for: naming formats the
/// whole op, which an op resharding many inputs would otherwise pay once per input.
pub struct OpName<'a> {
    op: &'a OperationIr,
    name: OnceCell<String>,
}

impl<'a> OpName<'a> {
    pub fn new(op: &'a OperationIr) -> Self {
        Self {
            op,
            name: OnceCell::new(),
        }
    }

    pub fn to_str(&self) -> &str {
        self.name.get_or_init(|| self.op.name())
    }
}

impl ReshardingRecorder {
    /// `op` moved `tensor` from `from` to `to`.
    pub fn record(&mut self, op: &OpName<'_>, tensor: &TensorIr, from: Sharding, to: Sharding) {
        let bytes = stored_bytes(tensor);
        let key = TallyKey {
            operation: op.to_str().to_owned(),
            from,
            to,
        };
        log::debug!(
            "`{}` reshards an input of shape {:?} from {from:?} to {to:?}, {bytes} bytes",
            key.operation,
            tensor.shape,
        );
        let tallies = &mut self.tallies;
        let position = *self.positions.entry(key).or_insert_with_key(|key| {
            tallies.push(ReshardingTally {
                operation: key.operation.clone(),
                from,
                to,
                count: 0,
                bytes: 0,
            });
            tallies.len() - 1
        });
        let tally = &mut tallies[position];
        tally.count += 1;
        tally.bytes += bytes;
    }

    /// Every tally, in the order each was first recorded.
    pub fn tallies(&self) -> &[ReshardingTally] {
        &self.tallies
    }
}

/// The size of the whole tensor as stored: a quantized tensor's packed values and its scales.
fn stored_bytes(tensor: &TensorIr) -> u64 {
    let bytes = match tensor.dtype {
        DType::QFloat(scheme) => quantized_data_len(&scheme, &tensor.shape),
        dtype => tensor.shape.num_elements() * dtype.size(),
    };
    bytes as u64
}

#[cfg(test)]
mod tests {
    use burn_ir::{TensorId, TensorStatus, UnaryOpIr};
    use burn_std::{QuantScheme, QuantStore, QuantValue, ScaleDtype, Shape};

    use super::*;

    fn tensor(shape: [usize; 2], dtype: DType) -> TensorIr {
        TensorIr {
            id: TensorId::new(0),
            shape: Shape::new(shape),
            status: TensorStatus::ReadOnly,
            dtype,
        }
    }

    fn exp(input: TensorIr) -> OperationIr {
        let out = input.clone();
        OperationIr::Float(
            DType::F32,
            burn_ir::FloatOperationIr::Exp(UnaryOpIr { input, out }),
        )
    }

    #[test]
    fn reshardings_of_one_op_and_shardings_add_up_in_one_tally() {
        let input = tensor([4, 8], DType::F32);
        let split = Sharding::Split { dim: 0 };
        let mut recorder = ReshardingRecorder::default();

        recorder.record(
            &OpName::new(&exp(input.clone())),
            &input,
            split,
            Sharding::Replicated,
        );
        recorder.record(
            &OpName::new(&exp(input.clone())),
            &input,
            Sharding::PartialSum,
            split,
        );
        recorder.record(
            &OpName::new(&exp(input.clone())),
            &input,
            split,
            Sharding::Replicated,
        );

        let tallies: Vec<_> = recorder
            .tallies()
            .iter()
            .map(|tally| {
                (
                    tally.operation.as_str(),
                    tally.from,
                    tally.count,
                    tally.bytes,
                )
            })
            .collect();
        assert_eq!(
            tallies,
            [
                ("Float::Exp", split, 2, 2 * 4 * 8 * 4),
                ("Float::Exp", Sharding::PartialSum, 1, 4 * 8 * 4),
            ]
        );
    }

    #[test]
    fn a_quantized_tensor_counts_its_packed_words_and_its_scale() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(0))
            .per_tensor(ScaleDtype::F32);
        let input = tensor([2, 3], DType::QFloat(scheme));
        let mut recorder = ReshardingRecorder::default();

        recorder.record(
            &OpName::new(&exp(input.clone())),
            &input,
            Sharding::Split { dim: 0 },
            Sharding::Replicated,
        );

        // Each row of three 8-bit values fills one 4-byte word, then one f32 scale.
        assert_eq!(recorder.tallies()[0].bytes, 2 * 4 + 4);
    }
}
