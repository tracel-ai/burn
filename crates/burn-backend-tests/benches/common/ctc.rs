// Representative fixtures shared by the CTC forward and training benchmarks.

use super::{Flex, FlexTensor, FloatDType};
use burn_backend::{Backend, TensorData, ops::FloatTensorOps, tensor::FloatTensor};

#[derive(Clone, Copy)]
pub(super) struct Scenario {
    /// Logical score dimensions followed by stored target width: [T, B, C, S].
    shape: [usize; 4],
    mixed_lengths: bool,
    strided: bool,
}

impl Scenario {
    pub fn fixture(&self, float_dtype: FloatDType) -> [FlexTensor; 4] {
        let [time, batch, classes, width] = self.shape;
        let logits_shape = if self.strided {
            [batch, time + 2, classes]
        } else {
            [time, batch, classes]
        };
        let logits: Vec<f64> = (0..logits_shape.iter().product())
            .map(|i| (i % 997) as f64 / 997.0 - 0.5)
            .collect();
        // Cast before log-softmax so all arithmetic uses the benchmarked dtype.
        let logits = Flex::float_cast(
            FlexTensor::from_data(TensorData::new(logits, logits_shape)),
            float_dtype,
        );
        let labels: Vec<i32> = (0..batch * width)
            .map(|i| (i % (classes - 1) + 1) as i32)
            .collect();
        let mut input_lengths = vec![time as i32; batch];
        let mut target_lengths = vec![width as i32; batch];
        if self.mixed_lengths {
            for (sample, (input, target)) in input_lengths
                .iter_mut()
                .zip(&mut target_lengths)
                .enumerate()
            {
                let (t, s) = match sample % 4 {
                    0 => (time, width),
                    1 => (3 * time / 4, width / 2),
                    2 => (time / 2, 0),
                    _ => (time / 4, width / 4),
                };
                *input = t as i32;
                *target = s as i32;
            }
        }
        [
            logits,
            FlexTensor::from_data(TensorData::new(labels, [batch, width])),
            FlexTensor::from_data(TensorData::new(input_lengths, [batch])),
            FlexTensor::from_data(TensorData::new(target_lengths, [batch])),
        ]
    }

    pub fn log_probs<B: Backend>(&self, logits: FloatTensor<B>) -> FloatTensor<B> {
        let scores = B::log_softmax(logits, 2);
        if self.strided {
            let time = self.shape[0];
            B::float_slice(
                B::float_swap_dims(scores, 0, 1),
                &[(1..time + 1).into(), (..).into(), (..).into()],
            )
        } else {
            scores
        }
    }
}

pub(super) const SHORT: Scenario = Scenario {
    shape: [16, 1, 5, 4],
    mixed_lengths: false,
    strided: false,
};

pub(super) const BATCH: Scenario = Scenario {
    shape: [256, 16, 80, 64],
    mixed_lengths: false,
    strided: false,
};

pub(super) const MIXED_LENGTHS: Scenario = Scenario {
    shape: [256, 16, 80, 64],
    mixed_lengths: true,
    strided: false,
};

pub(super) const STRIDED: Scenario = Scenario {
    shape: [128, 8, 32, 32],
    mixed_lengths: false,
    strided: true,
};
