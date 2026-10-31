// Shared CPU fixtures for the retained CTC benchmarks and measurement report.

use super::{Flex, FlexTensor, FloatDType};
use burn_backend::{
    TensorData,
    ops::{ActivationOps, FloatTensorOps},
};

#[derive(Clone, Copy)]
pub(super) struct Scenario {
    pub id: &'static str,
    pub name: &'static str,
    /// Logical score dimensions followed by stored target width: [T, B, C, S].
    pub shape: [usize; 4],
    profile: Profile,
}

#[derive(Clone, Copy)]
enum Profile {
    Standard,
    MixedLengths,
    RepeatedLabels,
    StridedScores,
    Unreachable,
    PaddedTargets,
    NanFallback,
}

impl Scenario {
    const fn new(
        id: &'static str,
        name: &'static str,
        shape: [usize; 4],
        profile: Profile,
    ) -> Self {
        Self {
            id,
            name,
            shape,
            profile,
        }
    }

    pub fn fixture(&self, float_dtype: FloatDType) -> ([FlexTensor; 4], usize) {
        let [time, batch, classes, width] = self.shape;
        let blank = if matches!(self.profile, Profile::RepeatedLabels) {
            2
        } else {
            0
        };
        let score_shape = if matches!(self.profile, Profile::StridedScores) {
            [batch, time + 2, classes]
        } else {
            [time, batch, classes]
        };
        let score_elements = score_shape.iter().product();
        let logits: Vec<f64> = if matches!(self.profile, Profile::NanFallback) {
            vec![f64::NAN; score_elements]
        } else {
            (0..score_elements)
                .map(|i| (i % 997) as f64 / 997.0 - 0.5)
                .collect()
        };
        let logits = Flex::float_cast(
            FlexTensor::from_data(TensorData::new(logits, score_shape)),
            float_dtype,
        );
        let scores = if matches!(self.profile, Profile::NanFallback) {
            logits
        } else {
            Flex::log_softmax(logits, 2)
        };
        let scores = if matches!(self.profile, Profile::StridedScores) {
            let layout = scores.layout().transpose(0, 1).narrow(0, 1, time);
            scores.with_layout(layout)
        } else {
            scores
        };

        let mut labels: Vec<i32> = (0..batch * width)
            .map(|i| {
                if matches!(self.profile, Profile::RepeatedLabels) {
                    let label = (i / 2) % (classes - 1);
                    (label + usize::from(label >= blank)) as i32
                } else {
                    (i % (classes - 1) + 1) as i32
                }
            })
            .collect();
        let mut input_lengths = vec![time as i32; batch];
        let mut target_lengths = vec![width as i32; batch];
        match self.profile {
            Profile::MixedLengths => {
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
            Profile::Unreachable => labels.fill(1),
            Profile::PaddedTargets => target_lengths.fill(4),
            _ => {}
        }
        (
            [
                scores,
                FlexTensor::from_data(TensorData::new(labels, [batch, width])),
                FlexTensor::from_data(TensorData::new(input_lengths, [batch])),
                FlexTensor::from_data(TensorData::new(target_lengths, [batch])),
            ],
            blank,
        )
    }
}

pub(super) const SCENARIOS: [Scenario; 16] = [
    Scenario::new("S01", "Tiny call", [1, 1, 3, 0], Profile::Standard),
    Scenario::new("S02", "Short sequence", [16, 1, 5, 4], Profile::Standard),
    Scenario::new(
        "S03",
        "Empty target batch",
        [128, 8, 32, 0],
        Profile::Standard,
    ),
    Scenario::new("S04", "Medium batch", [128, 8, 32, 32], Profile::Standard),
    Scenario::new(
        "S05",
        "Representative batch",
        [256, 16, 80, 64],
        Profile::Standard,
    ),
    Scenario::new("S06", "Long sequence", [512, 16, 32, 64], Profile::Standard),
    Scenario::new("S07", "Long targets", [256, 8, 32, 128], Profile::Standard),
    Scenario::new(
        "S08",
        "Long sequence, short targets",
        [512, 16, 32, 4],
        Profile::Standard,
    ),
    Scenario::new("S09", "Single sample", [256, 1, 80, 64], Profile::Standard),
    Scenario::new(
        "S10",
        "Wide vocabulary",
        [128, 4, 1024, 32],
        Profile::Standard,
    ),
    Scenario::new(
        "S11",
        "Mixed input and target lengths",
        [256, 16, 80, 64],
        Profile::MixedLengths,
    ),
    Scenario::new(
        "S12",
        "Repeated labels, blank 2",
        [128, 8, 32, 32],
        Profile::RepeatedLabels,
    ),
    Scenario::new(
        "S13",
        "Transposed and narrowed scores",
        [128, 8, 32, 32],
        Profile::StridedScores,
    ),
    Scenario::new(
        "S14",
        "Unreachable repeated target",
        [2, 4, 3, 2],
        Profile::Unreachable,
    ),
    Scenario::new(
        "S15",
        "Heavily padded targets",
        [256, 16, 80, 128],
        Profile::PaddedTargets,
    ),
    Scenario::new(
        "S16",
        "NaN fallback control",
        [16, 1, 5, 4],
        Profile::NanFallback,
    ),
];
