//! Forward-only CTC with rolling state rows. Autodiff uses the default decomposition.

use alloc::vec;
use burn_backend::{DType, Element, TensorMetadata, ops::ctc::ctc_loss_default};
use burn_std::{Bytes, Shape, bf16, f16};
use bytemuck::Pod;
use num_traits::{Float, PrimInt, ToPrimitive};

use crate::{Flex, FlexTensor, Layout};

pub(super) fn ctc_loss(
    log_probs: FlexTensor,
    targets: FlexTensor,
    input_lengths: FlexTensor,
    target_lengths: FlexTensor,
    blank: usize,
) -> FlexTensor {
    try_ctc_loss(&log_probs, &targets, &input_lengths, &target_lengths, blank).unwrap_or_else(
        || ctc_loss_default::<Flex>(log_probs, targets, input_lengths, target_lengths, blank),
    )
}

fn try_ctc_loss(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    blank: usize,
) -> Option<FlexTensor> {
    match log_probs.dtype() {
        DType::F16 => {
            dispatch_indices::<f16>(log_probs, targets, input_lengths, target_lengths, blank)
        }
        DType::BF16 => {
            dispatch_indices::<bf16>(log_probs, targets, input_lengths, target_lengths, blank)
        }
        DType::F32 => {
            dispatch_indices::<f32>(log_probs, targets, input_lengths, target_lengths, blank)
        }
        DType::F64 => {
            dispatch_indices::<f64>(log_probs, targets, input_lengths, target_lengths, blank)
        }
        _ => None,
    }
}

fn dispatch_indices<F: Float + Element + Pod>(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    blank: usize,
) -> Option<FlexTensor> {
    if targets.dtype() != input_lengths.dtype() || targets.dtype() != target_lengths.dtype() {
        return None;
    }
    macro_rules! run {
        ($index:ty) => {
            try_ctc_loss_typed::<F, $index>(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank,
            )
        };
    }
    match targets.dtype() {
        DType::I8 => run!(i8),
        DType::I16 => run!(i16),
        DType::I32 => run!(i32),
        DType::I64 => run!(i64),
        DType::U8 => run!(u8),
        DType::U16 => run!(u16),
        DType::U32 => run!(u32),
        DType::U64 => run!(u64),
        _ => None,
    }
}

fn try_ctc_loss_typed<F: Float + Element + Pod, I: PrimInt + Element + Pod>(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    blank: usize,
) -> Option<FlexTensor> {
    if log_probs.shape().num_dims() != 3 || targets.shape().num_dims() != 2 {
        return None;
    }
    let [time, batch, classes] = log_probs.shape().dims();
    let [target_batch, width] = targets.shape().dims();
    let states = width.checked_mul(2)?.checked_add(1)?;
    let index_max = ToPrimitive::to_u128(&I::max_value())?.min(isize::MAX as u128);
    if time == 0
        || batch == 0
        || classes == 0
        || blank >= classes
        || target_batch != batch
        || input_lengths.shape() != Shape::new([batch])
        || target_lengths.shape() != Shape::new([batch])
        // The default recurrence constructs timestep and state indices in I and
        // uses signed ranges/slices. Preserve wrapping-sensitive cases via fallback.
        || time as u128 > index_max
        || states as u128 > index_max
        || blank as u128 > index_max
    {
        return None;
    }
    time.checked_mul(batch)?.checked_mul(classes)?;
    time.checked_mul(batch)?.checked_mul(states)?;
    batch.checked_mul(width)?;
    states.checked_mul(2)?;

    let log_probs = log_probs.to_contiguous();
    let targets = targets.to_contiguous();
    let input_lengths = input_lengths.to_contiguous();
    let target_lengths = target_lengths.to_contiguous();
    let scores = log_probs.as_slice::<F>()?;
    let labels = targets.as_slice::<I>()?;
    let inputs = input_lengths.as_slice::<I>()?;
    let lengths = target_lengths.as_slice::<I>()?;

    // Keep unusual scores and invalid inputs on the existing path, including its
    // release-mode length behavior. The pre-gather also reads padded labels.
    if scores
        .iter()
        .any(|&score| score.is_nan() || score > F::zero())
        || labels
            .iter()
            .any(|label| ToPrimitive::to_usize(label).is_none_or(|label| label >= classes))
        || inputs.iter().zip(lengths).any(|(&input, &target)| {
            let (Some(input), Some(target)) = (
                ToPrimitive::to_usize(&input),
                ToPrimitive::to_usize(&target),
            ) else {
                return true;
            };
            input < 1
                || input > time
                || target > width
                || input < target
                // Generic endpoint extraction subtracts one before clamping.
                // For unsigned empty targets this wraps before the gather.
                || (I::min_value() == I::zero() && target == 0)
        })
    {
        return None;
    }

    let ctc = Ctc {
        scores,
        labels,
        inputs,
        lengths,
        batch,
        classes,
        width,
        states: 2 * lengths.iter().map(|&v| index(v)).max()? + 1,
        blank,
    };
    let mut losses = vec![F::zero(); batch];
    ctc.evaluate(&mut losses);
    Some(FlexTensor::new(
        Bytes::from_elems(losses),
        Layout::contiguous(Shape::new([batch])),
        F::dtype(),
    ))
}

struct Ctc<'a, F, I> {
    scores: &'a [F],
    labels: &'a [I],
    inputs: &'a [I],
    lengths: &'a [I],
    batch: usize,
    classes: usize,
    width: usize,
    states: usize,
    blank: usize,
}

impl<F: Float + Element + Pod, I: PrimInt + Element + Pod> Ctc<'_, F, I> {
    fn evaluate(&self, losses: &mut [F]) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let work = self
                .inputs
                .iter()
                .zip(self.lengths)
                .fold(0usize, |work, (&t, &s)| {
                    work.saturating_add(index(t).saturating_mul(2 * index(s) + 1))
                });
            if self.batch >= 4
                && work >= super::PARALLEL_THRESHOLD
                && rayon::current_num_threads() > 1
            {
                losses.par_iter_mut().enumerate().for_each_init(
                    || vec![F::neg_infinity(); 2 * self.states],
                    |scratch, (sample, loss)| *loss = self.sample(sample, scratch),
                );
                return;
            }
        }
        let mut scratch = vec![F::neg_infinity(); 2 * self.states];
        for (sample, loss) in losses.iter_mut().enumerate() {
            *loss = self.sample(sample, &mut scratch);
        }
    }

    fn sample(&self, sample: usize, scratch: &mut [F]) -> F {
        let target_length = index(self.lengths[sample]);
        let active = 2 * target_length + 1;
        let labels = &self.labels[sample * self.width..sample * self.width + target_length];
        let (previous, current) = scratch.split_at_mut(self.states);
        let (mut previous, mut current) = (&mut previous[..active], &mut current[..active]);
        previous.fill(F::neg_infinity());
        let start = sample * self.classes;
        previous[0] = self.scores[start + self.blank];
        if let Some(&first) = labels.first() {
            previous[1] = self.scores[start + index(first)];
        }

        // Transitions only read the same state and lower predecessors. Padded
        // states cannot contribute to the loss; rows after input_length are frozen.
        for time in 1..index(self.inputs[sample]) {
            let start = (time * self.batch + sample) * self.classes;
            for state in 0..active {
                let label = if state % 2 == 0 {
                    self.blank
                } else {
                    index(labels[state / 2])
                };
                let predecessor = if state > 0 {
                    previous[state - 1]
                } else {
                    F::neg_infinity()
                };
                let mut value = log_add(previous[state], predecessor);
                if state > 1 && label != self.blank && label != index(labels[state / 2 - 1]) {
                    value = log_add(value, previous[state - 2]);
                }
                current[state] = value + self.scores[start + label];
            }
            core::mem::swap(&mut previous, &mut current);
        }
        let last_label = if target_length == 0 {
            F::neg_infinity()
        } else {
            previous[active - 2]
        };
        -log_add(previous[active - 1], last_label)
    }
}

#[inline]
fn index<I: PrimInt>(value: I) -> usize {
    // All stored indices and lengths were checked before entering the recurrence.
    value.to_usize().unwrap()
}

#[inline]
fn log_add<F: Float>(a: F, b: F) -> F {
    let max = if a < b { b } else { a };
    if a == F::neg_infinity() || b == F::neg_infinity() {
        // The addition also preserves the default recurrence's signed-zero behavior.
        max + F::zero()
    } else {
        // Each operation stays in F, preserving the tensor path's half-precision rounding.
        max + (-(a - b).abs()).exp().ln_1p()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
