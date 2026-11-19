//! Native CTC: rolling forward rows, and alpha history with rolling beta rows for backward.

use alloc::{vec, vec::Vec};
use burn_backend::{
    DType, Element, TensorMetadata,
    ops::ctc::{ctc_loss_backward_default, ctc_loss_default},
};
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
    try_ctc(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        None,
        blank,
    )
}

pub(super) fn ctc_loss_backward(
    log_probs: FlexTensor,
    targets: FlexTensor,
    input_lengths: FlexTensor,
    target_lengths: FlexTensor,
    grad_loss: FlexTensor,
    blank: usize,
) -> FlexTensor {
    try_ctc(
        &log_probs,
        &targets,
        &input_lengths,
        &target_lengths,
        Some(&grad_loss),
        blank,
    )
    .unwrap_or_else(|| {
        ctc_loss_backward_default::<Flex>(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        )
    })
}

fn try_ctc(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    grad_loss: Option<&FlexTensor>,
    blank: usize,
) -> Option<FlexTensor> {
    match log_probs.dtype() {
        DType::F16 => dispatch_indices::<f16, f32>(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        ),
        DType::BF16 => dispatch_indices::<bf16, f32>(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        ),
        DType::F32 => dispatch_indices::<f32, f32>(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        ),
        DType::F64 => dispatch_indices::<f64, f64>(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            blank,
        ),
        _ => None,
    }
}

fn dispatch_indices<F: Float + Element + Pod, A: Float + Element>(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    grad_loss: Option<&FlexTensor>,
    blank: usize,
) -> Option<FlexTensor> {
    if targets.dtype() != input_lengths.dtype() || targets.dtype() != target_lengths.dtype() {
        return None;
    }
    macro_rules! run {
        ($index:ty) => {
            try_ctc_loss_typed::<F, $index, A>(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                grad_loss,
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

fn try_ctc_loss_typed<F: Float + Element + Pod, I: PrimInt + Element + Pod, A: Float + Element>(
    log_probs: &FlexTensor,
    targets: &FlexTensor,
    input_lengths: &FlexTensor,
    target_lengths: &FlexTensor,
    grad_loss: Option<&FlexTensor>,
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
        || grad_loss.is_some_and(|grad| grad.shape() != Shape::new([batch]) || grad.dtype() != F::dtype())
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
    let (output, layout) = if let Some(grad_loss) = grad_loss {
        let grad_loss = grad_loss.to_contiguous();
        let mut grad = vec![F::zero(); time * batch * classes];
        ctc.backward::<A>(&mut grad, grad_loss.as_slice::<F>()?, time);
        // Each sample owns a contiguous gradient slice, allowing parallel writes
        // without a full-sized intermediate or a transpose copy.
        (
            grad,
            Layout::contiguous(Shape::new([batch, time, classes])).transpose(0, 1),
        )
    } else {
        let mut losses = vec![F::zero(); batch];
        ctc.evaluate(&mut losses);
        (losses, Layout::contiguous(Shape::new([batch])))
    };
    Some(FlexTensor::new(
        Bytes::from_elems(output),
        layout,
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
    #[cfg(feature = "rayon")]
    fn parallel(&self) -> bool {
        let work = self
            .inputs
            .iter()
            .zip(self.lengths)
            .fold(0usize, |work, (&t, &s)| {
                work.saturating_add(index(t).saturating_mul(2 * index(s) + 1))
            });
        self.batch >= 4 && work >= super::PARALLEL_THRESHOLD && rayon::current_num_threads() > 1
    }

    fn evaluate(&self, losses: &mut [F]) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            if self.parallel() {
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
        let (previous, current) = scratch.split_at_mut(self.states);
        let (mut previous, mut current) = (&mut previous[..active], &mut current[..active]);
        self.seed_alpha(sample, previous);

        // Transitions only read the same state and lower predecessors. Padded
        // states cannot contribute to the loss; rows after input_length are frozen.
        for time in 1..index(self.inputs[sample]) {
            self.alpha_step(time, sample, previous, current);
            core::mem::swap(&mut previous, &mut current);
        }
        let last_label = if target_length == 0 {
            F::neg_infinity()
        } else {
            previous[active - 2]
        };
        -log_add(previous[active - 1], last_label)
    }

    fn label(&self, sample: usize, state: usize) -> usize {
        if state.is_multiple_of(2) {
            self.blank
        } else {
            index(self.labels[sample * self.width + state / 2])
        }
    }

    fn seed_alpha<A: Float + Element>(&self, sample: usize, row: &mut [A]) {
        row.fill(A::neg_infinity());
        row[0] = self.scores[sample * self.classes + self.blank].elem();
        if row.len() > 1 {
            row[1] = self.scores[sample * self.classes + self.label(sample, 1)].elem();
        }
    }

    fn alpha_step<A: Float + Element>(
        &self,
        time: usize,
        sample: usize,
        previous: &[A],
        current: &mut [A],
    ) {
        let start = (time * self.batch + sample) * self.classes;
        for (state, output) in current.iter_mut().enumerate() {
            let label = self.label(sample, state);
            let predecessor = if state > 0 {
                previous[state - 1]
            } else {
                A::neg_infinity()
            };
            let mut value = log_add(previous[state], predecessor);
            if state > 1 && label != self.blank && label != self.label(sample, state - 2) {
                value = log_add(value, previous[state - 2]);
            }
            *output = value + self.scores[start + label].elem::<A>();
        }
    }

    fn backward<A: Float + Element>(&self, grad: &mut [F], grad_loss: &[F], time: usize) {
        #[cfg(feature = "rayon")]
        if self.parallel() {
            use rayon::prelude::*;
            grad.par_chunks_mut(time * self.classes)
                .enumerate()
                .for_each_init(
                    || BackwardScratch::<A>::new(self.states, self.classes),
                    |scratch, (sample, grad)| {
                        self.sample_backward(sample, grad_loss[sample], grad, scratch)
                    },
                );
            return;
        }
        let mut scratch = BackwardScratch::<A>::new(self.states, self.classes);
        for (sample, grad) in grad.chunks_mut(time * self.classes).enumerate() {
            self.sample_backward(sample, grad_loss[sample], grad, &mut scratch);
        }
    }

    fn sample_backward<A: Float + Element>(
        &self,
        sample: usize,
        grad_loss: F,
        grad: &mut [F],
        scratch: &mut BackwardScratch<A>,
    ) {
        let time = index(self.inputs[sample]);
        let active = 2 * index(self.lengths[sample]) + 1;
        let weight = grad_loss.elem::<A>();
        if weight == A::zero() {
            return;
        }
        // Recompute alpha inside backward instead of retaining its history in
        // the autodiff graph. Scratch is reused across samples, per worker.
        scratch.alpha.resize(time * active, A::neg_infinity());
        self.seed_alpha(sample, &mut scratch.alpha[..active]);
        for t in 1..time {
            let (history, current) = scratch.alpha.split_at_mut(t * active);
            self.alpha_step(
                t,
                sample,
                &history[(t - 1) * active..],
                &mut current[..active],
            );
        }
        let last = &scratch.alpha[(time - 1) * active..];
        let last_label = if active > 1 {
            last[active - 2]
        } else {
            A::neg_infinity()
        };
        let nll = -log_add(last[active - 1], last_label);
        // Match the backend backward contract: unreachable samples and padded
        // timesteps have zero gradients, including zero_infinity masking.
        if nll.is_infinite() {
            return;
        }

        let (next, current) = scratch.beta.split_at_mut(self.states);
        let (mut next, mut current) = (&mut next[..active], &mut current[..active]);
        next.fill(A::neg_infinity());
        next[active - 1] = A::zero();
        if active > 1 {
            next[active - 2] = A::zero();
        }
        for t in (0..time).rev() {
            let start = (t * self.batch + sample) * self.classes;
            scratch.posterior.fill(A::zero());
            for (state, &beta) in next.iter().enumerate() {
                let alpha = scratch.alpha[t * active + state];
                let class = self.label(sample, state);
                scratch.posterior[class] = scratch.posterior[class] + (alpha + beta + nll).exp();
            }
            for (class, output) in grad[t * self.classes..(t + 1) * self.classes]
                .iter_mut()
                .enumerate()
            {
                let probability = self.scores[start + class].elem::<A>().exp();
                *output = ((probability - scratch.posterior[class]) * weight).elem();
            }
            if t == 0 {
                break;
            }
            // Beta excludes the current emission. This avoids subtracting
            // -inf emissions when combining alpha and beta into posteriors.
            for (state, output) in current.iter_mut().enumerate() {
                let label = self.label(sample, state);
                let mut value = next[state] + self.scores[start + label].elem::<A>();
                if state + 1 < active {
                    value = log_add(
                        value,
                        next[state + 1]
                            + self.scores[start + self.label(sample, state + 1)].elem::<A>(),
                    );
                }
                if state + 2 < active
                    && label != self.blank
                    && label != self.label(sample, state + 2)
                {
                    value = log_add(
                        value,
                        next[state + 2]
                            + self.scores[start + self.label(sample, state + 2)].elem::<A>(),
                    );
                }
                *output = value;
            }
            core::mem::swap(&mut next, &mut current);
        }
    }
}

struct BackwardScratch<A> {
    alpha: Vec<A>,
    beta: Vec<A>,
    posterior: Vec<A>,
}

impl<A: Float> BackwardScratch<A> {
    fn new(states: usize, classes: usize) -> Self {
        Self {
            alpha: Vec::new(),
            beta: vec![A::neg_infinity(); 2 * states],
            posterior: vec![A::zero(); classes],
        }
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
