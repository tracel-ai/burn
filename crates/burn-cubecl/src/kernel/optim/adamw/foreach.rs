//! Deferred AdamW updates grouped by the complete execution context captured at collection.

#[cfg(not(feature = "fusion"))]
use super::CubeTensor;
use super::{AdamWHostScalars, AdamWInputs, AdamWOutput};

#[cfg(not(feature = "fusion"))]
pub use raw::AdamWPending;

/// No raw updates can be prepared when kernel fusion owns execution.
#[cfg(feature = "fusion")]
pub enum AdamWPending {}

/// A completed batch update or a singleton awaiting release of the original parameter.
///
/// Pending tensors stay inline to avoid a heap allocation for each singleton.
#[allow(clippy::large_enum_variant)]
pub enum AdamWBatchResult<T> {
    /// The parameter was updated by a multi-tensor launch.
    Updated(AdamWOutput<T>),
    /// No kernel has run for this parameter yet.
    Deferred(AdamWPending),
}

#[cfg(not(feature = "fusion"))]
impl AdamWBatchResult<CubeTensor> {
    /// Finish this update after dropping the caller's retained original parameter.
    ///
    /// Deferred singletons use the single-parameter kernel on their captured stream. Releasing
    /// the original first lets that kernel reuse exclusively owned parameter and state buffers.
    pub fn finish(self) -> AdamWOutput<CubeTensor> {
        match self {
            Self::Updated(output) => output,
            Self::Deferred(pending) => pending.single(),
        }
    }
}

#[cfg(feature = "fusion")]
impl<T> AdamWBatchResult<T> {
    /// Return a completed update; kernel fusion cannot create a deferred raw update.
    pub fn finish(self) -> AdamWOutput<T> {
        match self {
            Self::Updated(output) => output,
            Self::Deferred(pending) => match pending {},
        }
    }
}

/// Capture an eligible CPU update without allocating outputs or submitting kernels.
///
/// # Errors
///
/// Returns the original inputs for unsupported tensors, execution modes, or launch budgets.
/// An item occupying the entire chunk budget is refused: no second nonempty item could join it.
#[cfg(not(feature = "fusion"))]
#[allow(clippy::result_large_err)]
pub fn prepare_adamw(
    scalars: AdamWHostScalars,
    cautious: bool,
    amsgrad: bool,
    inputs: AdamWInputs<CubeTensor>,
) -> Result<AdamWPending, AdamWInputs<CubeTensor>> {
    raw::prepare(scalars, cautious, amsgrad, inputs)
}

/// Refuse deferred raw updates when kernel fusion owns execution.
#[cfg(feature = "fusion")]
#[allow(clippy::result_large_err)]
pub fn prepare_adamw<T>(
    _scalars: AdamWHostScalars,
    _cautious: bool,
    _amsgrad: bool,
    inputs: AdamWInputs<T>,
) -> Result<AdamWPending, AdamWInputs<T>> {
    Err(inputs)
}

/// Apply compatible batches, returning results in input order.
///
/// Multi-tensor batches allocate and launch on their saved execution stream. Singletons stay
/// deferred until [`AdamWBatchResult::finish`] so callers can release retained parameters first.
/// Oversized inputs are refused at preparation.
#[cfg(not(feature = "fusion"))]
pub fn execute_adamw_batch(inputs: Vec<AdamWPending>) -> Vec<AdamWBatchResult<CubeTensor>> {
    raw::execute(inputs)
}

/// There are no pending raw updates under kernel fusion.
#[cfg(feature = "fusion")]
pub fn execute_adamw_batch<T>(inputs: Vec<AdamWPending>) -> Vec<AdamWBatchResult<T>> {
    assert!(
        inputs.is_empty(),
        "fusion never creates a pending raw update"
    );
    Vec::new()
}

#[cfg(not(feature = "fusion"))]
mod raw {
    use super::*;
    use crate::kernel::optim::adamw::eager::{self, AdamWLaunchScalars, DecayMode, update};
    use burn_backend::DType;
    use burn_std::id::StreamId;
    use cubecl::{
        prelude::*,
        std::tensor::layout::linear::{LinearView, LinearViewMut},
    };

    // Limits bound kernel arguments and compile-time expansion. Runtime binding limits can
    // reduce the tensor count further. Large inputs keep the vectorized single-step path.
    const MAX_TENSORS: usize = 8;
    fn chunk_limits(param: &CubeTensor) -> (usize, usize) {
        (
            4096,
            32.min(param.client.properties().hardware.max_cube_count.0 as usize),
        )
    }

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct TensorChunks {
        first_cube: usize,
        end_cube: usize,
        len: usize,
    }

    /// An eligible update with its collection-time execution stream.
    ///
    /// Constructed by [`prepare_adamw`]; inputs remain unmodified until batch execution.
    pub struct AdamWPending {
        inputs: AdamWInputs<CubeTensor>,
        scalars: AdamWHostScalars,
        decay: DecayMode,
        amsgrad: bool,
        stream: StreamId,
    }

    impl AdamWPending {
        fn param(&self) -> &CubeTensor {
            eager::roles(&self.inputs)[0].unwrap()
        }

        fn same_group(&self, other: &Self) -> bool {
            let a = self.scalars;
            let b = other.scalars;
            let state = |inputs: &AdamWInputs<CubeTensor>| match inputs {
                AdamWInputs::Initial { .. } => (false, false),
                AdamWInputs::State { max_moment_2, .. } => (true, max_moment_2.is_some()),
            };
            self.param().device == other.param().device
                && self.param().client.service_id() == other.param().client.service_id()
                && self.stream == other.stream
                && self.param().dtype == other.param().dtype
                && a.lr.to_bits() == b.lr.to_bits()
                && a.time == b.time
                && a.beta_1.to_bits() == b.beta_1.to_bits()
                && a.beta_2.to_bits() == b.beta_2.to_bits()
                && a.epsilon.to_bits() == b.epsilon.to_bits()
                && a.weight_decay.to_bits() == b.weight_decay.to_bits()
                && self.amsgrad == other.amsgrad
                && self.decay == other.decay
                && state(&self.inputs) == state(&other.inputs)
        }

        fn chunks(&self) -> usize {
            self.param()
                .meta
                .num_elements()
                .div_ceil(chunk_limits(self.param()).0)
        }

        fn bindings(&self) -> usize {
            // One metadata binding is shared by the launch.
            eager::num_bindings(&self.inputs, self.amsgrad) as usize - 1
        }

        pub(super) fn single(self) -> AdamWOutput<CubeTensor> {
            self.stream
                .executes(|| eager::launch(self.scalars, self.decay, self.amsgrad, self.inputs))
        }
    }

    #[allow(clippy::result_large_err)]
    pub(super) fn prepare(
        scalars: AdamWHostScalars,
        cautious: bool,
        amsgrad: bool,
        inputs: AdamWInputs<CubeTensor>,
    ) -> Result<AdamWPending, AdamWInputs<CubeTensor>> {
        if eager::roles(&inputs)[0].unwrap().device.runtime() != cubecl::RuntimeId::Cpu
            || !eager::supported(&inputs, amsgrad)
        {
            return Err(inputs);
        }
        let param = eager::roles(&inputs)[0].unwrap();
        let (elements, max_chunks) = chunk_limits(param);
        // An item filling the budget can never be batched. Refuse it before collection to
        // avoid a second module traversal for an update that must use the single-step path.
        if param.meta.num_elements().div_ceil(elements) >= max_chunks {
            return Err(inputs);
        }
        let stream = param.client.execution_stream();
        Ok(AdamWPending {
            inputs,
            scalars,
            decay: scalars.decay_mode(cautious),
            amsgrad,
            stream,
        })
    }

    pub(super) fn execute(inputs: Vec<AdamWPending>) -> Vec<AdamWBatchResult<CubeTensor>> {
        let mut outputs: Vec<_> = (0..inputs.len()).map(|_| None).collect();
        let mut groups: Vec<Vec<(usize, AdamWPending)>> = Vec::new();
        for (index, item) in inputs.into_iter().enumerate() {
            if let Some(group) = groups.iter_mut().find(|group| group[0].1.same_group(&item)) {
                group.push((index, item));
            } else {
                groups.push(vec![(index, item)]);
            }
        }
        for group in groups {
            let mut batch: Vec<(usize, AdamWPending)> = Vec::new();
            let mut chunks = 0;
            let mut bindings = 1;
            for item in group {
                let limit = item.1.param().client.properties().hardware.max_bindings as usize;
                let max_chunks = chunk_limits(item.1.param()).1;
                let n_chunks = item.1.chunks();
                let n_bindings = item.1.bindings();
                if !batch.is_empty()
                    && (batch.len() == MAX_TENSORS
                        || chunks + n_chunks > max_chunks
                        || bindings + n_bindings > limit)
                {
                    finish(core::mem::take(&mut batch), &mut outputs);
                    chunks = 0;
                    bindings = 1;
                }
                chunks += n_chunks;
                bindings += n_bindings;
                batch.push(item);
            }
            finish(batch, &mut outputs);
        }
        outputs.into_iter().map(Option::unwrap).collect()
    }

    fn finish(
        batch: Vec<(usize, AdamWPending)>,
        outputs: &mut [Option<AdamWBatchResult<CubeTensor>>],
    ) {
        if batch.is_empty() {
            return;
        }
        if batch.len() == 1 {
            let (index, item) = batch.into_iter().next().unwrap();
            outputs[index] = Some(AdamWBatchResult::Deferred(item));
            return;
        }
        let stream = batch[0].1.stream;
        stream.executes(|| {
            let (indices, items): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
            for (index, output) in indices.into_iter().zip(launch(items)) {
                outputs[index] = Some(AdamWBatchResult::Updated(output));
            }
        });
    }

    #[allow(clippy::needless_range_loop)] // CubeCL unrolls indices into Sequence arguments.
    #[cube(launch_unchecked, address_type = "dynamic")]
    fn adamw_foreach<F: Float>(
        params: Sequence<LinearView<'_, Vector<F, Const<1>>>>,
        grads: Sequence<LinearView<'_, Vector<F, Const<1>>>>,
        old_m1: ComptimeOption<Sequence<LinearView<'_, Vector<F, Const<1>>>>>,
        old_m2: ComptimeOption<Sequence<LinearView<'_, Vector<F, Const<1>>>>>,
        old_max: ComptimeOption<Sequence<LinearView<'_, Vector<F, Const<1>>>>>,
        mut out_params: Sequence<LinearViewMut<'_, Vector<F, Const<1>>>>,
        mut out_m1: Sequence<LinearViewMut<'_, Vector<F, Const<1>>>>,
        mut out_m2: Sequence<LinearViewMut<'_, Vector<F, Const<1>>>>,
        out_max: ComptimeOption<Sequence<LinearViewMut<'_, Vector<F, Const<1>>>>>,
        scalars: &AdamWLaunchScalars,
        #[comptime] chunks: Vec<TensorChunks>,
        #[comptime] chunk_elements: usize,
        #[comptime] decay: DecayMode,
        #[define(F)] _dtype: ElemType,
    ) {
        #[unroll]
        for t in 0..comptime![chunks.len()] {
            let first = comptime![chunks[t].first_cube];
            let end = comptime![chunks[t].end_cube];
            if CUBE_POS_X as usize >= first && (CUBE_POS_X as usize) < end {
                let start = (CUBE_POS_X as usize - first) * chunk_elements;
                let len = comptime![chunks[t].len];
                // One uniform branch per tensor covers all its chunks. Each CPU unit
                // iterates through several elements to amortize worker scheduling.
                let mut local = UNIT_POS_X as usize;
                while local < chunk_elements {
                    let pos = start + local;
                    if pos >= len {
                        break;
                    }
                    let (param, m1, m2, maximum) = update::<F, Const<1>>(
                        params.index(t).read(pos),
                        grads.index(t).read(pos),
                        old_m1.as_ref().map(|values| values.index(t).read(pos)),
                        old_m2.as_ref().map(|values| values.index(t).read(pos)),
                        old_max.as_ref().map(|values| values.index(t).read(pos)),
                        scalars,
                        decay,
                    );
                    #[comptime]
                    if let ComptimeOption::Some(mut out) = out_max.clone() {
                        out.index_mut(t).write(pos, maximum);
                    }
                    out_params.index_mut(t).write(pos, param);
                    out_m1.index_mut(t).write(pos, m1);
                    out_m2.index_mut(t).write(pos, m2);
                    local += CUBE_DIM_X as usize;
                }
            }
        }
    }

    fn launch(items: Vec<AdamWPending>) -> Vec<AdamWOutput<CubeTensor>> {
        let scalars = items[0].scalars;
        let decay = items[0].decay;
        let amsgrad = items[0].amsgrad;
        let client = items[0].param().client.clone();
        let reuse: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let others = items
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .flat_map(|(_, other)| eager::roles(&other.inputs).into_iter().flatten());
                eager::reuse_mask(&item.inputs, others)
            })
            .collect();
        let steps: Vec<_> = items
            .into_iter()
            .zip(&reuse)
            .map(|(item, &reuse)| eager::prepare_outputs(item.inputs, amsgrad, reuse))
            .collect();
        let n = steps.len();
        let with_state = steps[0].moment_1.is_some();
        let with_max = steps[0].old_max.is_some();
        let mut params = SequenceArg::new();
        let mut grads = SequenceArg::new();
        let mut old_m1 = SequenceArg::new();
        let mut old_m2 = SequenceArg::new();
        let mut old_max = SequenceArg::new();
        let mut out_params = SequenceArg::new();
        let mut out_m1 = SequenceArg::new();
        let mut out_m2 = SequenceArg::new();
        let mut out_max = SequenceArg::new();
        let mut chunks = Vec::new();
        let mut cube_count = 0;
        let chunk_elements = chunk_limits(&steps[0].param).0;
        let mut address = AddressType::U32;
        let output_view = |tensor: &CubeTensor, reuse, input_pos| {
            if reuse {
                tensor.as_linear_view_alias(input_pos)
            } else {
                tensor.clone().into_linear_view()
            }
        };
        for (i, step) in steps.iter().enumerate() {
            for tensor in [
                Some(&step.param),
                Some(&step.grad),
                step.moment_1.as_ref(),
                step.moment_2.as_ref(),
                step.old_max.as_ref(),
                Some(&step.out_param),
                Some(&step.out_moment_1),
                Some(&step.out_moment_2),
                step.out_max.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if tensor.required_address_type() == AddressType::U64 {
                    address = AddressType::U64;
                }
            }
            params.push(step.param.clone().into_linear_view());
            grads.push(step.grad.clone().into_linear_view());
            if let Some(t) = &step.moment_1 {
                old_m1.push(t.clone().into_linear_view());
            }
            if let Some(t) = &step.moment_2 {
                old_m2.push(t.clone().into_linear_view());
            }
            if let Some(t) = &step.old_max {
                old_max.push(t.clone().into_linear_view());
            }
            out_params.push(output_view(&step.out_param, reuse[i][0], i));
            out_m1.push(output_view(&step.out_moment_1, reuse[i][1], 2 * n + i));
            out_m2.push(output_view(&step.out_moment_2, reuse[i][2], 3 * n + i));
            if let Some(t) = &step.out_max {
                out_max.push(output_view(t, reuse[i][3], 4 * n + i));
            }
            let len = step.param.meta.num_elements();
            let end_cube = cube_count + len.div_ceil(chunk_elements);
            chunks.push(TensorChunks {
                first_cube: cube_count,
                end_cube,
                len,
            });
            cube_count = end_cube;
        }
        let units = 8
            .min(client.properties().hardware.max_units_per_cube)
            .min(client.properties().hardware.max_cube_dim.0)
            .max(1);
        let storage = burn_backend::cubecl::dtype_to_storage_type(DType::F32);
        unsafe {
            adamw_foreach::launch_unchecked(
                &client,
                CubeCount::Static(cube_count as u32, 1, 1),
                CubeDim::new_1d(units),
                address,
                params,
                grads,
                with_state.then_some(old_m1).into(),
                with_state.then_some(old_m2).into(),
                with_max.then_some(old_max).into(),
                out_params,
                out_m1,
                out_m2,
                amsgrad.then_some(out_max).into(),
                scalars.to_launch(storage),
                chunks,
                chunk_elements,
                decay,
                storage,
            );
        }
        steps
            .into_iter()
            .map(|step| AdamWOutput {
                param: step.out_param,
                moment_1: step.out_moment_1,
                moment_2: step.out_moment_2,
                max_moment_2: step.out_max.or(step.preserved_max),
            })
            .collect()
    }
}
