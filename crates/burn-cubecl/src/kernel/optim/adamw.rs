//! Fused AdamW optimizer step.
//!
//! A vector kernel and an optional scalar tail apply an AdamW update to one parameter: they read it,
//! its gradient and whichever state exists, and writes the parameter and moment tensors. Only
//! `F32` row tensors are handled here; every other case is refused before a launch is submitted
//! so the caller can run its reference implementation instead.

#[cfg(not(feature = "fusion"))]
use crate::tensor::CubeTensor;

/// The tensors one AdamW step reads.
pub enum AdamWInputs<T> {
    /// No optimizer state exists yet; the moments are initialized from the gradient.
    Initial {
        /// The parameter to update.
        param: T,
        /// The gradient of the loss with respect to the parameter.
        grad: T,
    },
    /// The state exists; the first and second moments are updated from it.
    State {
        /// The parameter to update.
        param: T,
        /// The gradient of the loss with respect to the parameter.
        grad: T,
        /// The first moment.
        moment_1: T,
        /// The second moment.
        moment_2: T,
        /// The AMSGrad maximum of the second moment, when one was recorded.
        max_moment_2: Option<T>,
    },
}

/// The tensors one AdamW step produces, reusing exclusively owned inputs when possible.
pub struct AdamWOutput<T> {
    /// The updated parameter.
    pub param: T,
    /// The updated first moment.
    pub moment_1: T,
    /// The updated second moment.
    pub moment_2: T,
    /// The updated AMSGrad maximum when AMSGrad is enabled; otherwise the input maximum,
    /// passed through untouched.
    pub max_moment_2: Option<T>,
}

/// Host scalars for one AdamW step.
///
/// Everything is computed in the same precision and order as the reference implementation:
/// the bias corrections are divisors computed with `f32::powi`, never reciprocals, and the
/// decay rates keep their `f64` host precision until they are converted for the kernel.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "fusion", allow(dead_code))]
pub struct AdamWHostScalars {
    lr: f64,
    beta_1: f32,
    beta_2: f32,
    factor_1: f32,
    factor_2: f32,
    bias_1: f32,
    bias_2: f32,
    epsilon: f32,
    decay_rate: f64,
    one_minus_decay_rate: f64,
}

impl AdamWHostScalars {
    /// Compute the scalars for a step at `time`.
    ///
    /// `time` is the 1-based step count used for bias correction: a fresh state passes `1`.
    pub fn new(
        lr: f64,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        weight_decay: f32,
        time: usize,
    ) -> Self {
        debug_assert!(time >= 1, "AdamW bias correction uses a 1-based time");
        let time_i32 = time as i32;
        let decay_rate = lr * (weight_decay as f64);
        Self {
            lr,
            beta_1,
            beta_2,
            factor_1: 1.0 - beta_1,
            factor_2: 1.0 - beta_2,
            bias_1: 1.0 - beta_1.powi(time_i32),
            bias_2: 1.0 - beta_2.powi(time_i32),
            epsilon,
            decay_rate,
            one_minus_decay_rate: 1.0 - decay_rate,
        }
    }
}

#[cfg(not(feature = "fusion"))]
mod eager {
    use super::{AdamWHostScalars, AdamWInputs, AdamWOutput};
    use crate::{
        kernel::utils::address_type,
        ops::{max_vector_size, numeric::empty_device_dtype},
        tensor::CubeTensor,
    };
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_backend::{DType, TensorMetadata};
    use cubecl::{
        calculate_cube_count_elemwise,
        prelude::*,
        std::tensor::layout::linear::{LinearView, LinearViewMut},
    };

    /// The weight decay variant applied by one AdamW step.
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    enum DecayMode {
        /// No weight decay: the parameter is used as-is.
        None,
        /// Standard decoupled weight decay.
        Standard,
        /// Cautious weight decay: skip the decay where it opposes the update direction.
        Cautious,
    }

    #[derive(CubeLaunch, CubeType)]
    struct AdamWLaunchScalars {
        lr: InputScalar,
        beta_1: InputScalar,
        beta_2: InputScalar,
        factor_1: InputScalar,
        factor_2: InputScalar,
        bias_1: InputScalar,
        bias_2: InputScalar,
        epsilon: InputScalar,
        decay_rate: InputScalar,
        one_minus_decay_rate: InputScalar,
        negative_zero: InputScalar,
    }

    impl AdamWHostScalars {
        /// The decay variant these scalars call for.
        fn decay_mode(&self, cautious_weight_decay: bool) -> DecayMode {
            if self.decay_rate == 0.0 {
                DecayMode::None
            } else if cautious_weight_decay {
                DecayMode::Cautious
            } else {
                DecayMode::Standard
            }
        }

        fn to_launch(self, dtype: ElemType) -> AdamWLaunchScalarsLaunch {
            // Every scalar enters the kernel as the tensor's own float type. The host keeps
            // `lr` and the decay rates in `f64`; they are narrowed here, once, exactly like
            // the reference implementation narrows them when it multiplies a tensor by them.
            let scalar = |value: f32| InputScalar::new(value, dtype);
            AdamWLaunchScalarsLaunch::new(
                scalar(self.lr as f32),
                scalar(self.beta_1),
                scalar(self.beta_2),
                scalar(self.factor_1),
                scalar(self.factor_2),
                scalar(self.bias_1),
                scalar(self.bias_2),
                scalar(self.epsilon),
                scalar(self.decay_rate as f32),
                scalar(self.one_minus_decay_rate as f32),
                scalar(-0.0),
            )
        }
    }

    /// Round a product before the next operation, including overflow and signed zero.
    ///
    /// CubeCL contracts a multiply feeding an add/subtract even without fast-math flags.
    /// An FMA with a negative-zero addend computes the rounded product (positive zero would
    /// erase a negative-zero product) without joining the following add/subtract. The addend
    /// must be a runtime scalar: a literal `-0.0` lets the WGSL pipeline fold the FMA back to
    /// a multiply and contract it with the consumer, which reintroduces the overflow difference.
    #[cube]
    fn mul<F: Float, N: Size>(
        lhs: Vector<F, N>,
        rhs: Vector<F, N>,
        negative_zero: Vector<F, N>,
    ) -> Vector<F, N> {
        fma(lhs, rhs, negative_zero)
    }

    /// Apply the decay and the bias-corrected update to one element.
    ///
    /// `v` is the second moment used for the update: the moment itself, or the AMSGrad maximum.
    #[cube]
    fn apply<F: Float, N: Size>(
        param: Vector<F, N>,
        moment_1: Vector<F, N>,
        v: Vector<F, N>,
        scalars: &AdamWLaunchScalars,
        #[comptime] decay: DecayMode,
    ) -> Vector<F, N> {
        let delta = (moment_1 / Vector::new(scalars.bias_1.get::<F>()))
            / ((v / Vector::new(scalars.bias_2.get::<F>())).sqrt()
                + Vector::new(scalars.epsilon.get::<F>()));
        let negative_zero = Vector::new(scalars.negative_zero.get::<F>());

        let decayed = match comptime![decay] {
            DecayMode::None => param,
            DecayMode::Standard => mul::<F, N>(
                param,
                Vector::new(scalars.one_minus_decay_rate.get::<F>()),
                negative_zero,
            ),
            DecayMode::Cautious => {
                let zero = Vector::new(F::new(0.0));
                // The two signs are compared separately: `(p * m1) >= 0` differs from the
                // reference for signed zeros and underflow.
                let same_sign = param
                    .greater_equal(&zero)
                    .equal(&moment_1.greater_equal(&zero));
                param
                    - select_many(
                        same_sign,
                        mul::<F, N>(
                            param,
                            Vector::new(scalars.decay_rate.get::<F>()),
                            negative_zero,
                        ),
                        zero,
                    )
            }
        };

        decayed - mul::<F, N>(delta, Vector::new(scalars.lr.get::<F>()), negative_zero)
    }

    /// The AMSGrad maximum, reproducing `old_max.max_pair(moment_2)`: the new moment wins only
    /// when it is strictly greater, so a NaN and a signed zero keep the old value's bits.
    #[cube]
    fn update_max<F: Float, N: Size>(
        previous: Vector<F, N>,
        moment_2: Vector<F, N>,
    ) -> Vector<F, N> {
        select_many(previous.less_than(&moment_2), moment_2, previous)
    }

    /// Optional moments specialize away at compile time, preserving the reference's
    /// initial-state evaluation boundaries.
    #[allow(clippy::type_complexity)] // CubeCL lowers the four register values directly.
    #[cube]
    fn update<F: Float, N: Size>(
        param: Vector<F, N>,
        grad: Vector<F, N>,
        old_moment_1: ComptimeOption<Vector<F, N>>,
        old_moment_2: ComptimeOption<Vector<F, N>>,
        old_max: ComptimeOption<Vector<F, N>>,
        scalars: &AdamWLaunchScalars,
        #[comptime] decay: DecayMode,
    ) -> (Vector<F, N>, Vector<F, N>, Vector<F, N>, Vector<F, N>) {
        let negative_zero = Vector::new(scalars.negative_zero.get::<F>());
        let mut m1 = mul::<F, N>(
            grad,
            Vector::new(scalars.factor_1.get::<F>()),
            negative_zero,
        );
        let mut m2 = mul::<F, N>(
            grad * grad,
            Vector::new(scalars.factor_2.get::<F>()),
            negative_zero,
        );
        #[comptime]
        if let ComptimeOption::Some(previous) = old_moment_1 {
            // Keep the reference operand order and separately rounded products.
            m1 = mul::<F, N>(
                previous,
                Vector::new(scalars.beta_1.get::<F>()),
                negative_zero,
            ) + m1;
            m2 = mul::<F, N>(
                old_moment_2.unwrap(),
                Vector::new(scalars.beta_2.get::<F>()),
                negative_zero,
            ) + m2;
        }
        let mut maximum = m2;
        #[comptime]
        if let ComptimeOption::Some(previous) = old_max {
            maximum = update_max::<F, N>(previous, m2);
        }
        (
            apply::<F, N>(param, m1, maximum, scalars, decay),
            m1,
            m2,
            maximum,
        )
    }

    /// Every state shape is a combination of optional views, chosen at compile time: the
    /// kernel is specialized per combination and a missing view is never bound, so no dummy
    /// buffer reaches the alias planner.
    #[cube(launch_unchecked, address_type = "dynamic")]
    fn adamw<F: Float, N: Size>(
        param: LinearView<'_, Vector<F, N>>,
        grad: LinearView<'_, Vector<F, N>>,
        old_moment_1: ComptimeOption<LinearView<'_, Vector<F, N>>>,
        old_moment_2: ComptimeOption<LinearView<'_, Vector<F, N>>>,
        old_max_moment_2: ComptimeOption<LinearView<'_, Vector<F, N>>>,
        mut out_param: LinearViewMut<'_, Vector<F, N>>,
        mut out_moment_1: LinearViewMut<'_, Vector<F, N>>,
        mut out_moment_2: LinearViewMut<'_, Vector<F, N>>,
        out_max_moment_2: ComptimeOption<LinearViewMut<'_, Vector<F, N>>>,
        scalars: &AdamWLaunchScalars,
        offset: usize,
        units: usize,
        #[comptime] has_offset: bool,
        #[comptime] decay: DecayMode,
        #[define(F)] _dtype: ElemType,
    ) {
        if comptime![has_offset] {
            if ABSOLUTE_POS != 0 {
                terminate!();
            }
        } else if !out_param.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        // The CPU tail has fewer than one vector's elements. One unit handles them
        // serially, avoiding a separate worker for each of a handful of scalar values.
        let count = if comptime![has_offset] {
            units
        } else {
            1usize.runtime()
        };
        for lane in 0..count {
            let pos = if comptime![has_offset] {
                offset + lane
            } else {
                ABSOLUTE_POS
            };
            let (p, m1, m2, maximum) = update::<F, N>(
                param.read(pos),
                grad.read(pos),
                old_moment_1.as_ref().map(|value| value.read(pos)),
                old_moment_2.as_ref().map(|value| value.read(pos)),
                old_max_moment_2.as_ref().map(|value| value.read(pos)),
                scalars,
                decay,
            );
            #[comptime]
            if let ComptimeOption::Some(mut out_max) = out_max_moment_2 {
                out_max.write(pos, maximum);
            }
            out_param.write(pos, p);
            out_moment_1.write(pos, m1);
            out_moment_2.write(pos, m2);
        }
    }

    /// Whether the tensor's metadata fits the buffer it binds.
    ///
    /// The kernel trusts shapes and strides; a caller-built tensor whose span exceeds its
    /// handle would read out of bounds. The handle's used range is recomputed with checked
    /// arithmetic because its public offset fields may not describe a valid range.
    fn fits(tensor: &CubeTensor) -> bool {
        let shape = tensor.meta.shape();
        let strides = tensor.meta.strides();
        if shape.len() != strides.len() {
            return false;
        }
        let handle = &tensor.handle;
        let start = handle.offset_start.unwrap_or(0);
        let end = handle.offset_end.unwrap_or(0);
        if !start.is_multiple_of(tensor.dtype.size() as u64) {
            return false;
        }
        let Some(available) = handle
            .size()
            .checked_sub(start)
            .and_then(|used| used.checked_sub(end))
        else {
            return false;
        };

        let mut last = 0u64;
        for (size, stride) in shape.iter().zip(strides.iter()) {
            if *size == 0 {
                return true;
            }
            let Some(next) = (*size as u64 - 1)
                .checked_mul(*stride as u64)
                .and_then(|span| last.checked_add(span))
            else {
                return false;
            };
            last = next;
        }
        last.checked_add(1)
            .and_then(|span| span.checked_mul(tensor.dtype.size() as u64))
            .is_some_and(|bytes| bytes <= available)
    }

    /// Tensor bindings plus one metadata buffer. Reserve the latter even on runtimes that
    /// pass metadata without a storage binding, so the same bound is safe for WGSL.
    fn num_bindings<T>(inputs: &AdamWInputs<T>, amsgrad: bool) -> u32 {
        let inputs = match inputs {
            AdamWInputs::Initial { .. } => 2,
            AdamWInputs::State { max_moment_2, .. } => {
                4 + u32::from(amsgrad && max_moment_2.is_some())
            }
        };
        inputs + 3 + u32::from(amsgrad) + 1
    }

    /// Whether the inputs can go through the fused kernel: `F32`, same shape, device and
    /// client service, non-overlapping rows, and not empty.
    fn supported(inputs: &AdamWInputs<CubeTensor>, amsgrad: bool) -> bool {
        let (param, grad, moments) = match inputs {
            AdamWInputs::Initial { param, grad } => (param, grad, None),
            AdamWInputs::State {
                param,
                grad,
                moment_1,
                moment_2,
                max_moment_2,
            } => (param, grad, Some((moment_1, moment_2, max_moment_2))),
        };
        let shape = param.shape();
        if num_bindings(inputs, amsgrad) > param.client.properties().hardware.max_bindings {
            return false;
        }
        // The reference elementwise launch computes `num_dims() - 1`; rank 0 cannot run there,
        // so the fused path must not silently diverge by accepting it either.
        if shape.num_elements() == 0 || shape.num_dims() == 0 {
            return false;
        }
        let service = param.client.service_id();
        let matches = |tensor: &CubeTensor| {
            tensor.dtype == DType::F32
                && !tensor.meta.is_tiled()
                && tensor.qparams.is_none()
                && tensor.is_nonoverlapping()
                && tensor.shape() == shape
                && tensor.device == param.device
                && tensor.client.service_id() == service
                && tensor.handle.service == service
                && fits(tensor)
        };
        if !matches(param) || !matches(grad) {
            return false;
        }
        match moments {
            Some((moment_1, moment_2, max_moment_2)) => {
                matches(moment_1)
                    && matches(moment_2)
                    && max_moment_2.as_ref().map(matches).unwrap_or(true)
            }
            None => true,
        }
    }

    /// Run a fused AdamW step when the inputs qualify, returning them untouched otherwise.
    ///
    /// # Errors
    ///
    /// Returns `inputs` when the tensors are not `F32` row tensors sharing one shape and device.
    #[allow(clippy::result_large_err)]
    pub(super) fn try_adamw(
        scalars: AdamWHostScalars,
        cautious_weight_decay: bool,
        amsgrad: bool,
        inputs: AdamWInputs<CubeTensor>,
    ) -> Result<AdamWOutput<CubeTensor>, AdamWInputs<CubeTensor>> {
        if !supported(&inputs, amsgrad) {
            return Err(inputs);
        }
        let decay = scalars.decay_mode(cautious_weight_decay);
        Ok(launch(scalars, decay, amsgrad, inputs))
    }

    /// Compute ownership before output views add handle references, including aliases with
    /// a disabled AMSGrad maximum that must be preserved.
    fn reuse_mask(inputs: &AdamWInputs<CubeTensor>) -> [bool; 4] {
        let roles = match inputs {
            AdamWInputs::Initial { param, grad } => [Some(param), Some(grad), None, None, None],
            AdamWInputs::State {
                param,
                grad,
                moment_1,
                moment_2,
                max_moment_2,
            } => [
                Some(param),
                Some(grad),
                Some(moment_1),
                Some(moment_2),
                max_moment_2.as_ref(),
            ],
        };
        let distinct = |a: &CubeTensor, b: &CubeTensor| {
            a.handle.service != b.handle.service
                || a.handle.memory.descriptor().id != b.handle.memory.descriptor().id
        };
        let reuse = |role: usize| {
            roles[role].is_some_and(|tensor| {
                tensor.can_mut()
                    // Queued bindings can still read this allocation after their tensor handles
                    // are released. Do not overwrite it before those reads have finished.
                    && !tensor.handle.memory.is_bound()
                    && tensor.is_contiguous()
                    && roles.iter().enumerate().all(|(index, other)| {
                        index == role || other.is_none_or(|other| distinct(tensor, other))
                    })
            })
        };
        [reuse(0), reuse(2), reuse(3), reuse(4)]
    }

    struct PreparedStep {
        param: CubeTensor,
        grad: CubeTensor,
        moment_1: Option<CubeTensor>,
        moment_2: Option<CubeTensor>,
        old_max: Option<CubeTensor>,
        preserved_max: Option<CubeTensor>,
        out_param: CubeTensor,
        out_moment_1: CubeTensor,
        out_moment_2: CubeTensor,
        out_max: Option<CubeTensor>,
    }

    fn prepare_outputs(
        inputs: AdamWInputs<CubeTensor>,
        amsgrad: bool,
        reuse: [bool; 4],
    ) -> PreparedStep {
        let (param, grad, moments, old_max) = match inputs {
            AdamWInputs::Initial { param, grad } => (param, grad, None, None),
            AdamWInputs::State {
                param,
                grad,
                moment_1,
                moment_2,
                max_moment_2,
            } => (param, grad, Some((moment_1, moment_2)), max_moment_2),
        };
        let (moment_1, moment_2) = match moments {
            Some((moment_1, moment_2)) => (Some(moment_1), Some(moment_2)),
            None => (None, None),
        };
        // The recorded maximum only participates in an AMSGrad step; otherwise it is passed
        // through untouched.
        let (old_max, preserved_max) = if amsgrad {
            (old_max, None)
        } else {
            (None, old_max)
        };

        let shape = param.shape();
        let output = |source: Option<&CubeTensor>, reusable: bool| {
            if reusable {
                source.unwrap().clone()
            } else {
                empty_device_dtype(
                    param.client.clone(),
                    param.device.clone(),
                    shape.clone(),
                    param.dtype,
                )
            }
        };
        let out_param = output(Some(&param), reuse[0]);
        let out_moment_1 = output(moment_1.as_ref(), reuse[1]);
        let out_moment_2 = output(moment_2.as_ref(), reuse[2]);
        let out_max = amsgrad.then(|| output(old_max.as_ref(), reuse[3]));
        PreparedStep {
            param,
            grad,
            moment_1,
            moment_2,
            old_max,
            preserved_max,
            out_param,
            out_moment_1,
            out_moment_2,
            out_max,
        }
    }

    fn launch(
        scalars: AdamWHostScalars,
        decay: DecayMode,
        amsgrad: bool,
        inputs: AdamWInputs<CubeTensor>,
    ) -> AdamWOutput<CubeTensor> {
        let reuse = reuse_mask(&inputs);
        let PreparedStep {
            param,
            grad,
            moment_1,
            moment_2,
            old_max,
            preserved_max,
            out_param,
            out_moment_1,
            out_moment_2,
            out_max,
        } = prepare_outputs(inputs, amsgrad, reuse);
        let dtype = param.dtype;
        let client = param.client.clone();
        let device = param.device.clone();
        // Positions count actual tensor inputs. State always binds both moments at 2/3;
        // maximum can alias position 4 only when that optional input exists.
        let view = |tensor: &CubeTensor, reuse: bool, position| {
            if reuse {
                tensor.as_linear_view_alias(position)
            } else {
                tensor.clone().into_linear_view()
            }
        };
        let tensors = [
            Some(&param),
            Some(&grad),
            moment_1.as_ref(),
            moment_2.as_ref(),
            old_max.as_ref(),
            Some(&out_param),
            Some(&out_moment_1),
            Some(&out_moment_2),
            out_max.as_ref(),
        ];
        let contiguous = tensors.iter().flatten().all(|t| t.is_contiguous());
        let numel = param.meta.num_elements();
        // Vectorize eligible CPU inputs; other runtimes keep one scalar launch.
        let mut vector_size = if device.runtime() == cubecl::RuntimeId::Cpu {
            client
                .io_optimized_vector_sizes(dtype.size())
                .max()
                .unwrap_or(1)
        } else {
            1
        };
        for tensor in tensors.into_iter().flatten() {
            if !contiguous {
                vector_size = vector_size.min(max_vector_size(tensor));
            }
            while vector_size > 1
                && tensor.handle.offset_start.unwrap_or(0) % (vector_size * dtype.size()) as u64
                    != 0
            {
                vector_size /= 2;
            }
        }
        // A scalar tail is a second launch. Small irregular tensors stay in one scalar
        // launch; use a vector body plus tail only once a full hardware workgroup fits.
        if numel % vector_size != 0
            && numel / vector_size < client.properties().hardware.max_units_per_cube as usize
        {
            vector_size = 1;
        }
        let storage = dtype_to_storage_type(dtype);
        let address = address_type!(
            param,
            grad,
            moment_1,
            moment_2,
            old_max,
            out_param,
            out_moment_1,
            out_moment_2,
            out_max
        );

        let submit = |param: CubeTensor,
                      grad: CubeTensor,
                      moment_1: Option<CubeTensor>,
                      moment_2: Option<CubeTensor>,
                      old_max: Option<CubeTensor>,
                      width,
                      offset,
                      units| {
            let cube_dim = if offset == 0 {
                CubeDim::new(&client, units)
            } else {
                CubeDim::new_single()
            };
            let cube_count = calculate_cube_count_elemwise(
                &client,
                if offset == 0 { units } else { 1 },
                cube_dim,
            );
            unsafe {
                adamw::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    address,
                    width,
                    param.into_linear_view(),
                    grad.into_linear_view(),
                    moment_1.map(CubeTensor::into_linear_view).into(),
                    moment_2.map(CubeTensor::into_linear_view).into(),
                    old_max.map(CubeTensor::into_linear_view).into(),
                    view(&out_param, reuse[0], 0),
                    view(&out_moment_1, reuse[1], 2),
                    view(&out_moment_2, reuse[2], 3),
                    out_max.as_ref().map(|t| view(t, reuse[3], 4)).into(),
                    scalars.to_launch(storage),
                    offset,
                    units,
                    offset != 0,
                    decay,
                    storage,
                );
            }
        };
        let body = numel / vector_size * vector_size;
        if body != numel {
            submit(
                param.clone(),
                grad.clone(),
                moment_1.clone(),
                moment_2.clone(),
                old_max.clone(),
                vector_size,
                0,
                body / vector_size,
            );
            submit(
                param,
                grad,
                moment_1,
                moment_2,
                old_max,
                1,
                body,
                numel - body,
            );
        } else {
            submit(
                param,
                grad,
                moment_1,
                moment_2,
                old_max,
                vector_size,
                0,
                body / vector_size,
            );
        }

        AdamWOutput {
            param: out_param,
            moment_1: out_moment_1,
            moment_2: out_moment_2,
            max_moment_2: out_max.or(preserved_max),
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn binding_budget_accounts_for_state_variants() {
            let initial = AdamWInputs::Initial {
                param: (),
                grad: (),
            };
            assert_eq!(num_bindings(&initial, false), 6);
            assert_eq!(num_bindings(&initial, true), 7);
            for (maximum, bindings) in [(None, 9), (Some(()), 10)] {
                let state = AdamWInputs::State {
                    param: (),
                    grad: (),
                    moment_1: (),
                    moment_2: (),
                    max_moment_2: maximum,
                };
                assert_eq!(num_bindings(&state, true), bindings);
                assert_eq!(num_bindings(&state, false), 8);
                assert!(num_bindings(&state, true) > 8);
            }
        }
    }
}

#[cfg(not(feature = "fusion"))]
/// Run a fused AdamW step when the inputs qualify, returning them untouched otherwise.
///
/// # Errors
///
/// Returns `inputs` when the tensors are not `F32` row tensors sharing one shape and device,
/// and for rank-0, empty, storage-tiled or quantized inputs.
#[allow(clippy::result_large_err)]
pub fn try_adamw(
    scalars: AdamWHostScalars,
    cautious_weight_decay: bool,
    amsgrad: bool,
    inputs: AdamWInputs<CubeTensor>,
) -> Result<AdamWOutput<CubeTensor>, AdamWInputs<CubeTensor>> {
    eager::try_adamw(scalars, cautious_weight_decay, amsgrad, inputs)
}

#[cfg(feature = "fusion")]
/// A fused step is only implemented for raw Cube tensors, so the inputs are returned untouched.
///
/// # Errors
///
/// Always returns `inputs`.
#[allow(clippy::result_large_err)]
pub fn try_adamw<T>(
    _scalars: AdamWHostScalars,
    _cautious_weight_decay: bool,
    _amsgrad: bool,
    inputs: AdamWInputs<T>,
) -> Result<AdamWOutput<T>, AdamWInputs<T>> {
    Err(inputs)
}
