use core::sync::atomic::{AtomicBool, Ordering};

use crate::{DType, QuantScheme, QuantStore};

/// Whether a tensor's dtype decides the dtype of the operations it enters.
///
/// A [`DType`] describes data; this describes how that data flows. When an operation combines
/// tensors of different dtypes, the output takes the dtype of the inputs that propagate, and an
/// input that inhibits never decides it: a bf16 activation times an f32 weight stays bf16 when the
/// weight inhibits, and an activation quantized to MXFP8 keeps every output it reaches quantized.
///
/// The variants are ordered, `Inhibit < Propagate`, so an operation's output propagates when its
/// strongest input does.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum DTypePropagation {
    /// The tensor's dtype never decides an output's: what a module parameter holds.
    Inhibit,
    /// The tensor's dtype decides the outputs it reaches: inputs, created tensors and casts.
    #[default]
    Propagate,
}

/// A tensor's dtype together with whether it propagates: what decides the dtype of an operation's
/// output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorType {
    /// The data the tensor holds.
    pub dtype: DType,
    /// Whether that dtype decides the outputs the tensor reaches.
    pub propagation: DTypePropagation,
}

/// How a tie between inputs of different dtypes is resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeResolution {
    /// The widest dtype wins, so a mistake costs speed rather than precision.
    Widen,
    /// A tie between propagating inputs of different dtypes is a [`DTypeConflict`]: a test runs
    /// this way so a component that silently widens the signal fails.
    Strict,
}

/// Two propagating inputs of different dtypes met under [`DTypeResolution::Strict`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DTypeConflict {
    /// One propagating input's dtype.
    pub lhs: DType,
    /// Another propagating input's dtype, different from [`lhs`](Self::lhs).
    pub rhs: DType,
}

impl core::fmt::Display for DTypeConflict {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "an operation combined propagating tensors of different dtypes ({:?} and {:?}); \
             cast one of them explicitly",
            self.lhs, self.rhs
        )
    }
}

impl TensorType {
    /// A tensor of `dtype` that propagates.
    pub fn propagating(dtype: DType) -> Self {
        Self {
            dtype,
            propagation: DTypePropagation::Propagate,
        }
    }

    /// A tensor of `dtype` that inhibits.
    pub fn inhibiting(dtype: DType) -> Self {
        Self {
            dtype,
            propagation: DTypePropagation::Inhibit,
        }
    }

    /// The type of a combining operation's output — an elementwise operation, a matmul, a
    /// concatenation, a reduction — from the types of its float inputs.
    ///
    /// The output takes the dtype of the inputs that propagate, or of every input when none
    /// does, the widest winning a tie; it propagates when any input does. Two different dtypes
    /// of the same width promote to `f32`, the narrowest float that holds both. A quantized
    /// dtype is compared by its format alone, its block laid along the innermost axis, so an
    /// operand transposed for a product does not count as a different dtype, and the output
    /// takes that orientation.
    ///
    /// # Panics
    ///
    /// If `inputs` is empty.
    pub fn combine(
        inputs: impl IntoIterator<Item = TensorType>,
        resolution: DTypeResolution,
    ) -> Result<TensorType, DTypeConflict> {
        let mut inputs = inputs.into_iter().map(|input| TensorType {
            dtype: innermost(input.dtype),
            ..input
        });
        let first = inputs
            .next()
            .expect("an operation combines at least one input");
        inputs.try_fold(first, |out, input| out.meet(input, resolution))
    }

    /// Whether a tensor of `dtype` holds this type's format, its orientation aside: a quantized
    /// tensor a layout operation transposed still does.
    pub fn same_format(&self, dtype: DType) -> bool {
        format_key(self.dtype) == format_key(dtype)
    }

    /// The type two inputs resolve to, the order between them irrelevant.
    fn meet(self, other: TensorType, resolution: DTypeResolution) -> Result<Self, DTypeConflict> {
        let propagation = self.propagation.max(other.propagation);
        let dtype = match self.propagation.cmp(&other.propagation) {
            core::cmp::Ordering::Greater => self.dtype,
            core::cmp::Ordering::Less => other.dtype,
            core::cmp::Ordering::Equal if self.same_format(other.dtype) => self.dtype,
            core::cmp::Ordering::Equal => match (resolution, propagation) {
                (DTypeResolution::Strict, DTypePropagation::Propagate) => {
                    return Err(DTypeConflict {
                        lhs: self.dtype,
                        rhs: other.dtype,
                    });
                }
                _ => widest(self.dtype, other.dtype),
            },
        };
        Ok(Self { dtype, propagation })
    }
}

/// The wider of two different dtypes, `f32` when they are as wide.
fn widest(lhs: DType, rhs: DType) -> DType {
    match width(lhs).cmp(&width(rhs)) {
        core::cmp::Ordering::Greater => lhs,
        core::cmp::Ordering::Less => rhs,
        core::cmp::Ordering::Equal if width(lhs) >= 64 => DType::F64,
        core::cmp::Ordering::Equal => DType::F32,
    }
}

/// The bits one value of `dtype` carries, the precision a combining operation compares.
fn width(dtype: DType) -> usize {
    match dtype {
        DType::QFloat(scheme) => scheme.value.size_bits(),
        dtype => dtype.size() * 8,
    }
}

/// `dtype` with its quantized format's orientation removed: the block's extents sorted and the
/// values packed along the innermost axis, so two orientations of one format compare equal.
fn format_key(dtype: DType) -> DType {
    let DType::QFloat(scheme) = innermost(dtype) else {
        return dtype;
    };
    let Some(block) = scheme.block_scale() else {
        return DType::QFloat(scheme);
    };
    let mut extents = block.size.to_vec();
    extents.sort_unstable();
    let base = QuantScheme::default()
        .with_value(scheme.value)
        .with_store(scheme.store)
        .with_mode(scheme.mode)
        .per_block(extents, block.dtype);
    DType::QFloat(match scheme.tensor_scale() {
        Some(global) => base.per_tensor(global),
        None => base,
    })
}

/// `dtype` with a quantized format's block laid along the innermost axis and its values packed
/// along it: the orientation every output of a combining operation takes. A block spanning
/// several axes keeps its own.
fn innermost(dtype: DType) -> DType {
    let DType::QFloat(scheme) = dtype else {
        return dtype;
    };
    DType::QFloat(innermost_scheme(scheme))
}

fn innermost_scheme(scheme: QuantScheme) -> QuantScheme {
    let store = match scheme.store {
        QuantStore::Native => QuantStore::Native,
        QuantStore::PackedNative(_) => QuantStore::PackedNative(0),
        QuantStore::PackedU32(_) => QuantStore::PackedU32(0),
    };
    let reoriented = scheme.with_store(store);
    let Some(block) = scheme.block_scale() else {
        return reoriented;
    };
    // Only a block spanning one axis has an orientation to move; a tile keeps its own.
    let spans: usize = block.size.iter().filter(|size| **size > 1).count();
    if spans > 1 {
        return reoriented;
    }
    let values = block.size.num_elements() as u8;
    let base = QuantScheme::default()
        .with_value(scheme.value)
        .with_store(store)
        .with_mode(scheme.mode)
        .per_block([values], block.dtype);
    match scheme.tensor_scale() {
        Some(global) => base.per_tensor(global),
        None => base,
    }
}

static STRICT: AtomicBool = AtomicBool::new(false);

impl DTypeResolution {
    /// The resolution every operation of this process runs under: [`Widen`](Self::Widen) unless
    /// [`set_current`](Self::set_current) chose [`Strict`](Self::Strict).
    pub fn current() -> Self {
        match STRICT.load(Ordering::Relaxed) {
            true => Self::Strict,
            false => Self::Widen,
        }
    }

    /// Run every operation of this process under `resolution`, for a test suite that holds its
    /// components to the dtype of their signal.
    pub fn set_current(resolution: Self) {
        STRICT.store(resolution == Self::Strict, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantValue, ScaleDtype};

    fn mxfp8() -> DType {
        DType::QFloat(
            QuantScheme::default()
                .with_value(QuantValue::E4M3)
                .with_store(QuantStore::Native)
                .per_block([32], ScaleDtype::UE8M0),
        )
    }

    fn nvfp4() -> DType {
        DType::QFloat(
            QuantScheme::default()
                .with_value(QuantValue::E2M1)
                .with_store(QuantStore::PackedU32(0))
                .per_block([16], ScaleDtype::UE4M3)
                .per_tensor(ScaleDtype::F32),
        )
    }

    fn signal(dtype: DType) -> TensorType {
        TensorType::propagating(dtype)
    }

    fn param(dtype: DType) -> TensorType {
        TensorType::inhibiting(dtype)
    }

    fn combine(inputs: &[TensorType]) -> TensorType {
        TensorType::combine(inputs.iter().copied(), DTypeResolution::Widen).unwrap()
    }

    #[test]
    fn a_weight_never_lends_its_format() {
        assert_eq!(combine(&[signal(mxfp8()), param(nvfp4())]), signal(mxfp8()));
    }

    #[test]
    fn a_quantized_signal_survives_an_f32_master_weight() {
        assert_eq!(
            combine(&[signal(mxfp8()), param(DType::F32)]),
            signal(mxfp8())
        );
    }

    #[test]
    fn a_bf16_signal_survives_an_f32_master_weight() {
        assert_eq!(
            combine(&[signal(DType::BF16), param(DType::F32)]),
            signal(DType::BF16)
        );
    }

    #[test]
    fn a_bf16_signal_survives_an_f32_bias() {
        assert_eq!(
            combine(&[param(DType::F32), signal(DType::BF16)]),
            signal(DType::BF16)
        );
    }

    #[test]
    fn a_float_residual_wins_over_a_dequantized_branch() {
        assert_eq!(
            combine(&[signal(DType::BF16), signal(DType::BF16)]),
            signal(DType::BF16)
        );
    }

    #[test]
    fn the_wider_of_two_quantized_activations_wins() {
        assert_eq!(
            combine(&[signal(nvfp4()), signal(mxfp8())]),
            signal(mxfp8())
        );
    }

    #[test]
    fn an_f32_state_wins_over_a_bf16_update() {
        assert_eq!(
            combine(&[signal(DType::F32), signal(DType::BF16)]),
            signal(DType::F32)
        );
    }

    #[test]
    fn floats_as_wide_promote_to_f32() {
        assert_eq!(
            combine(&[signal(DType::F16), signal(DType::BF16)]),
            signal(DType::F32)
        );
    }

    #[test]
    fn parameters_alone_inhibit() {
        assert_eq!(
            combine(&[param(DType::F32), param(DType::BF16)]),
            param(DType::F32)
        );
    }

    #[test]
    fn one_input_keeps_its_type() {
        assert_eq!(combine(&[param(nvfp4())]), param(nvfp4()));
        assert_eq!(combine(&[signal(DType::F16)]), signal(DType::F16));
    }

    #[test]
    fn a_transposed_operand_is_the_same_format() {
        let DType::QFloat(mut transposed) = mxfp8() else {
            unreachable!()
        };
        transposed.swap_dims(2, 0, 1);
        let out = combine(&[signal(mxfp8()), signal(DType::QFloat(transposed))]);
        assert_eq!(out, signal(mxfp8()));
        // Alone, it comes out along the innermost axis.
        assert_eq!(
            combine(&[signal(DType::QFloat(transposed))]),
            signal(mxfp8())
        );
    }

    #[test]
    fn a_tile_transposed_is_the_same_format() {
        let tile = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block([1, 2, 16], ScaleDtype::F32);
        let mut transposed = tile;
        transposed.swap_dims(3, 0, 2);
        let out = combine(&[
            signal(DType::QFloat(tile)),
            signal(DType::QFloat(transposed)),
        ]);
        assert_eq!(out, signal(DType::QFloat(tile)));
        assert!(out.same_format(DType::QFloat(transposed)));
    }

    #[test]
    fn strict_resolution_refuses_a_widening_tie() {
        let out = TensorType::combine(
            [signal(DType::F32), signal(DType::BF16)],
            DTypeResolution::Strict,
        );
        assert_eq!(
            out,
            Err(DTypeConflict {
                lhs: DType::F32,
                rhs: DType::BF16
            })
        );
    }

    #[test]
    fn strict_resolution_lets_a_parameter_meet_the_signal() {
        let out = TensorType::combine(
            [signal(mxfp8()), param(DType::F32), param(nvfp4())],
            DTypeResolution::Strict,
        );
        assert_eq!(out, Ok(signal(mxfp8())));
    }
}
