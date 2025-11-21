use crate::{DType, bf16, element::Element, f16};
use alloc::format;
use alloc::string::String;
use num_traits::{Float, ToPrimitive};

use super::TensorData;

/// The tolerance used to compare to floating point numbers.
///
/// Generally, two numbers `x` and `y` are approximately equal if
///
/// ```text
/// |x - y| < max(R * (|x + y|), A)
/// ```
///
/// where `R` is the relative tolerance and `A` is the absolute tolerance.
///
///
/// The most common way to initialize this struct is to use `Tolerance::<F>::default()`.
/// In that case, the relative and absolute tolerances are computed using an heuristic based
/// on the EPSILON and MIN_POSITIVE values of the given floating point type `F`.
///
/// Another common initialization is `Tolerance::<F>::rel_abs(1e-4, 1e-5).set_half_precision_relative(1e-2)`.
/// This will use a sane default to manage values too close to 0.0 and
/// use different relative tolerances depending on the floating point precision.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance<F> {
    relative: F,
    absolute: F,
}

impl<F: Float> Default for Tolerance<F> {
    fn default() -> Self {
        Self::balanced()
    }
}

impl<F: Float> Tolerance<F> {
    /// Create a tolerance with strict precision setting.
    pub fn strict() -> Self {
        Self {
            relative: F::from(0.00).unwrap(),
            absolute: F::from(64).unwrap() * F::min_positive_value(),
        }
    }
    /// Create a tolerance with balanced precision setting.
    pub fn balanced() -> Self {
        Self {
            relative: F::from(0.005).unwrap(), // 0.5%
            absolute: F::from(1e-5).unwrap(),
        }
    }

    /// Create a tolerance with permissive precision setting.
    pub fn permissive() -> Self {
        Self {
            relative: F::from(0.01).unwrap(), // 1.0%
            absolute: F::from(0.01).unwrap(),
        }
    }
    /// When comparing two numbers, this uses both the relative and absolute differences.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < max(R * (|x + y|), A)
    /// ```
    ///
    /// where `R` is the `relative` tolerance and `A` is the `absolute` tolerance.
    pub fn rel_abs<FF: ToPrimitive>(relative: FF, absolute: FF) -> Self {
        let relative = Self::check_relative(relative);
        let absolute = Self::check_absolute(absolute);

        Self { relative, absolute }
    }

    /// When comparing two numbers, this uses only the relative difference.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < R * max(|x|, |y|)
    /// ```
    ///
    /// where `R` is the relative `tolerance`.
    pub fn relative<FF: ToPrimitive>(tolerance: FF) -> Self {
        let relative = Self::check_relative(tolerance);

        Self {
            relative,
            absolute: F::from(0.0).unwrap(),
        }
    }

    /// When comparing two numbers, this uses only the absolute difference.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < A
    /// ```
    ///
    /// where `A` is the absolute `tolerance`.
    pub fn absolute<FF: ToPrimitive>(tolerance: FF) -> Self {
        let absolute = Self::check_absolute(tolerance);

        Self {
            relative: F::from(0.0).unwrap(),
            absolute,
        }
    }

    /// Change the relative tolerance to the given one.
    pub fn set_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        self.relative = Self::check_relative(tolerance);
        self
    }

    /// Change the relative tolerance to the given one only if `F` is half precision.
    pub fn set_half_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 2 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the relative tolerance to the given one only if `F` is single precision.
    pub fn set_single_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 4 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the relative tolerance to the given one only if `F` is double precision.
    pub fn set_double_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 8 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one.
    pub fn set_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        self.absolute = Self::check_absolute(tolerance);
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is half precision.
    pub fn set_half_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 2 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is single precision.
    pub fn set_single_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 4 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is double precision.
    pub fn set_double_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 8 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Checks if `x` and `y` are approximately equal given the tolerance.
    pub fn approx_eq(&self, x: F, y: F) -> bool {
        // See the accepted answer here
        // https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison

        // This also handles the case where both a and b are infinity so that we don't need
        // to manage it in the rest of the function.
        if x == y {
            return true;
        }

        let diff = (x - y).abs();
        let max = F::max(x.abs(), y.abs());

        diff < self.absolute.max(self.relative * max)
    }

    fn check_relative<FF: ToPrimitive>(tolerance: FF) -> F {
        let tolerance = F::from(tolerance).unwrap();
        assert!(tolerance <= F::one());
        tolerance
    }

    fn check_absolute<FF: ToPrimitive>(tolerance: FF) -> F {
        let tolerance = F::from(tolerance).unwrap();
        assert!(tolerance >= F::zero());
        tolerance
    }
}

impl TensorData {
    /// Asserts the data is equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `strict` - If true, the data types must the be same.
    ///   Otherwise, the comparison is done in the current data type.
    ///
    /// # Panics
    ///
    /// Panics if the data is not equal.
    #[track_caller]
    pub fn assert_eq(&self, other: &Self, strict: bool) {
        if strict {
            assert_eq!(
                self.dtype, other.dtype,
                "Data types differ ({:?} != {:?})",
                self.dtype, other.dtype
            );
        }

        match self.dtype {
            DType::F64 => self.assert_eq_elem::<f64>(other),
            DType::F32 | DType::Flex32 => self.assert_eq_elem::<f32>(other),
            DType::F16 => self.assert_eq_elem::<f16>(other),
            DType::BF16 => self.assert_eq_elem::<bf16>(other),
            DType::I64 => self.assert_eq_elem::<i64>(other),
            DType::I32 => self.assert_eq_elem::<i32>(other),
            DType::I16 => self.assert_eq_elem::<i16>(other),
            DType::I8 => self.assert_eq_elem::<i8>(other),
            DType::U64 => self.assert_eq_elem::<u64>(other),
            DType::U32 => self.assert_eq_elem::<u32>(other),
            DType::U16 => self.assert_eq_elem::<u16>(other),
            DType::U8 => self.assert_eq_elem::<u8>(other),
            DType::Bool => self.assert_eq_elem::<bool>(other),
            DType::QFloat(q) => {
                // Strict or not, it doesn't make sense to compare quantized data to not quantized data for equality
                let q_other = if let DType::QFloat(q_other) = other.dtype {
                    q_other
                } else {
                    panic!("Quantized data differs from other not quantized data")
                };

                // Data equality mostly depends on input quantization type, but we also check level
                if q.value == q_other.value && q.level == q_other.level {
                    self.assert_eq_elem::<i8>(other)
                } else {
                    panic!("Quantization schemes differ ({q:?} != {q_other:?})")
                }
            }
        }
    }

    #[track_caller]
    fn assert_eq_elem<E: Element>(&self, other: &Self) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let mut num_diff = 0;
        let max_num_diff = 5;
        for (i, (a, b)) in self.iter::<E>().zip(other.iter::<E>()).enumerate() {
            if a.cmp(&b).is_ne() {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    message += format!("\n  => Position {i}: {a} != {b}").as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - max_num_diff).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not eq:{message}");
        }
    }

    /// Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `tolerance` - The tolerance of the comparison.
    ///
    /// # Panics
    ///
    /// Panics if the data is not approximately equal.
    #[track_caller]
    pub fn assert_approx_eq<F: Float + Element>(&self, other: &Self, tolerance: Tolerance<F>) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let iter = self.iter::<F>().zip(other.iter::<F>());

        let mut num_diff = 0;
        let max_num_diff = 5;

        for (i, (a, b)) in iter.enumerate() {
            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf =
                a.is_infinite() && b.is_infinite() && ((a > F::zero()) == (b > F::zero()));

            if both_nan || both_inf {
                continue;
            }

            if !tolerance.approx_eq(F::from(a).unwrap(), F::from(b).unwrap()) {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    let diff_abs = ToPrimitive::to_f64(&(a - b).abs()).unwrap();
                    let max = F::max(a.abs(), b.abs());
                    let diff_rel = diff_abs / ToPrimitive::to_f64(&max).unwrap();

                    let tol_rel = ToPrimitive::to_f64(&tolerance.relative).unwrap();
                    let tol_abs = ToPrimitive::to_f64(&tolerance.absolute).unwrap();

                    message += format!(
                        "\n  => Position {i}: {a} != {b}\n     diff (rel = {diff_rel:+.2e}, abs = {diff_abs:+.2e}), tol (rel = {tol_rel:+.2e}, abs = {tol_abs:+.2e})"
                    )
                    .as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - 5).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not approx eq:{message}");
        }
    }

    /// Asserts each value is within a given range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively below
    /// and exclusively above (`start..end`).
    pub fn assert_within_range<E: Element>(&self, range: core::ops::Range<E>) {
        for elem in self.iter::<E>() {
            if elem.cmp(&range.start).is_lt() || elem.cmp(&range.end).is_ge() {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }

    /// Asserts each value is within a given inclusive range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively (`start..=end`).
    pub fn assert_within_range_inclusive<E: Element>(&self, range: core::ops::RangeInclusive<E>) {
        let start = range.start();
        let end = range.end();

        for elem in self.iter::<E>() {
            if elem.cmp(start).is_lt() || elem.cmp(end).is_gt() {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_assert_appox_eq_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.03, 5.0, 6.0]]);

        data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(3e-2));
        data1.assert_approx_eq::<f16>(&data2, Tolerance::absolute(3e-2));
    }

    #[test]
    #[should_panic]
    fn should_assert_approx_eq_above_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.031, 5.0, 6.0]]);

        data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(1e-2));
    }

    #[test]
    #[should_panic]
    fn should_assert_approx_eq_check_shape() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0, 7.0]]);
        let data2 = TensorData::from([[3.0, 5.0, 6.0]]);

        data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(1e-2));
    }
}
