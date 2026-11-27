use alloc::vec::Vec;
use burn_core::tensor::{DType, Device, Tensor};

/// The squared norm is `peak² * squares`. Keeping the factors separate also represents
/// a norm outside the gradient dtype's range.
pub(crate) struct L2Norm {
    peak: Tensor<1>,
    squares: Tensor<1>,
}

pub(crate) fn norm_precision<const D: usize>(grad: Tensor<D>) -> Tensor<D> {
    match grad.dtype() {
        DType::F64 | DType::F32 => grad,
        dtype @ DType::QFloat(_) => {
            panic!("GlobalNorm requires floating-point gradients, got {dtype:?}")
        }
        _ => grad.cast(DType::F32),
    }
}

impl L2Norm {
    pub(crate) fn compute<const D: usize>(grad: Tensor<D>) -> Self {
        let grad = norm_precision(grad);
        if grad.shape().num_elements() == 0 {
            let zero = grad.sum();
            return Self {
                peak: zero.clone(),
                squares: zero,
            };
        }
        let peak = grad.clone().abs().max();
        // A zero tensor contributes zero. An infinite entry should still give an infinite
        // norm, without introducing a NaN from dividing infinity by itself.
        let divisor = peak
            .clone()
            .mask_fill(peak.clone().equal_elem(0.), 1.)
            .mask_fill(peak.clone().is_inf(), 1.);
        let squares = (grad / divisor.unsqueeze()).square().sum();
        Self { peak, squares }
    }

    pub(crate) fn dtype(&self) -> DType {
        self.peak.dtype()
    }

    #[allow(clippy::wrong_self_convention)] // Match Tensor::to_device's consuming receiver.
    pub(crate) fn to_device(self, device: &Device) -> Self {
        Self {
            peak: self.peak.to_device(device),
            squares: self.squares.to_device(device),
        }
    }

    /// Combine nonempty partial reductions at a common magnitude before adding their squares.
    pub(crate) fn combine(mut norms: Vec<Self>) -> Self {
        if norms.len() == 1 {
            return norms.pop().unwrap();
        }
        let dtype = if norms.iter().any(|norm| norm.dtype() == DType::F64) {
            DType::F64
        } else {
            DType::F32
        };
        let (peaks, squares) = norms
            .into_iter()
            .map(|norm| (norm.peak.cast(dtype), norm.squares.cast(dtype)))
            .unzip();
        let peaks = Tensor::cat(peaks, 0);
        let peak = peaks.clone().max();
        let relative = (peaks.clone() / peak.clone()).mask_fill(peaks.equal(peak.clone()), 1.);
        let squares = (Tensor::cat(squares, 0) * relative.square()).sum();
        Self { peak, squares }
    }

    pub(crate) fn clip_coefficient(self, limit: f32) -> Tensor<1> {
        if limit == f32::INFINITY {
            return self.peak.ones_like();
        }
        let zero_norm = self.peak.clone().equal_elem(0.);
        let limit = self.peak.full_like(limit);
        // Forming peak * sqrt(squares) can overflow even when the coefficient is representable.
        (limit / self.peak / self.squares.sqrt())
            .clamp_max(1.)
            .mask_fill(zero_norm, 1.)
    }
}

pub(crate) fn scale_gradient<const D: usize>(
    grad: Tensor<D>,
    coefficient: Tensor<1>,
    output_dtype: DType,
) -> Tensor<D> {
    (grad.cast(coefficient.dtype()) * coefficient.unsqueeze()).cast(output_dtype)
}
