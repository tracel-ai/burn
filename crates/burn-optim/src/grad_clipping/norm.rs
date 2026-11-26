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
    pub(crate) fn of<const D: usize>(grad: Tensor<D>) -> Self {
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

    pub(crate) fn into_device(self, device: &Device) -> Self {
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

    /// Two multiplications by this factor apply the coefficient. Its square can underflow
    /// on an F32 device even when the scaled gradient is representable there.
    pub(crate) fn sqrt_scale(self, limit: f32) -> Tensor<1> {
        if limit == f32::INFINITY {
            return self.peak.ones_like();
        }
        let length = self.squares.sqrt();
        let limit_tensor = self.peak.full_like(limit);
        let sqrt_scale =
            limit_tensor.clone().sqrt() / (self.peak.clone().sqrt() * length.clone().sqrt());
        let within = self.peak.clone().lower_equal(limit_tensor / length);
        sqrt_scale
            .clamp_max(1.)
            .mask_fill(within, 1.)
            .mask_fill(self.peak.equal_elem(0.), 1.)
    }
}

pub(crate) fn scale_gradient<const D: usize>(
    grad: Tensor<D>,
    sqrt_scale: Tensor<1>,
    dtype: DType,
) -> Tensor<D> {
    let sqrt_scale = sqrt_scale.unsqueeze();
    let scaled = grad.cast(sqrt_scale.dtype()) * sqrt_scale.clone() * sqrt_scale;
    scaled.cast(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_norm_scale_survives_transfer_to_f32() {
        let device = Device::default();
        if !device.supports_dtype(DType::F64) {
            return;
        }
        let large = Tensor::<1>::from_data([1e50f64], (&device, DType::F64));
        let small = Tensor::<1>::from_floats([1e15], &device);
        let norm = L2Norm::combine(alloc::vec![L2Norm::of(large), L2Norm::of(small.clone())]);
        let sqrt_scale = norm.sqrt_scale(1.).cast(DType::F32);
        let actual = scale_gradient(small, sqrt_scale, DType::F32).into_scalar::<f32>();
        assert!(
            (actual / 1e-35 - 1.).abs() < 1e-5,
            "scaled gradient: {actual}"
        );
    }
}
