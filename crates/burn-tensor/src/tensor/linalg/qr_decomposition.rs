use alloc::vec::Vec;

use crate::{
    DType, Element, ElementConversion, backend::Backend, cast::ToElement, linalg::outer, s,
    tensor::Tensor,
};

struct Householder<B: Backend> {
    v: Tensor<B, 1>,
    tau: B::FloatElem,
}

fn eps_for_dtype(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 1e-3,
        DType::BF16 => 1e-2,
        DType::F32 => 1e-7,
        DType::F64 => 1e-15,
        _ => 1e-7,
    }
}

/// Performs QR decomposition of a matrix using Householder reflections.
///
/// The input matrix `A` is factored into `Q` and `R` such that `A = Q * R`,
/// where `Q` has orthonormal columns and `R` is upper trapezoidal.
///
/// # Returns
///
/// A tuple containing:
/// - `Q`: a matrix of shape `[m, k]`
/// - `R`: a matrix of shape `[k, n]`
///
/// where `m` and `n` are the input dimensions and `k = min(m, n)`.
pub fn qr_decomposition<B: Backend>(tensor: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let device = tensor.device();
    let [m, n] = tensor.shape().dims::<2>();
    let k_max = m.min(n);

    if k_max == 0 {
        let q = Tensor::<B, 2>::zeros([m, k_max], &device);
        let r = Tensor::<B, 2>::zeros([k_max, n], &device);
        return (q, r);
    }

    let mut r = tensor;
    // Store Householder vectors to build Q after R is formed.
    let mut reflectors: Vec<Option<Householder<B>>> = Vec::with_capacity(k_max);
    let eps_base = eps_for_dtype(<B::FloatElem as Element>::dtype());

    for k in 0..k_max {
        let r_sub = r.clone().slice(s![k.., k..]);
        // Current column segment to be zeroed below the diagonal.
        let x = r_sub.clone().slice(s![.., 0..1]).squeeze_dim(1);
        let rows = m - k;
        let x0 = x.clone().slice(s![0]);
        let x0_scalar = x0.clone().into_scalar().to_f64();
        let xnorm = if rows > 1 {
            x.clone()
                .slice(s![1..])
                .square()
                .sum()
                .sqrt()
                .into_scalar()
                .to_f64()
        } else {
            0.0
        };
        let scale = x0_scalar.abs().max(xnorm).max(1.0);
        let eps = eps_base * scale;
        if xnorm <= eps {
            reflectors.push(None);
            continue;
        }

        // Choose sign to avoid cancellation in beta.
        let sign = if x0_scalar >= 0.0 { 1.0 } else { -1.0 };
        let norm = (x0_scalar * x0_scalar + xnorm * xnorm).sqrt();
        let beta = -sign * norm;
        let denom = x0_scalar - beta;
        if denom.abs() <= eps || !beta.is_finite() {
            reflectors.push(None);
            continue;
        }
        let tau_scalar = (beta - x0_scalar) / beta;
        let tau = <B::FloatElem as ElementConversion>::from_elem(tau_scalar);
        let mut v = x.mul_scalar(1.0 / denom);
        let v0 = x0.clone().mul_scalar(0.0).add_scalar(1.0);
        v = v.slice_assign(s![0], v0);

        // w = R^T * v for the rank-1 update.
        let w = (r_sub.clone().transpose() * v.clone().unsqueeze_dim::<2>(0))
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        // R = R - tau * v * w^T
        let update = outer::<B, 1, 2, _>(v.clone(), w).mul_scalar(tau);
        let r_sub = r_sub - update;
        r = r.slice_assign(s![k.., k..], r_sub);

        reflectors.push(Some(Householder { v, tau }));
    }

    // Start with identity, then apply reflectors in reverse order.
    let mut q = Tensor::<B, 2>::eye(m, &device);
    if k_max < m {
        q = q.slice(s![.., 0..k_max]);
    }

    for k in (0..k_max).rev() {
        let Some(reflector) = reflectors.get_mut(k).and_then(|r| r.take()) else {
            continue;
        };

        let v = reflector.v;
        let tau = reflector.tau;

        let q_sub = q.clone().slice(s![k.., ..]);
        // Apply reflector: Q = Q - tau * v * (Q^T v)^T
        let wq = (q_sub.clone().transpose() * v.clone().unsqueeze_dim::<2>(0))
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let update_q = outer::<B, 1, 2, _>(v, wq).mul_scalar(tau);
        let q_sub = q_sub - update_q;
        q = q.slice_assign(s![k.., ..], q_sub);
    }

    let r = r.slice(s![0..k_max, ..]);
    (q, r)
}
