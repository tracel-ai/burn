use crate::tensor::{Tensor, backend::Backend};

/// Classifier-free guidance: `uncond + s * (cond - uncond)`.
pub fn cfg<B: Backend, const D: usize>(
    cond: Tensor<B, D>,
    uncond: Tensor<B, D>,
    strength: f32,
) -> Tensor<B, D> {
    uncond.clone() + (cond - uncond).mul_scalar(strength)
}

/// Double-condition guidance used when splitting signals (e.g., text vs. lyric).
///
/// `(1 - g_text) * uncond + (g_text - g_lyric) * only_text + g_lyric * cond`.
pub fn cfg_double<B: Backend, const D: usize>(
    cond: Tensor<B, D>,
    uncond: Tensor<B, D>,
    only_text: Tensor<B, D>,
    guidance_scale_text: f32,
    guidance_scale_lyric: f32,
) -> Tensor<B, D> {
    uncond
        .mul_scalar(1.0 - guidance_scale_text)
        + only_text.mul_scalar(guidance_scale_text - guidance_scale_lyric)
        + cond.mul_scalar(guidance_scale_lyric)
}

/// Momentum buffer for APG-like guidance.
#[derive(Debug, Clone)]
pub struct MomentumBuffer<B: Backend, const D: usize> {
    momentum: f32,
    state: Option<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> MomentumBuffer<B, D> {
    /// Create a new momentum buffer.
    pub fn new(momentum: f32) -> Self {
        Self { momentum, state: None }
    }

    /// Update internal state with a new value.
    pub fn update(&mut self, value: &Tensor<B, D>) {
        match &self.state {
            Some(s) => {
                self.state = Some(value.clone() + s.clone().mul_scalar(self.momentum));
            }
            None => self.state = Some(value.clone()),
        }
    }

    /// Get current state (if any).
    pub fn get(&self) -> Option<Tensor<B, D>> {
        self.state.clone()
    }
}

/// APG-like guidance: project update with momentum and optional norm stabilization.
pub fn apg<B: Backend, const D: usize>(
    pred_cond: Tensor<B, D>,
    pred_uncond: Tensor<B, D>,
    guidance_scale: f32,
    buffer: Option<&mut MomentumBuffer<B, D>>,
    norm_eps: Option<f32>,
) -> Tensor<B, D> {
    let mut diff = pred_cond.clone() - pred_uncond;
    if let Some(buf) = buffer {
        buf.update(&diff);
        if let Some(state) = buf.get() {
            diff = state;
        }
    }
    if let Some(eps) = norm_eps {
        // Stabilize update with L2 over all elements.
        let denom = diff.clone().powi_scalar(2).mean().sqrt().add_scalar(eps);
        diff = diff / denom.unsqueeze::<D>();
    }
    pred_cond + diff.mul_scalar(guidance_scale - 1.0)
}

/// Zero-star variant: `alpha = <pos, neg> / ||neg||^2`; `out = alpha * uncond + s*(cond - alpha*uncond)`.
pub fn cfg_zero_star<B: Backend, const D: usize>(
    cond: Tensor<B, D>,
    uncond: Tensor<B, D>,
    guidance_scale: f32,
) -> Tensor<B, D> {
    // Flatten last all dims -> [N]
    let numel = cond.shape().num_elements();
    let c = cond.clone().reshape([numel]);
    let u = uncond.clone().reshape([numel]);
    let dot = (c.clone() * u.clone()).sum();
    let norm2 = (u.clone().powi_scalar(2)).sum().add_scalar(1e-8);
    let alpha = dot / norm2;
    let base = uncond.mul_scalar(alpha.into_scalar());
    base.clone() + (cond - base).mul_scalar(guidance_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::{Distribution, Tensor};

    #[test]
    fn cfg_shapes() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 3>::random([2, 4, 8], Distribution::Default, &device);
        let b = Tensor::<TestBackend, 3>::random([2, 4, 8], Distribution::Default, &device);
        let y = cfg(a, b, 1.5);
        assert_eq!(y.dims(), [2, 4, 8]);
    }

    #[test]
    fn apg_runs() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 2>::random([3, 5], Distribution::Default, &device);
        let b = Tensor::<TestBackend, 2>::random([3, 5], Distribution::Default, &device);
        let mut buf = MomentumBuffer::<TestBackend, 2>::new(-0.75);
        let y = apg(a, b, 8.0, Some(&mut buf), Some(1e-6));
        assert_eq!(y.dims(), [3, 5]);
    }
}
