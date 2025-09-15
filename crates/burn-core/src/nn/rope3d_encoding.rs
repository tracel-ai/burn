use crate as burn;
use crate::config::Config;
use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::tensor::{Int, Tensor, backend::Backend};

/// Configuration for 3D rotary positional encoding over (F, H, W).
#[derive(Config, Debug)]
pub struct Rope3dEncodingConfig {
    /// Maximum frames (F) positions.
    pub max_f: usize,
    /// Maximum height (H) positions.
    pub max_h: usize,
    /// Maximum width (W) positions.
    pub max_w: usize,
    /// Head dimension (must be even).
    pub d_head: usize,
    /// Optional split of half-dim across (F, H, W). Sum must equal d_head/2.
    #[config(default = "None")]
    pub half_dim_split: Option<[usize; 3]>,
    /// Rotary scaling base (theta).
    #[config(default = "10000.0")]
    pub theta: f32,
}

impl Rope3dEncodingConfig {
    /// Initialize a new 3D rotary encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Rope3dEncoding<B> {
        assert!(self.d_head % 2 == 0, "d_head must be even");
        let half = self.d_head / 2;

        let split = if let Some([f, h, w]) = self.half_dim_split {
            assert!(f + h + w == half, "sum(split) must equal d_head/2");
            [f, h, w]
        } else {
            let base = half / 3;
            let rem = half - base * 2;
            [rem, base, base] // favor F axis if not divisible by 3
        };

        Rope3dEncoding::new(
            self.max_f, self.max_h, self.max_w, split, self.theta, device,
        )
    }
}

/// 3D rotary positional encoding.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Rope3dEncoding<B: Backend> {
    /// Precomputed [max_f, f_pairs, 2]
    freq_f: Tensor<B, 3>,
    /// Precomputed [max_h, h_pairs, 2]
    freq_h: Tensor<B, 3>,
    /// Precomputed [max_w, w_pairs, 2]
    freq_w: Tensor<B, 3>,
    /// Per-axis pair counts.
    split: [usize; 3],
}

impl<B: Backend> ModuleDisplay for Rope3dEncoding<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        let [f_pairs, h_pairs, w_pairs] = self.split;
        content
            .add("f_pairs", &f_pairs)
            .add("h_pairs", &h_pairs)
            .add("w_pairs", &w_pairs)
            .optional()
    }
}

impl<B: Backend> Rope3dEncoding<B> {
    fn new(
        max_f: usize,
        max_h: usize,
        max_w: usize,
        split: [usize; 3],
        theta: f32,
        device: &B::Device,
    ) -> Rope3dEncoding<B> {
        let [f_pairs, h_pairs, w_pairs] = split;
        let freq_f = Self::precompute_axis(max_f, f_pairs, theta, device);
        let freq_h = Self::precompute_axis(max_h, h_pairs, theta, device);
        let freq_w = Self::precompute_axis(max_w, w_pairs, theta, device);

        Rope3dEncoding {
            freq_f,
            freq_h,
            freq_w,
            split,
        }
    }

    /// Precompute cos/sin grid for an axis: [max_pos, pairs, 2].
    fn precompute_axis(
        max_pos: usize,
        pairs: usize,
        theta: f32,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        if pairs == 0 {
            return Tensor::<B, 3>::zeros([max_pos, 0, 2], device);
        }
        // Exponent indices 0..pairs
        let exponent = Tensor::<B, 1, Int>::arange(0..pairs as i64, device)
            .float()
            .div_scalar(pairs as f32 * 2.0);
        let base = exponent.mul_scalar(theta.ln()).exp().recip();

        let pos = Tensor::<B, 1, Int>::arange(0..max_pos as i64, device).float(); // [max_pos]
        let pos = pos.unsqueeze_dim::<2>(1); // [max_pos, 1]
        let base = base.unsqueeze_dim::<2>(0); // [1, pairs]
        let freqs = pos.matmul(base); // [max_pos, pairs]
        let cos = freqs.clone().cos();
        let sin = freqs.sin();
        let cos = cos.unsqueeze_dim::<3>(2);
        let sin = sin.unsqueeze_dim::<3>(2);
        Tensor::cat(vec![cos, sin], 2)
    }

    /// Apply 3D RoPE over tokens laid out as [B, S, Hh, Dh].
    /// grid_sizes = [F, H, W], start_frame offsets the F coordinate.
    pub fn apply(
        &self,
        x: Tensor<B, 4>,
        grid_sizes: [usize; 3],
        start_frame: usize,
    ) -> Tensor<B, 4> {
        let [b, s, n_heads, d_head] = x.dims();
        let half = d_head / 2;
        assert_eq!(d_head % 2, 0, "d_head must be even");
        let [f_pairs, h_pairs, w_pairs] = self.split;
        assert_eq!(
            f_pairs + h_pairs + w_pairs,
            half,
            "split must sum to d_head/2"
        );
        let [f, h, w] = grid_sizes;
        assert_eq!(f * h * w, s, "grid_sizes must match sequence length");

        // Build indices per token.
        let hw = h * w;
        let f_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &x.device())
            .float()
            .div_scalar(hw as f32)
            .floor()
            .int()
            .add_scalar(start_frame as i64)
            .clamp(0, self.freq_f.dims()[0] as i64 - 1);
        let h_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &x.device())
            .div_scalar(w as i64)
            .remainder_scalar(h as i64);
        let w_idx =
            Tensor::<B, 1, Int>::arange(0..s as i64, &x.device()).remainder_scalar(w as i64);

        // Gather cos/sin per token.
        let gather_axis = |table: &Tensor<B, 3>, idx: Tensor<B, 1, Int>| -> Tensor<B, 3> {
            // table: [max, pairs, 2], idx: [s]
            let max = table.dims()[0];
            let pairs = table.dims()[1];
            if pairs == 0 {
                return Tensor::<B, 3>::zeros([s, 0, 2], &x.device());
            }
            let idx_clamped = idx.clamp(0, max as i64 - 1).reshape([s, 1, 1]);
            let idx_tiled = idx_clamped.repeat_dim(1, pairs).repeat_dim(2, 2);
            table.clone().gather(0, idx_tiled)
        };

        let fc = gather_axis(&self.freq_f, f_idx);
        let hc = gather_axis(&self.freq_h, h_idx);
        let wc = gather_axis(&self.freq_w, w_idx);
        let freq = Tensor::cat(vec![fc, hc, wc], 1); // [s, half, 2]
        let freq = freq
            .unsqueeze_dim::<4>(2) // [s, half, 1, 2]
            .repeat_dim(2, 2) // [s, half, 2, 2]
            .reshape([s, d_head, 2]); // [s, d_head, 2]

        // Rotate: x -> [B*Hh, S, half, 2]; apply rotation; reshape back.
        let sign =
            Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &x.device());

        let x_rs = x
            .reshape([b * n_heads, s, d_head / 2, 2])
            .matmul(sign.unsqueeze())
            .reshape([b * n_heads, s, d_head, 2]);

        let out = x_rs
            * freq
                .clone()
                .unsqueeze_dim::<4>(0) // [1, s, d_head, 2]
                .repeat_dim(0, b * n_heads)
                .reshape([b * n_heads, s, d_head, 2]);

        out.sum_dim(3).reshape([b, s, n_heads, d_head])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::{Distribution, Tensor};

    #[test]
    fn rope3d_shapes_and_basic() {
        let device = Default::default();
        let d_head = 32;
        let enc = Rope3dEncodingConfig::new(8, 4, 4, d_head).init::<TestBackend>(&device);
        let x = Tensor::<TestBackend, 4>::random([2, 8, 4, d_head], Distribution::Default, &device);
        let y = enc.apply(x, [2, 2, 2], 0);
        assert_eq!(y.dims(), [2, 8, 4, d_head]);
    }
}
