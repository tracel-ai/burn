use alloc::vec;
use alloc::vec::Vec;

use burn_core as burn;

use crate::Initializer;
use crate::{Linear, LinearConfig};
use burn::config::Config;
use burn::instrument::compute;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, Param};
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::ops::PadMode;
use burn::tensor::{DType, Device, Distribution, Tensor, assert_shape};

/// Configuration to create a [Gated DeltaNet](GatedDeltaNet) layer using the
/// [init function](GatedDeltaNetConfig::init).
#[derive(Config, Debug)]
pub struct GatedDeltaNetConfig {
    /// The size of the input and output features.
    pub d_model: usize,
    /// The number of query/key heads.
    pub n_key_heads: usize,
    /// The number of value heads. Must be a multiple of `n_key_heads`; each key head is shared
    /// by `n_value_heads / n_key_heads` consecutive value heads.
    pub n_value_heads: usize,
    /// The size of each query/key head.
    pub d_key_head: usize,
    /// The size of each value head.
    pub d_value_head: usize,
    /// The kernel size of the depthwise causal convolution over the query, key and value.
    #[config(default = 4)]
    pub conv_kernel_size: usize,
    /// Epsilon of the gated RMS norm on the output.
    #[config(default = 1e-6)]
    pub epsilon: f64,
    /// Tokens per chunk of the delta rule. Longer chunks mean fewer sequential steps and larger
    /// matrix products.
    #[config(default = 64)]
    pub chunk_size: usize,
}

impl GatedDeltaNetConfig {
    /// Initialize a new [Gated DeltaNet](GatedDeltaNet) layer.
    ///
    /// # Panics
    ///
    /// Panics if `n_value_heads` is not a multiple of `n_key_heads`, or if `conv_kernel_size`
    /// or `chunk_size` is zero.
    pub fn init(&self, device: &Device) -> GatedDeltaNet {
        assert!(
            self.n_key_heads > 0 && self.n_value_heads.is_multiple_of(self.n_key_heads),
            "n_value_heads must be a multiple of n_key_heads"
        );
        assert!(
            self.conv_kernel_size > 0,
            "conv_kernel_size must be positive"
        );
        assert!(self.chunk_size > 0, "chunk_size must be positive");

        let key_dim = self.n_key_heads * self.d_key_head;
        let value_dim = self.n_value_heads * self.d_value_head;
        let conv_dim = 2 * key_dim + value_dim;
        let linear = |d_input, d_output| {
            LinearConfig::new(d_input, d_output)
                .with_bias(false)
                .init(device)
        };
        // PyTorch's default for a depthwise Conv1d: U(-1/sqrt(k), 1/sqrt(k)).
        let bound = 1.0 / num_traits::Float::sqrt(self.conv_kernel_size as f64);
        let conv = Initializer::Uniform {
            min: -bound,
            max: bound,
        };
        // The decay rate exp(a_log) starts in U(1, 16).
        let rate = Tensor::<1>::random(
            [self.n_value_heads],
            Distribution::Uniform(1.0, 16.0),
            device,
        );
        let a_log = Param::from_tensor(rate.log());

        GatedDeltaNet {
            in_proj_qkv: linear(self.d_model, conv_dim),
            in_proj_z: linear(self.d_model, value_dim),
            in_proj_b: linear(self.d_model, self.n_value_heads),
            in_proj_a: linear(self.d_model, self.n_value_heads),
            conv1d: conv.init([conv_dim, self.conv_kernel_size], device),
            dt_bias: Initializer::Ones.init([self.n_value_heads], device),
            a_log,
            norm: Initializer::Ones.init([self.d_value_head], device),
            out_proj: linear(value_dim, self.d_model),
            n_key_heads: self.n_key_heads,
            n_value_heads: self.n_value_heads,
            d_key_head: self.d_key_head,
            d_value_head: self.d_value_head,
            epsilon: self.epsilon,
            chunk_size: self.chunk_size,
        }
    }
}

/// The Gated DeltaNet layer, a linear attention with a fixed-size state.
///
/// Each value head keeps a `[d_key_head, d_value_head]` memory `S`. Per token:
///
/// ```text
/// S = S * exp(g)                  decay
/// S = S + k (beta * (v - S^T k))^T  delta-rule write
/// y = S^T q / sqrt(d_key_head)    read
/// ```
///
/// `q`, `k` and `v` come from one projection followed by a depthwise causal convolution and
/// SiLU, `q` and `k` are L2-normalized, `beta = sigmoid(in_proj_b(x))` and
/// `g = -exp(a_log) * softplus(in_proj_a(x) + dt_bias)`. The output is
/// `out_proj(rms_norm(y) * norm * silu(in_proj_z(x)))`.
///
/// The sequence is processed in chunks: inside a chunk the tokens are solved together with
/// matrix products, and only the state is carried from chunk to chunk.
///
/// Introduced in [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
/// and used by Qwen3.5 and Qwen3-Next. The chunked form follows Hugging Face Transformers' and
/// flash-linear-attention's. Field names match those checkpoints' `linear_attn` weights.
///
/// Should be created with [GatedDeltaNetConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct GatedDeltaNet {
    /// Projection to the query, key and value, side by side.
    pub in_proj_qkv: Linear,
    /// Projection to the output gate.
    pub in_proj_z: Linear,
    /// Projection to the write strength `beta`, one per value head.
    pub in_proj_b: Linear,
    /// Projection to the decay input, one per value head.
    pub in_proj_a: Linear,
    /// Depthwise causal convolution weights, `[2 * key_dim + value_dim, conv_kernel_size]`.
    pub conv1d: Param<Tensor<2>>,
    /// Bias added before the softplus of the decay, `[n_value_heads]`.
    pub dt_bias: Param<Tensor<1>>,
    /// Log of the decay rate, `[n_value_heads]`.
    pub a_log: Param<Tensor<1>>,
    /// Weight of the gated RMS norm, `[d_value_head]`.
    pub norm: Param<Tensor<1>>,
    /// Projection back to `d_model`.
    pub out_proj: Linear,
    /// The number of query/key heads.
    pub n_key_heads: usize,
    /// The number of value heads.
    pub n_value_heads: usize,
    /// The size of each query/key head.
    pub d_key_head: usize,
    /// The size of each value head.
    pub d_value_head: usize,
    /// Epsilon of the gated RMS norm.
    pub epsilon: f64,
    /// Tokens per chunk of the delta rule.
    pub chunk_size: usize,
}

/// What a [Gated DeltaNet](GatedDeltaNet) layer carries from one call to the next, so a
/// sequence can be fed in parts.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetState {
    /// The last `conv_kernel_size - 1` convolution inputs, `[batch_size, conv_dim, conv_kernel_size - 1]`.
    pub conv: Tensor<3>,
    /// The memory of each value head, `[batch_size, n_value_heads, d_key_head, d_value_head]`,
    /// kept in f32.
    pub recurrent: Tensor<4>,
}

impl GatedDeltaNet {
    /// Applies the forward pass on the input tensor.
    ///
    /// `state` is the state returned by a previous call on the preceding tokens, or `None` to
    /// start from an empty memory.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    ///
    /// # Panics
    ///
    /// Panics if the input is empty along `seq_length` or its last axis is not `d_model`.
    #[compute]
    pub fn forward(
        &self,
        input: Tensor<3>,
        state: Option<GatedDeltaNetState>,
    ) -> (Tensor<3>, GatedDeltaNetState) {
        let [d_model, _] = self.in_proj_qkv.weight.shape().dims();
        assert_shape!(input, [_, _, d_model]);
        let [batch_size, seq_length, _] = input.dims();
        let (nk, nv) = (self.n_key_heads, self.n_value_heads);
        let (dk, dv) = (self.d_key_head, self.d_value_head);
        let (key_dim, value_dim) = (nk * dk, nv * dv);
        let (conv_state, recurrent) = match state {
            Some(state) => (Some(state.conv), Some(state.recurrent)),
            None => (None, None),
        };

        let (qkv, conv) = self.causal_conv(self.in_proj_qkv.forward(input.clone()), conv_state);
        let q = qkv.clone().narrow(2, 0, key_dim);
        let k = qkv.clone().narrow(2, key_dim, key_dim);
        let v = qkv.narrow(2, 2 * key_dim, value_dim);
        let heads = |t: Tensor<3>, n, d| t.reshape([batch_size, seq_length, n, d]);
        let q = self.repeat_key_heads(l2_normalize(heads(q, nk, dk)));
        let k = self.repeat_key_heads(l2_normalize(heads(k, nk, dk)));
        let v = heads(v, nv, dv);

        let beta = sigmoid(self.in_proj_b.forward(input.clone()));
        let rate = self.a_log.val().exp().reshape([1, 1, nv]);
        let a = self.in_proj_a.forward(input.clone()) + self.dt_bias.val().reshape([1, 1, nv]);
        let g = (softplus(a, 1.0) * rate).neg();

        let (y, recurrent) = gated_delta_rule(
            q.swap_dims(1, 2),
            k.swap_dims(1, 2),
            v.swap_dims(1, 2),
            g.swap_dims(1, 2),
            beta.swap_dims(1, 2),
            recurrent,
            self.chunk_size,
        );
        let y = y.swap_dims(1, 2);

        let z = heads(self.in_proj_z.forward(input), nv, dv);
        let y = rms_normalize(y, self.epsilon) * self.norm.val().reshape([1, 1, 1, dv]) * silu(z);
        let output = self
            .out_proj
            .forward(y.reshape([batch_size, seq_length, value_dim]));
        (output, GatedDeltaNetState { conv, recurrent })
    }

    /// Depthwise causal convolution and SiLU over `[batch_size, seq_length, channels]`, preceded
    /// by the carried inputs (zeros when starting fresh). Returns the output and the last
    /// `kernel - 1` inputs.
    fn causal_conv(&self, x: Tensor<3>, state: Option<Tensor<3>>) -> (Tensor<3>, Tensor<3>) {
        let [batch_size, seq_length, channels] = x.dims();
        let weight = self.conv1d.val();
        let [_, kernel] = weight.dims();
        let x = x.swap_dims(1, 2);
        let past = state.unwrap_or_else(|| {
            Tensor::zeros([batch_size, channels, kernel - 1], (&x.device(), x.dtype()))
        });
        let full = Tensor::cat(vec![past, x], 2);
        let mut output = full.clone().narrow(2, 0, seq_length)
            * weight.clone().narrow(1, 0, 1).reshape([1, channels, 1]);
        for j in 1..kernel {
            output = output
                + full.clone().narrow(2, j, seq_length)
                    * weight.clone().narrow(1, j, 1).reshape([1, channels, 1]);
        }
        let carry = full.narrow(2, seq_length, kernel - 1);
        (silu(output).swap_dims(1, 2), carry)
    }

    /// `[.., n_key_heads, d]` to `[.., n_value_heads, d]`, each key head repeated for its
    /// consecutive value heads.
    fn repeat_key_heads(&self, t: Tensor<4>) -> Tensor<4> {
        let repeats = self.n_value_heads / self.n_key_heads;
        if repeats == 1 {
            return t;
        }
        let [batch_size, seq_length, heads, d] = t.dims();
        t.unsqueeze_dim::<5>(3)
            .expand([batch_size, seq_length, heads, repeats, d])
            .reshape([batch_size, seq_length, heads * repeats, d])
    }
}

impl ModuleDisplay for GatedDeltaNet {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model, _] = self.in_proj_qkv.weight.shape().dims();
        content
            .add("d_model", &d_model)
            .add("n_key_heads", &self.n_key_heads)
            .add("n_value_heads", &self.n_value_heads)
            .add("d_key_head", &self.d_key_head)
            .add("d_value_head", &self.d_value_head)
            .add("chunk_size", &self.chunk_size)
            .optional()
    }
}

/// `x / sqrt(sum(x^2) + 1e-6)` over the last axis, the query/key normalization of the
/// reference implementation.
fn l2_normalize(x: Tensor<4>) -> Tensor<4> {
    let norm = (x.clone().square().sum_dim(3) + 1e-6).sqrt();
    x / norm
}

fn rms_normalize(x: Tensor<4>, epsilon: f64) -> Tensor<4> {
    let dtype = x.dtype();
    let rms = (x.clone().cast(DType::F32).square().mean_dim(3) + epsilon).sqrt();
    x / rms.cast(dtype)
}

/// The gated delta rule over a whole sequence, in chunks of `chunk_size` tokens.
///
/// - `q`, `k`: `[batch, heads, seq, d_key]`, L2-normalized; `q` is scaled by `1 / sqrt(d_key)` here.
/// - `v`: `[batch, heads, seq, d_value]`
/// - `g`: `[batch, heads, seq]`, log of the per-token decay
/// - `beta`: `[batch, heads, seq]`, write strength
/// - `state`: `[batch, heads, d_key, d_value]` or `None` for zeros
///
/// Returns the output in the dtype of `v` and the final state in f32. The rule itself runs in
/// f32: the state accumulates over the whole sequence.
fn gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    state: Option<Tensor<4>>,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>) {
    let [batch, heads, seq, dk] = q.dims();
    let dv = v.dims()[3];
    assert!(seq > 0, "the sequence must not be empty");
    let dtype = v.dtype();
    let f32 = |t: Tensor<4>| t.cast(DType::F32);
    let (q, k, v) = (f32(q), f32(k), f32(v));
    let (g, beta) = (g.cast(DType::F32), beta.cast(DType::F32));
    let bh = batch * heads;

    // A sequence shorter than a chunk is one shorter chunk. The tail is padded with g = 0 and
    // beta = 0: padded tokens write nothing and do not decay, so they change neither the real
    // outputs nor the final state.
    let c = chunk_size.min(seq);
    let n = seq.div_ceil(c);
    let pad = n * c - seq;
    let pad4 = |x: Tensor<4>| match pad {
        0 => x,
        _ => x.pad([(0, 0), (0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0)),
    };
    let pad3 = |x: Tensor<3>| match pad {
        0 => x,
        _ => x.pad([(0, 0), (0, 0), (0, pad)], PadMode::Constant(0.0)),
    };

    let rows = bh * n;
    let scale = 1.0 / num_traits::Float::sqrt(dk as f64);
    let q = pad4(q.mul_scalar(scale)).reshape([rows, c, dk]);
    let k = pad4(k).reshape([rows, c, dk]);
    let v = pad4(v).reshape([rows, c, dv]);
    let g_cum = pad3(g).reshape([rows, c]).cumsum(1);
    let beta = pad3(beta).reshape([rows, c, 1]);
    let k_beta = k.clone() * beta.clone();
    let v_beta = v * beta;

    // decay[i, j] = exp(g_cum[i] - g_cum[j]) for i >= j. The exponent is zeroed above the
    // diagonal before exp so those entries cannot overflow; they are masked afterwards.
    let decay = (g_cum.clone().unsqueeze_dim::<3>(2) - g_cum.clone().unsqueeze_dim::<3>(1))
        .tril(0)
        .exp();
    let k_t = k.clone().swap_dims(1, 2);
    let a = (k_beta.clone().matmul(k_t.clone()) * decay.clone()).tril(-1);
    let qk = (q.clone().matmul(k_t) * decay).tril(0);

    let t_inv = unit_lower_inverse(a);
    let g_exp = g_cum.clone().exp().unsqueeze_dim::<3>(2);
    let v_fixed = t_inv.clone().matmul(v_beta);
    let k_decayed = t_inv.matmul(k_beta * g_exp.clone());
    let q_decayed = q * g_exp;
    let g_last = g_cum.clone().narrow(1, c - 1, 1);
    let k_to_end = k * (g_last.clone() - g_cum).exp().unsqueeze_dim::<3>(2);
    let g_last_exp = g_last.exp().reshape([bh, n, 1, 1]);

    let v_fixed = v_fixed.reshape([bh, n, c, dv]);
    let k_decayed = k_decayed.reshape([bh, n, c, dk]);
    let qk = qk.reshape([bh, n, c, c]);
    let q_decayed = q_decayed.reshape([bh, n, c, dk]);
    let k_to_end_t = k_to_end.reshape([bh, n, c, dk]).swap_dims(2, 3);

    let mut st = match state {
        Some(state) => {
            assert_shape!(state, [batch, heads, dk, dv]);
            state.cast(DType::F32).reshape([bh, dk, dv])
        }
        None => Tensor::zeros([bh, dk, dv], (&v_fixed.device(), DType::F32)),
    };
    let chunk = |x: &Tensor<4>, i: usize| -> Tensor<3> {
        let [a, _, r, cols] = x.dims();
        x.clone().narrow(1, i, 1).reshape([a, r, cols])
    };
    // Only the state carry between chunks is sequential.
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let v_new = chunk(&v_fixed, i) - chunk(&k_decayed, i).matmul(st.clone());
        ys.push(chunk(&q_decayed, i).matmul(st.clone()) + chunk(&qk, i).matmul(v_new.clone()));
        let decay = g_last_exp.clone().narrow(1, i, 1).reshape([bh, 1, 1]);
        st = st * decay + chunk(&k_to_end_t, i).matmul(v_new);
    }
    let y = Tensor::cat(ys, 1)
        .narrow(1, 0, seq)
        .reshape([batch, heads, seq, dv]);
    (y.cast(dtype), st.reshape([batch, heads, dk, dv]))
}

/// Size of the diagonal blocks of [unit_lower_inverse].
const INVERSE_BLOCK: usize = 16;

/// `(I + A)^-1` for a batch of strictly lower-triangular `A`, `[rows, c, c]`, by blocked forward
/// substitution: the diagonal blocks row by row, then one block row at a time.
///
/// A power series would be shorter but is unstable: when the keys in a chunk are similar the
/// entries of `A` approach 1 and its powers grow far beyond f32 before they cancel.
fn unit_lower_inverse(a: Tensor<3>) -> Tensor<3> {
    let [rows, c, _] = a.dims();
    let b = INVERSE_BLOCK.min(c);
    let nb = c.div_ceil(b);
    let cp = nb * b;
    // Zero padding keeps A strictly lower; the top-left c x c of the padded inverse is the answer.
    let a = match cp - c {
        0 => a,
        p => a.pad([(0, 0), (0, p), (0, p)], PadMode::Constant(0.0)),
    };

    let diag = Tensor::cat(
        (0..nb)
            .map(|j| a.clone().narrow(1, j * b, b).narrow(2, j * b, b))
            .collect(),
        0,
    );
    let diag_inv = substitute(diag);
    let block = |j: usize| diag_inv.clone().narrow(0, j * rows, rows);

    let mut x = block(0);
    for i in 1..nb {
        let a_row = a.clone().narrow(1, i * b, b).narrow(2, 0, i * b);
        let x_ii = block(i);
        let left = x_ii.clone().matmul(a_row.matmul(x.clone())).neg();
        let zeros = Tensor::zeros([rows, i * b, b], (&left.device(), left.dtype()));
        let top = Tensor::cat(vec![x, zeros], 2);
        let bottom = Tensor::cat(vec![left, x_ii], 2);
        x = Tensor::cat(vec![top, bottom], 1);
    }
    match cp - c {
        0 => x,
        _ => x.narrow(1, 0, c).narrow(2, 0, c),
    }
}

/// Exact `(I + A)^-1` by forward substitution, one row per step: row `i` is
/// `e_i - A[i, :i] X[:i, :]`.
fn substitute(a: Tensor<3>) -> Tensor<3> {
    let [n, b, _] = a.dims();
    let eye = Tensor::<2>::eye(b, &a.device())
        .cast(a.dtype())
        .unsqueeze_dim::<3>(0)
        .expand([n, b, b]);
    let mut x = eye.clone().narrow(1, 0, 1);
    for i in 1..b {
        let row = eye.clone().narrow(1, i, 1)
            - a.clone().narrow(1, i, 1).narrow(2, 0, i).matmul(x.clone());
        x = Tensor::cat(vec![x, row], 1);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec;
    use burn::tensor::TensorData;
    #[cfg(not(feature = "std"))]
    use num_traits::Float;

    /// Deterministic inputs (splitmix64).
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Approximately standard normal (sum of 12 uniforms).
        fn normal(&mut self) -> f64 {
            (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
        }

        fn normals(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.normal()).collect()
        }
    }

    #[derive(Clone)]
    struct Case {
        heads: usize,
        seq: usize,
        dk: usize,
        dv: usize,
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        g: Vec<f64>,
        beta: Vec<f64>,
        state: Option<Vec<f64>>,
    }

    fn l2_normalized(mut x: Vec<f64>, d: usize) -> Vec<f64> {
        for row in x.chunks_mut(d) {
            let norm = (row.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt();
            row.iter_mut().for_each(|a| *a /= norm);
        }
        x
    }

    /// Random inputs. Weak decay keeps the chunk matrices close to 1, the hard case for the
    /// inverse; strong decay is what a trained layer produces.
    fn case(seed: u64, heads: usize, seq: usize, dk: usize, dv: usize, weak_decay: bool) -> Case {
        let mut r = Rng(seed);
        let n = heads * seq;
        let g = (0..n)
            .map(|_| match weak_decay {
                true => -0.02 * r.uniform(),
                false => -(1.0 + 15.0 * r.uniform()) * (1.0 + r.normal().exp()).ln(),
            })
            .collect();
        Case {
            heads,
            seq,
            dk,
            dv,
            q: l2_normalized(r.normals(n * dk), dk),
            k: l2_normalized(r.normals(n * dk), dk),
            v: r.normals(n * dv),
            g,
            beta: (0..n).map(|_| 1.0 / (1.0 + (-r.normal()).exp())).collect(),
            state: Some(r.normals(heads * dk * dv).iter().map(|x| 0.5 * x).collect()),
        }
    }

    /// The rule one token at a time in f64, sharing no code with the chunked version.
    fn reference(c: &Case) -> (Vec<f64>, Vec<f64>) {
        let (seq, dk, dv) = (c.seq, c.dk, c.dv);
        let scale = 1.0 / (dk as f64).sqrt();
        let mut y = vec![0.0; c.heads * seq * dv];
        let mut states = Vec::new();
        for h in 0..c.heads {
            let mut st = match &c.state {
                Some(st) => st[h * dk * dv..(h + 1) * dk * dv].to_vec(),
                None => vec![0.0; dk * dv],
            };
            for t in 0..seq {
                let at = h * seq + t;
                let q = &c.q[at * dk..(at + 1) * dk];
                let k = &c.k[at * dk..(at + 1) * dk];
                let v = &c.v[at * dv..(at + 1) * dv];
                let decay = c.g[at].exp();
                st.iter_mut().for_each(|x| *x *= decay);
                for j in 0..dv {
                    let recalled: f64 = (0..dk).map(|i| st[i * dv + j] * k[i]).sum();
                    let delta = c.beta[at] * (v[j] - recalled);
                    (0..dk).for_each(|i| st[i * dv + j] += k[i] * delta);
                }
                for j in 0..dv {
                    y[at * dv + j] = (0..dk).map(|i| st[i * dv + j] * q[i] * scale).sum();
                }
            }
            states.extend(st);
        }
        (y, states)
    }

    fn tensor<const D: usize>(x: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
        let x: Vec<f32> = x.iter().map(|&a| a as f32).collect();
        Tensor::from_data(TensorData::new(x, shape), device)
    }

    fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_into_vec::<f32>().unwrap()
    }

    fn run(c: &Case, chunk_size: usize, device: &Device) -> (Tensor<4>, Tensor<4>) {
        let (h, s, dk, dv) = (c.heads, c.seq, c.dk, c.dv);
        gated_delta_rule(
            tensor(&c.q, [1, h, s, dk], device),
            tensor(&c.k, [1, h, s, dk], device),
            tensor(&c.v, [1, h, s, dv], device),
            tensor(&c.g, [1, h, s], device),
            tensor(&c.beta, [1, h, s], device),
            c.state
                .as_ref()
                .map(|st| tensor(st, [1, h, dk, dv], device)),
            chunk_size,
        )
    }

    fn max_err(got: Tensor<4>, want: &[f64]) -> f64 {
        values(got)
            .iter()
            .zip(want)
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0, f64::max)
    }

    fn assert_matches_reference(c: &Case, chunk_size: usize, what: &str) {
        let (y_ref, st_ref) = reference(c);
        let (y, st) = run(c, chunk_size, &Default::default());
        let (ey, es) = (max_err(y, &y_ref), max_err(st, &st_ref));
        assert!(
            ey < 2e-4 && es < 2e-4,
            "{what}, chunk {chunk_size}: y err {ey:.2e}, state err {es:.2e}"
        );
    }

    #[test]
    fn chunked_rule_matches_the_per_token_rule() {
        // Shorter than a chunk, exactly one, one over, and several with a padded tail.
        for (i, seq) in [1, 5, 16, 17, 40].into_iter().enumerate() {
            for weak_decay in [false, true] {
                let c = case(100 + i as u64, 2, seq, 8, 6, weak_decay);
                assert_matches_reference(&c, 16, &format!("seq {seq}, weak decay {weak_decay}"));
            }
        }
    }

    #[test]
    fn chunk_size_does_not_change_the_result() {
        let c = case(7, 2, 37, 8, 8, true);
        for chunk_size in [1, 3, 8, 37, 64] {
            assert_matches_reference(&c, chunk_size, "seq 37");
        }
    }

    #[test]
    fn similar_keys_stay_accurate() {
        // Keys that point in nearly the same direction, little decay and strong writes make the
        // chunk matrix badly conditioned, as on real text. A power-series inverse fails here.
        let (heads, seq, dk, dv) = (1, 64, 32, 8);
        let mut r = Rng(77);
        let base = r.normals(dk);
        let k = (0..heads * seq)
            .flat_map(|_| {
                base.iter()
                    .map(|x| x + 0.15 * r.normal())
                    .collect::<Vec<_>>()
            })
            .collect();
        let c = Case {
            heads,
            seq,
            dk,
            dv,
            q: l2_normalized(r.normals(heads * seq * dk), dk),
            k: l2_normalized(k, dk),
            v: r.normals(heads * seq * dv),
            g: (0..heads * seq).map(|_| -0.01 * r.uniform()).collect(),
            beta: (0..heads * seq).map(|_| 0.9 + 0.1 * r.uniform()).collect(),
            state: None,
        };
        assert_matches_reference(&c, 64, "similar keys");
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        let device = Default::default();
        let mut r = Rng(3);
        // One block, a partial block, and several blocks with padding.
        for c in [1, 7, 16, 40] {
            let a: Vec<f64> = (0..2 * c * c)
                .map(|idx| match (idx / c) % c > idx % c {
                    true => 0.3 * r.normal(),
                    false => 0.0,
                })
                .collect();
            let a = tensor(&a, [2, c, c], &device);
            let eye = Tensor::<2>::eye(c, &device)
                .unsqueeze_dim::<3>(0)
                .expand([2, c, c]);
            let product = (eye.clone() + a.clone()).matmul(unit_lower_inverse(a));
            let err = values(product - eye)
                .iter()
                .fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(err < 1e-4, "c = {c}: |(I + A) X - I| = {err:.2e}");
        }
    }

    fn layer(device: &Device) -> GatedDeltaNet {
        GatedDeltaNetConfig::new(12, 2, 4, 4, 3)
            .with_chunk_size(8)
            .init(device)
    }

    #[test]
    fn splitting_the_input_carries_the_state() {
        let device = Default::default();
        let layer = layer(&device);
        let mut r = Rng(9);
        let input = tensor(&r.normals(2 * 21 * 12), [2, 21, 12], &device);

        let (whole, whole_state) = layer.forward(input.clone(), None);
        let (first, state) = layer.forward(input.clone().narrow(1, 0, 13), None);
        let (second, split_state) = layer.forward(input.narrow(1, 13, 8), Some(state));

        assert_eq!(whole.dims(), [2, 21, 12]);
        assert_eq!(whole_state.conv.dims(), [2, 2 * 8 + 12, 3]);
        assert_eq!(whole_state.recurrent.dims(), [2, 4, 4, 3]);
        Tensor::cat(vec![first, second], 1)
            .into_data()
            .assert_approx_eq::<f32>(&whole.into_data(), Default::default());
        split_state
            .recurrent
            .into_data()
            .assert_approx_eq::<f32>(&whole_state.recurrent.into_data(), Default::default());
    }

    #[test]
    fn gradients_reach_every_parameter() {
        let device = Device::default().autodiff();
        let layer = layer(&device);
        let mut r = Rng(11);
        let input = tensor(&r.normals(20 * 12), [1, 20, 12], &device);

        let grads = layer.forward(input, None).0.square().sum().backward();

        let norms = [
            ("in_proj_qkv", layer.in_proj_qkv.weight.grad(&grads)),
            ("in_proj_a", layer.in_proj_a.weight.grad(&grads)),
            ("conv1d", layer.conv1d.grad(&grads)),
            ("a_log", layer.a_log.grad(&grads).map(|g| g.reshape([1, 4]))),
        ];
        for (name, grad) in norms {
            let grad = grad.unwrap_or_else(|| panic!("no gradient for {name}"));
            let norm = values(grad.square().sum().reshape([1, 1, 1, 1]))[0];
            assert!(
                norm.is_finite() && norm > 0.0,
                "{name}: gradient norm {norm}"
            );
        }
    }

    #[test]
    fn display() {
        let layer = layer(&Default::default());
        assert_eq!(
            format!("{layer}"),
            "GatedDeltaNet {d_model: 12, n_key_heads: 2, n_value_heads: 4, d_key_head: 4, d_value_head: 3, chunk_size: 8, params: 843}"
        );
    }
}
