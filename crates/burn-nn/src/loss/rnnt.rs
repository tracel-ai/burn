use super::Reduction;
use alloc::vec;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Bool, Device, Int, Tensor, s};
use burn_core as burn;
use core::f32;

/// Configuration for [RNNTLoss](RNNTLoss).
#[derive(Config, Debug)]
pub struct RNNTLossConfig {
    /// Index of the blank label in the vocabulary. Default: `0`.
    #[config(default = 0)]
    pub blank: usize,
    /// Treat the inputs as logits, applying a log-softmax on the last dimension internally.
    /// If `false`, the input must already be log-probabilities. Default: `true`.
    #[config(default = true)]
    pub logits: bool,
}

impl RNNTLossConfig {
    /// Initializes a [RNNTLoss](RNNTLoss) module.
    pub fn init(&self) -> RNNTLoss {
        RNNTLoss {
            blank: self.blank,
            logits: self.logits,
        }
    }
}

/// RNN Transducer (RNNT) loss, as described in
/// [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711).
///
/// Computes the negative log-likelihood over a 2D lattice of encoder time steps (T)
/// and output labels (U), marginalizing over all valid alignments.
///
/// # Example
///
/// ```rust,ignore
/// let rnnt = RNNTLossConfig::new().init();
///
/// // logits: [B, T, U+1, V] from the joiner network
/// let loss = rnnt.forward(logits, targets, logit_lengths, target_lengths);
/// ```
#[derive(Module, Debug)]
pub struct RNNTLoss {
    blank: usize,
    logits: bool,
}

impl RNNTLoss {
    /// Computes per-sample RNNT loss (no reduction). Returns shape `[B]`.
    ///
    /// - `logits`: `[B, T, U+1, V]` — joiner output (raw logits or log-probs)
    /// - `targets`: `[B, U]` — target label indices (must not contain blank)
    /// - `logit_lengths`: `[B]` — actual encoder lengths per sample
    /// - `target_lengths`: `[B]` — actual target lengths per sample
    pub fn forward(
        &self,
        logits: Tensor<4>,
        targets: Tensor<2, Int>,
        logit_lengths: Tensor<1, Int>,
        target_lengths: Tensor<1, Int>,
    ) -> Tensor<1> {
        let device = logits.device();
        let [b, max_t, max_up1, v] = logits.dims();
        let max_u = max_up1 - 1;

        self.check_inputs(b, v, &targets, &logit_lengths, &target_lengths, max_u);

        let log_probs = if self.logits {
            let vocab_dim = 3; // last dim of [B, T, U+1, V]
            burn::tensor::activation::log_softmax(logits, vocab_dim)
        } else {
            logits
        };

        let (lpb, lpl) = self.extract_log_probs(log_probs, targets);
        let u_mask = self.create_u_mask(&target_lengths, b, max_up1, &device);
        let neg_inf = Tensor::<2>::full([b, max_up1], f32::NEG_INFINITY, &device);

        // Forward pass: compute log_alpha across the (T, U) lattice
        let mut alpha = self.init_alpha(&lpl, b, max_up1, &device);
        alpha = neg_inf.clone().mask_where(u_mask.clone(), alpha);

        let logit_lengths_exp = logit_lengths.clone().reshape([b, 1]).expand([b, max_up1]);

        for t in 1..max_t {
            let new = self.step_alpha(&alpha, &lpb, &lpl, t);
            let new = neg_inf.clone().mask_where(u_mask.clone(), new);

            // Only update alpha for samples where t < logit_lengths[b]
            let valid = logit_lengths_exp.clone().greater_elem(t as i64);
            alpha = alpha.mask_where(valid, new);
        }

        self.gather_loss(alpha, &lpb, logit_lengths, target_lengths, b)
    }

    /// Computes RNNT loss with the given reduction. Returns shape `[1]`.
    pub fn forward_with_reduction(
        &self,
        logits: Tensor<4>,
        targets: Tensor<2, Int>,
        logit_lengths: Tensor<1, Int>,
        target_lengths: Tensor<1, Int>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward(logits, targets, logit_lengths, target_lengths);
        match reduction {
            Reduction::Auto | Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Gathers `log_prob_blank[B, T, U+1]` and `log_prob_label[B, T, U]` from the full
    /// log-probability tensor by indexing into the vocab dimension.
    fn extract_log_probs(
        &self,
        log_probs: Tensor<4>,
        targets: Tensor<2, Int>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [b, max_t, max_up1, v] = log_probs.dims();
        let max_u = max_up1 - 1;
        let vocab_dim = 3;

        // Blank probabilities: slice log_probs in vocab dim using the blank index
        let lpb = log_probs
            .clone()
            .slice_dim(vocab_dim, self.blank)
            .squeeze_dim::<3>(vocab_dim);

        // Label probabilities: gather target labels across vocab dim (only first U positions)
        let tgt = targets
            .reshape([b, 1, max_u, 1])
            .expand([b, max_t, max_u, 1]);
        let lpl = log_probs
            .slice(s![.., .., 0..max_u, 0..v])
            .gather(vocab_dim, tgt)
            .squeeze_dim::<3>(vocab_dim);

        (lpb, lpl)
    }

    /// Sets up log_alpha at t=0: `alpha(0,0) = 0`, then cumsum of label probs along u.
    fn init_alpha(&self, lpl: &Tensor<3>, b: usize, max_up1: usize, device: &Device) -> Tensor<2> {
        // Label probs at t=0
        let lpl_0 = lpl.clone().slice(s![.., 0..1, ..]).squeeze_dim::<2>(1);
        let zero_col = Tensor::<2>::zeros([b, 1], device);
        let prefix = Tensor::cat(vec![zero_col, lpl_0.slice(s![.., 0..(max_up1 - 1)])], 1);

        prefix.cumsum(1)
    }

    /// Boolean mask `[B, U+1]` that is true where `u <= target_lengths[b]`.
    fn create_u_mask(
        &self,
        target_lengths: &Tensor<1, Int>,
        b: usize,
        max_up1: usize,
        device: &Device,
    ) -> Tensor<2, Bool> {
        let indices = Tensor::<1, Int>::arange(0..max_up1 as i64, device)
            .reshape([1, max_up1])
            .expand([b, max_up1]);
        let lengths = target_lengths.clone().reshape([b, 1]).expand([b, max_up1]);
        indices.lower_equal(lengths)
    }

    /// One time step of the forward recurrence:
    ///
    ///   alpha(t, u) = logaddexp(
    ///       alpha(t-1, u) + blank(t-1, u),
    ///       alpha(t, u-1) + label(t, u-1),
    ///   )
    fn step_alpha(
        &self,
        alpha: &Tensor<2>,
        lpb: &Tensor<3>,
        lpl: &Tensor<3>,
        t: usize,
    ) -> Tensor<2> {
        let [b, max_up1] = alpha.dims();
        let device = alpha.device();

        // Blank transition: alpha(t-1, :) + blank_prob(t-1, :)
        let blank_prob = lpb
            .clone()
            .slice(s![.., (t - 1)..t, ..])
            .squeeze_dim::<2>(1);
        let from_blank = alpha.clone().add(blank_prob);

        let mut new = Tensor::<2>::full([b, max_up1], f32::NEG_INFINITY, &device);
        new = new.slice_assign(s![.., 0..1], from_blank.clone().slice(s![.., 0..1]));

        // Label probs at time t
        let label_prob = lpl
            .clone()
            .slice(s![.., t..(t + 1), ..])
            .squeeze_dim::<2>(1);

        for u in 1..max_up1 {
            let via_blank = from_blank.clone().slice(s![.., u..(u + 1)]);
            let via_label = new
                .clone()
                .slice(s![.., (u - 1)..u])
                .add(label_prob.clone().slice(s![.., (u - 1)..u]));
            new = new.slice_assign(s![.., u..(u + 1)], self.log_sum_exp(via_blank, via_label));
        }
        new
    }

    /// Extracts `-(alpha(T_b, U_b) + blank(T_b, U_b))` for each sample in the batch.
    fn gather_loss(
        &self,
        alpha: Tensor<2>,
        lpb: &Tensor<3>,
        logit_lengths: Tensor<1, Int>,
        target_lengths: Tensor<1, Int>,
        b: usize,
    ) -> Tensor<1> {
        let device = alpha.device();
        // Anchor the index dtype on `u_idx` so all three coordinate tensors share a
        // common int dtype before stacking - `cat` panics on dtype mismatch and the
        // caller's lengths may not use the device's default IntElem.
        let u_idx = target_lengths;
        let int_dtype = u_idx.dtype();
        let t_idx = logit_lengths.sub_scalar(1).cast(int_dtype);
        let b_idx = Tensor::<1, Int>::arange(0..b as i64, (&device, int_dtype));

        let alpha_tu: Tensor<1> =
            alpha.gather_nd(Tensor::stack::<2>(vec![b_idx.clone(), u_idx.clone()], 1));
        let lpb_tu: Tensor<1> = lpb
            .clone()
            .gather_nd(Tensor::stack::<2>(vec![b_idx, t_idx, u_idx], 1));

        alpha_tu.add(lpb_tu).neg()
    }

    fn check_inputs(
        &self,
        b: usize,
        v: usize,
        targets: &Tensor<2, Int>,
        logit_lengths: &Tensor<1, Int>,
        target_lengths: &Tensor<1, Int>,
        max_u: usize,
    ) {
        assert!(
            self.blank < v,
            "blank index {} must be less than vocab_size {}",
            self.blank,
            v
        );
        assert_eq!(
            targets.dims()[0],
            b,
            "targets batch dimension {} must equal batch_size {}",
            targets.dims()[0],
            b
        );
        assert_eq!(
            targets.dims()[1],
            max_u,
            "targets length dimension {} must equal max_target_len (max_u) {}",
            targets.dims()[1],
            max_u
        );
        assert_eq!(
            logit_lengths.dims()[0],
            b,
            "logit_lengths length {} must equal batch_size {}",
            logit_lengths.dims()[0],
            b
        );
        assert_eq!(
            target_lengths.dims()[0],
            b,
            "target_lengths length {} must equal batch_size {}",
            target_lengths.dims()[0],
            b
        );
    }

    /// Numerically stable `log(exp(a) + exp(b))`, handling `-inf` inputs.
    fn log_sum_exp<const D: usize>(&self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
        let a_inf = a.clone().equal_elem(f32::NEG_INFINITY);
        let b_inf = b.clone().equal_elem(f32::NEG_INFINITY);

        // Replace -inf with 0 to prevent NaN in the subtraction (masked out below)
        let a_safe = a.clone().mask_fill(a_inf.clone(), 0.0);
        let b_safe = b.clone().mask_fill(b_inf.clone(), 0.0);

        // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
        let max = a_safe.clone().max_pair(b_safe.clone());
        let result = max.add(a_safe.sub(b_safe).abs().neg().exp().add_scalar(1.0).log());

        // If a=-inf, result is b; if b=-inf, result is a; if both -inf, stays -inf
        let result = result.mask_where(a_inf, b);
        result.mask_where(b_inf, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, Tolerance};
    const NUM_LABELS: usize = 2; // vocab size for simple unit tests

    #[test]
    fn config_defaults() {
        let cfg = RNNTLossConfig::new();
        assert_eq!(cfg.blank, 0);
        assert!(cfg.logits);
    }

    #[test]
    #[should_panic(expected = "blank index")]
    fn panics_on_invalid_blank() {
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().with_blank(5).init();
        rnnt.forward(
            Tensor::<4>::zeros([1, 2, 2, 3], &dev),
            Tensor::<2, Int>::from_data([[1_i32]], &dev),
            Tensor::<1, Int>::from_data([2], &dev),
            Tensor::<1, Int>::from_data([1], &dev),
        );
    }

    #[test]
    #[should_panic(expected = "must equal batch_size")]
    fn panics_on_batch_mismatch() {
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().init();
        rnnt.forward(
            Tensor::<4>::zeros([2, 3, 2, 3], &dev),
            Tensor::<2, Int>::from_data([[1_i32]], &dev),
            Tensor::<1, Int>::from_data([3, 3], &dev),
            Tensor::<1, Int>::from_data([1, 1], &dev),
        );
    }

    #[test]
    #[should_panic(expected = "logit_lengths length")]
    fn panics_on_logit_lengths_mismatch() {
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().init();
        rnnt.forward(
            Tensor::<4>::zeros([2, 3, 2, 3], &dev),
            Tensor::<2, Int>::from_data([[1_i32], [2]], &dev),
            Tensor::<1, Int>::from_data([3], &dev),
            Tensor::<1, Int>::from_data([1, 1], &dev),
        );
    }

    #[test]
    #[should_panic(expected = "target_lengths length")]
    fn panics_on_target_lengths_mismatch() {
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().init();
        rnnt.forward(
            Tensor::<4>::zeros([2, 3, 2, 3], &dev),
            Tensor::<2, Int>::from_data([[1_i32], [2]], &dev),
            Tensor::<1, Int>::from_data([3, 3], &dev),
            Tensor::<1, Int>::from_data([1], &dev),
        );
    }

    #[test]
    fn single_token_uniform_probs() {
        // B=1, T=2, U=1, V=2, uniform probs: P(blank) = P(label) = 1/V
        //
        // Two alignment paths (label emitted at t=0 or t=1), each with T+U emissions:
        //   total_prob = T * (1/V)^(T+1) = 2 * (1/2)^3 = 1/4
        //   loss = -ln(1/4) = 2*ln(2)
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().with_logits(false).init();
        let time_steps = 2;
        let target_len = 1;
        let v = NUM_LABELS as f32;
        let log_uniform = (1.0 / v).ln();

        let loss = rnnt.forward(
            Tensor::<4>::full(
                [1, time_steps, target_len + 1, NUM_LABELS],
                log_uniform,
                &dev,
            ),
            Tensor::<2, Int>::from_data([[1_i32]], &dev),
            Tensor::<1, Int>::from_data([time_steps as i64], &dev),
            Tensor::<1, Int>::from_data([target_len as i64], &dev),
        );
        // Each path: T-1 blanks + U labels + 1 final blank = T + U emissions
        let num_paths = time_steps as f32;
        let emissions_per_path = (time_steps + target_len) as f32;
        let total_prob = num_paths * v.powf(-emissions_per_path);
        let expected_loss = -total_prob.ln();
        loss.into_data().assert_approx_eq::<f32>(
            &TensorData::from([expected_loss]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn empty_target() {
        // B=1, T=3, U=0, V=2, uniform probs: only the all-blanks path exists.
        //
        // Single path with T emissions (T-1 blanks + 1 final blank, all at u=0):
        //   total_prob = (1/V)^T = (1/2)^3 = 1/8
        //   loss = T*ln(V) = 3*ln(2)
        let dev = Default::default();
        let rnnt = RNNTLossConfig::new().with_logits(false).init();
        let time_steps = 3;
        let target_len = 0;
        let v = NUM_LABELS as f32;
        let log_uniform = (1.0 / v).ln();

        let loss = rnnt.forward(
            Tensor::<4>::full([1, time_steps, 2, NUM_LABELS], log_uniform, &dev),
            Tensor::<2, Int>::from_data([[1_i32]], &dev),
            Tensor::<1, Int>::from_data([time_steps as i64], &dev),
            Tensor::<1, Int>::from_data([target_len as i64], &dev),
        );
        // T + U = T emissions total for U=0
        let expected_loss = -v.powf(-((time_steps + target_len) as f32)).ln();
        loss.into_data().assert_approx_eq::<f32>(
            &TensorData::from([expected_loss]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn logits_equivalence() {
        // Verify that logits=true (internal log_softmax on raw logits)
        // gives the same loss as logits=false with external log_softmax.
        let dev = Default::default();
        let [bs, time_steps, up1, vocab] = [1, 2, 3, 4];
        let num_elements = bs * time_steps * up1 * vocab;
        let target_len = up1 - 1;

        let data: Vec<f32> = (0..num_elements).map(|i| (i as f32 * 0.3).sin()).collect();
        let logits = Tensor::<4>::from_data(
            burn_core::tensor::TensorData::new(data, [bs, time_steps, up1, vocab]),
            &dev,
        );
        let targets = Tensor::<2, Int>::from_data([[1_i32, 2]], &dev);
        let logit_lengths = Tensor::<1, Int>::from_data([time_steps as i64], &dev);
        let target_lengths = Tensor::<1, Int>::from_data([target_len as i64], &dev);

        let vocab_dim = 3;
        let fused = RNNTLossConfig::new().with_logits(true).init().forward(
            logits.clone(),
            targets.clone(),
            logit_lengths.clone(),
            target_lengths.clone(),
        );

        let log_probs = burn::tensor::activation::log_softmax(logits, vocab_dim);
        let manual = RNNTLossConfig::new().with_logits(false).init().forward(
            log_probs,
            targets,
            logit_lengths,
            target_lengths,
        );

        fused
            .into_data()
            .assert_approx_eq::<f32>(&manual.into_data(), Tolerance::absolute(1e-4));
    }
}

/// Tests comparing forward loss and backward gradients against torchaudio.functional.rnnt_loss.
///
/// Logits are generated deterministically via sin((b*11+t*7+u*13+v*3)*0.1) so the same
/// values can be reproduced in a Python script for cross-checking.
#[cfg(test)]
#[allow(clippy::identity_op, clippy::too_many_arguments)]
mod pytorch_comparison_tests {
    use super::*;
    use burn::tensor::{TensorData, Tolerance};
    fn tol() -> Tolerance<f32> {
        Tolerance::absolute(1e-3)
    }

    /// Deterministic logits matching the Python reference generator.
    /// Uses coprime coefficients to avoid repeating patterns across dimensions.
    fn make_logits(bs: usize, t: usize, u: usize, v: usize, dev: &Device) -> Tensor<4> {
        let mut data = Vec::with_capacity(bs * t * u * v);
        for bi in 0..bs {
            for ti in 0..t {
                for ui in 0..u {
                    for vi in 0..v {
                        let idx = bi * 11 + ti * 7 + ui * 13 + vi * 3;
                        data.push((idx as f32 * 0.1).sin());
                    }
                }
            }
        }
        Tensor::from_data(TensorData::new(data, [bs, t, u, v]), dev)
    }

    /// Checks that gradients along the vocab dim sum to ~0 at every (b, t, u) position.
    /// This must hold because log_softmax is applied on the last dim,
    /// and the Jacobian of log_softmax has the property that each row sums to zero.
    fn check_vocab_grad_sums(grad: &[f32], bs: usize, t: usize, up1: usize, v: usize) {
        for bi in 0..bs {
            for ti in 0..t {
                for ui in 0..up1 {
                    let base = ((bi * t + ti) * up1 + ui) * v;
                    let sum: f32 = (0..v).map(|vi| grad[base + vi]).sum();
                    TensorData::from([sum])
                        .assert_approx_eq::<f32>(&TensorData::from([0.0f32]), tol());
                }
            }
        }
    }

    /// Returns the V-sized gradient slice at position (b, t, u) in a flattened [B, T, U+1, V] grad.
    fn grad_at(
        grad: &[f32],
        b: usize,
        t: usize,
        u: usize,
        max_t: usize,
        up1: usize,
        v: usize,
    ) -> &[f32] {
        let base = ((b * max_t + t) * up1 + u) * v;
        &grad[base..base + v]
    }

    /// Asserts that a gradient slice at position (b, t, u) matches expected values.
    fn assert_grad(
        grad: &[f32],
        b: usize,
        t: usize,
        u: usize,
        max_t: usize,
        up1: usize,
        v: usize,
        expected: &[f32],
    ) {
        TensorData::from(grad_at(grad, b, t, u, max_t, up1, v))
            .assert_approx_eq::<f32>(&TensorData::from(expected), tol());
    }

    #[test]
    fn basic_b1() {
        // B=1, T=4, U+1=3, V=3, targets=[1,2]
        let dev = Device::default().autodiff();
        let rnnt = RNNTLossConfig::new().init();
        let logits = make_logits(1, 4, 3, 3, &dev).require_grad();

        let loss = rnnt.forward(
            logits.clone(),
            Tensor::<2, Int>::from_data([[1_i32, 2]], &dev),
            Tensor::<1, Int>::from_data([4_i32], &dev),
            Tensor::<1, Int>::from_data([2_i32], &dev),
        );
        loss.clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([4.4491f32]), tol());

        let grads = loss.sum().backward();
        let grad = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Spot-check first, middle, and last (t, u) positions against torchaudio
        assert_grad(&grad, 0, 0, 0, 4, 3, 3, &[-0.2041, -0.2246, 0.4287]);
        assert_grad(&grad, 0, 2, 0, 4, 3, 3, &[0.0079, -0.0640, 0.0561]);
        assert_grad(&grad, 0, 3, 2, 4, 3, 3, &[-0.6899, 0.3231, 0.3667]);
        check_vocab_grad_sums(&grad, 1, 4, 3, 3);
    }

    #[test]
    fn batched_b2() {
        // B=2, T=5, U+1=4, V=4, targets=[[1,2,3],[2,1,3]]
        let dev = Device::default().autodiff();
        let rnnt = RNNTLossConfig::new().init();
        let logits = make_logits(2, 5, 4, 4, &dev).require_grad();

        let loss = rnnt.forward(
            logits.clone(),
            Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 2, 1, 3], [2, 3]), &dev),
            Tensor::<1, Int>::from_data([5_i32, 5], &dev),
            Tensor::<1, Int>::from_data([3_i32, 3], &dev),
        );
        loss.clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([7.9356f32, 7.2033]), tol());

        let grads = loss.sum().backward();
        let grad = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Spot-check: first position of each sample, and last position
        assert_grad(&grad, 0, 0, 0, 5, 4, 4, &[-0.3161, -0.3113, 0.2796, 0.3479]);
        assert_grad(&grad, 1, 0, 0, 5, 4, 4, &[-0.2766, 0.2602, -0.2248, 0.2411]);
        assert_grad(&grad, 0, 4, 3, 5, 4, 4, &[-0.8216, 0.2296, 0.2786, 0.3133]);
        assert_grad(&grad, 1, 4, 3, 5, 4, 4, &[-0.7185, 0.2735, 0.2437, 0.2012]);
        check_vocab_grad_sums(&grad, 2, 5, 4, 4);
    }

    #[test]
    fn variable_lengths_b3() {
        // B=3, T=6, U+1=4, V=5
        // logit_lengths=[6,4,5], target_lengths=[3,2,1]
        // Tests that masking works correctly for variable-length sequences.
        let dev = Device::default().autodiff();
        let rnnt = RNNTLossConfig::new().init();
        let logits = make_logits(3, 6, 4, 5, &dev).require_grad();

        let loss = rnnt.forward(
            logits.clone(),
            Tensor::<2, Int>::from_data(
                TensorData::new(vec![1_i32, 2, 3, 4, 1, 0, 2, 0, 0], [3, 3]),
                &dev,
            ),
            Tensor::<1, Int>::from_data([6_i32, 4, 5], &dev),
            Tensor::<1, Int>::from_data([3_i32, 2, 1], &dev),
        );
        loss.clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([10.7458f32, 8.0196, 8.3316]), tol());

        let grads = loss.sum().backward();
        let grad = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let stride = 4 * 5; // U+1 * V per time step
        let zeros = vec![0.0f32; 5];

        // Sample 0 (full length=6): spot-check first and last active positions
        assert_grad(
            &grad,
            0,
            0,
            0,
            6,
            4,
            5,
            &[-0.4232, -0.3114, 0.1992, 0.2478, 0.2876],
        );
        assert_grad(
            &grad,
            0,
            5,
            3,
            6,
            4,
            5,
            &[-0.8016, 0.2170, 0.2172, 0.1991, 0.1683],
        );

        // Sample 1 (logit_length=4): gradients beyond t=3 should be zero
        assert_grad(
            &grad,
            1,
            0,
            0,
            6,
            4,
            5,
            &[-0.2502, 0.2160, 0.2173, 0.2002, -0.3833],
        );
        let sample1_t4_start = 1 * 6 * stride + 4 * stride;
        for i in 0..(2 * stride) {
            // t=4 and t=5 should all be zero
            assert!(
                grad[sample1_t4_start + i].abs() < 1e-3,
                "sample 1, t>=4: grad[{}] = {} (expected 0)",
                i,
                grad[sample1_t4_start + i]
            );
        }

        // Sample 1 (target_length=2): u=3 positions should be zero within active time steps
        for ti in 0..4 {
            assert_grad(&grad, 1, ti, 3, 6, 4, 5, &zeros);
        }

        // Sample 2 (logit_length=5): t=5 should be zero
        let sample2_t5_start = 2 * 6 * stride + 5 * stride;
        for i in 0..stride {
            assert!(
                grad[sample2_t5_start + i].abs() < 1e-3,
                "sample 2, t=5: grad[{}] = {} (expected 0)",
                i,
                grad[sample2_t5_start + i]
            );
        }

        check_vocab_grad_sums(&grad, 3, 6, 4, 5);
    }

    #[test]
    fn sum_reduction() {
        let dev = Device::default().autodiff();
        let rnnt = RNNTLossConfig::new().init();
        let logits = make_logits(2, 5, 4, 4, &dev).require_grad();
        let tgt =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 2, 1, 3], [2, 3]), &dev);
        let il = Tensor::<1, Int>::from_data([5_i32, 5], &dev);
        let tl = Tensor::<1, Int>::from_data([3_i32, 3], &dev);

        let loss = rnnt.forward_with_reduction(logits.clone(), tgt, il, tl, Reduction::Sum);
        // 7.9356 + 7.2033 = 15.1389
        loss.clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([15.1389f32]), tol());

        let grads = loss.backward();
        let g = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        TensorData::from(&g[..4]).assert_approx_eq::<f32>(
            &TensorData::from([-0.3161f32, -0.3113, 0.2796, 0.3479]),
            tol(),
        );
    }

    #[test]
    fn mean_reduction() {
        let dev = Device::default().autodiff();
        let rnnt = RNNTLossConfig::new().init();
        let logits = make_logits(2, 5, 4, 4, &dev).require_grad();
        let tgt =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 2, 1, 3], [2, 3]), &dev);
        let il = Tensor::<1, Int>::from_data([5_i32, 5], &dev);
        let tl = Tensor::<1, Int>::from_data([3_i32, 3], &dev);

        let loss = rnnt.forward_with_reduction(logits.clone(), tgt, il, tl, Reduction::Mean);
        // 15.1389 / 2 = 7.5694
        loss.clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([7.5694f32]), tol());

        // Gradients should be half the sum-reduction gradients (mean over batch of 2)
        let grads = loss.backward();
        let g = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        TensorData::from(&g[..4]).assert_approx_eq::<f32>(
            &TensorData::from([-0.1581f32, -0.1557, 0.1398, 0.1739]),
            tol(),
        );
    }
}
