use super::*;
use burn_tensor::Tolerance;
use burn_tensor::activation::log_softmax;
use burn_tensor::module::ctc_loss;

#[test]
fn test_ctc_loss_unreachable_target_is_inf() {
    // T=2, N=1, C=3, target=[1, 1]. CTC requires at least `target_length
    // + repeats = 3` steps to produce this target, but input_length is 2,
    // so no valid alignment exists and the loss must be +inf. burn-nn's
    // `zero_infinity` relies on `is_inf` to detect this case, so fused
    // backend kernels must preserve the +inf (not collapse to a large
    // finite number).
    let device = Default::default();
    let log_probs = TestTensor::<3>::full([2, 1, 3], (1.0f32 / 3.0).ln(), &device);
    let targets = TestTensorInt::<2>::from([[1_i64, 1]]);
    let input_lengths = TestTensorInt::<1>::from([2_i64]);
    let target_lengths = TestTensorInt::<1>::from([2_i64]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);
    let loss_data = loss.into_data();
    let value = loss_data.iter::<f32>().next().unwrap();
    assert!(
        value.is_infinite() && value > 0.0,
        "Expected +inf for an unreachable target, got {value}"
    );
}

/// Regression for NaN gradient contamination on unreachable targets via the
/// default `ctc_loss_backward` path (the one used by burn-cubecl's fused
/// kernel via `ctc_grad_from_alpha_beta_default`). `log_alpha + log_beta -
/// log_probs + nll` evaluates to `(-inf) + (+inf) = NaN` for nll = +inf
/// samples, and `NaN * 0 = NaN` under IEEE 754 would defeat any outer
/// zero_infinity masking. The fix masks gradient entries for is_inf(nll)
/// samples directly inside `ctc_grad_from_alpha_beta_default`.
///
/// Skipped on backends without a native `ctc_loss_backward`; for those
/// backends the gradient is built by autodiff differentiating through the
/// decomposed forward, which never enters `ctc_grad_from_alpha_beta_default`.
#[test]
fn test_ctc_loss_backward_unreachable_is_finite() {
    use burn_tensor::ops::ModuleOps;

    if !<TestBackend as ModuleOps<TestBackend>>::has_ctc_loss_backward() {
        return;
    }

    let device = Default::default();

    // T=2, target=[1, 1] requires three steps, so nll = +inf for this sample.
    let log_probs = TestTensor::<3>::full([2, 1, 3], (1.0f32 / 3.0).ln(), &device);
    let targets =
        TestTensorInt::<2>::from_data(burn_tensor::TensorData::from([[1_i64, 1]]), &device);
    let input_lengths =
        TestTensorInt::<1>::from_data(burn_tensor::TensorData::from([2_i64]), &device);
    let target_lengths =
        TestTensorInt::<1>::from_data(burn_tensor::TensorData::from([2_i64]), &device);
    let grad_loss = TestTensor::<1>::ones([1], &device);

    let grad = <TestBackend as ModuleOps<TestBackend>>::ctc_loss_backward(
        log_probs.into_primitive().tensor(),
        targets.into_primitive(),
        input_lengths.into_primitive(),
        target_lengths.into_primitive(),
        grad_loss.into_primitive().tensor(),
        0,
    );
    let grad_tensor = TestTensor::<3>::from_primitive(burn_tensor::TensorPrimitive::Float(grad));
    let grad_data = grad_tensor.into_data();
    for v in grad_data.iter::<f32>() {
        assert!(
            v.is_finite(),
            "gradient for unreachable target must be finite, got {v}"
        );
        assert_eq!(
            v, 0.0f32,
            "gradient for unreachable target must be zero, got {v}"
        );
    }
}

#[test]
fn test_ctc_loss_empty_target() {
    // T=3, N=1, C=2, blank=0, target_length=0, uniform P = 1/2.
    // With an empty target, the only valid alignment is all blanks.
    // Loss = -ln((1/2)^3) = 3 * ln(2) ~ 2.0794.
    //
    // Uses a [1, 1] targets shape rather than [1, 0] since not every backend
    // supports zero-sized dimensions. The target slot at [0, 0] is never read
    // because target_lengths[0] == 0.
    let log_probs = TestTensor::<3>::full([3, 1, 2], (0.5f32).ln(), &Default::default());
    let targets = TestTensorInt::<2>::from([[0]]);
    let input_lengths = TestTensorInt::<1>::from([3]);
    let target_lengths = TestTensorInt::<1>::from([0]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([3.0f32 * 2.0f32.ln()]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}

#[test]
fn test_ctc_loss_uniform() {
    // T=3, N=1, C=2, blank=0, target=[1, 1], uniform P = 1/2.
    // Only valid path is (1, 0, 1) with prob (1/2)^3.
    // Loss = -ln(1/8) = 3 * ln(2) ~ 2.0794
    let log_probs = TestTensor::<3>::full([3, 1, 2], (0.5f32).ln(), &Default::default());
    let targets = TestTensorInt::<2>::from([[1, 1]]);
    let input_lengths = TestTensorInt::<1>::from([3]);
    let target_lengths = TestTensorInt::<1>::from([2]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([3.0f32 * 2.0f32.ln()]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}

#[test]
fn test_ctc_loss_long_sequence() {
    // T=400, N=2, C=36, target_lengths=[60, 40]. Exercises the alpha
    // recursion at training-realistic sequence lengths where numerical
    // issues (shared-memory overflow, log-sum-exp accumulation drift)
    // would surface but don't show in the short T=5..12 tests.
    let t: usize = 400;
    let n: usize = 2;
    let c: usize = 36;
    let mut data = Vec::with_capacity(t * n * c);
    for ti in 0..t {
        for ni in 0..n {
            for ci in 0..c {
                data.push(((ti * 7 + ni * 13 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits = TestTensor::<3>::from(burn_tensor::TensorData::new(data, [t, n, c]));
    let log_probs = log_softmax(logits, 2);

    // Targets: distinct labels for each batch element, no consecutive repeats.
    let tgt_a: Vec<i64> = (0..60).map(|i| (i % 35 + 1) as i64).collect();
    let tgt_b: Vec<i64> = (0..60)
        .map(|i| if i < 40 { (i % 35 + 1) as i64 } else { 0 })
        .collect();
    let mut tgt_flat = tgt_a;
    tgt_flat.extend(tgt_b);
    let targets = TestTensorInt::<2>::from(burn_tensor::TensorData::new(tgt_flat, [n, 60]));
    let input_lengths = TestTensorInt::<1>::from([400, 400]);
    let target_lengths = TestTensorInt::<1>::from([60, 40]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);
    // `iter::<f32>` converts from the backend's native dtype (f16, f32, ...)
    // to f32 for comparison; `to_vec::<f32>` would require exact dtype match.
    let loss_data: Vec<f32> = loss.into_data().iter::<f32>().collect();

    // We don't have a PyTorch reference for this exact input, but the loss
    // must be positive (it's -log P where 0 < P <= 1) and finite.
    for (i, &v) in loss_data.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "sample {i}: expected positive finite loss, got {v}"
        );
    }
    // With 36 classes and random-ish logits, loss should be in the hundreds.
    // An off-by-orders-of-magnitude result (< 1 or > 10000) indicates a bug.
    for (i, &v) in loss_data.iter().enumerate() {
        assert!(
            v > 10.0 && v < 10000.0,
            "sample {i}: loss {v} outside plausible range [10, 10000]"
        );
    }
}

#[test]
fn test_ctc_loss_long_mixed_input_lengths() {
    // T=494, N=2, C=36, input_lengths=[494, 390], target_lengths=[64, 43].
    // The second sample has 104 frames of padding (input_length < T).
    // Reproduces the exact dimensions of a real training batch that gave
    // incorrect results on the cubecl kernel.
    let t: usize = 494;
    let n: usize = 2;
    let c: usize = 36;
    let mut data = Vec::with_capacity(t * n * c);
    for ti in 0..t {
        for ni in 0..n {
            for ci in 0..c {
                data.push(((ti * 7 + ni * 13 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits = TestTensor::<3>::from(burn_tensor::TensorData::new(data, [t, n, c]));
    let log_probs = log_softmax(logits, 2);

    let max_tgt = 64;
    let tgt_a: Vec<i64> = (0..max_tgt).map(|i| (i % 35 + 1) as i64).collect();
    let tgt_b: Vec<i64> = (0..max_tgt)
        .map(|i| if i < 43 { (i % 35 + 1) as i64 } else { 0 })
        .collect();
    let mut tgt_flat = tgt_a;
    tgt_flat.extend(tgt_b);
    let targets = TestTensorInt::<2>::from(burn_tensor::TensorData::new(tgt_flat, [n, max_tgt]));
    let input_lengths = TestTensorInt::<1>::from([494, 390]);
    let target_lengths = TestTensorInt::<1>::from([64, 43]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);
    let loss_data: Vec<f32> = loss.into_data().iter::<f32>().collect();

    for (i, &v) in loss_data.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "sample {i}: expected positive finite loss, got {v}"
        );
        assert!(
            v > 10.0 && v < 10000.0,
            "sample {i}: loss {v} outside plausible range [10, 10000]"
        );
    }
}

#[test]
fn test_ctc_loss_narrowed_input() {
    // CTC on a narrowed (non-contiguous, offset != 0) log_probs tensor.
    // Reproduces the pattern: model outputs [T_total, B, C], training
    // code narrows to [T_total - warmup, B, C] before passing to CTC.
    // If the kernel ignores the tensor's base offset, it reads the wrong
    // data and produces garbage loss.
    let t_full: usize = 10;
    let warmup: usize = 3;
    let t_eff: usize = t_full - warmup;
    let n: usize = 1;
    let c: usize = 4;

    let mut data = Vec::with_capacity(t_full * n * c);
    for ti in 0..t_full {
        for ni in 0..n {
            for ci in 0..c {
                data.push(((ti * 7 + ni * 13 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let full_logits = TestTensor::<3>::from(burn_tensor::TensorData::new(data, [t_full, n, c]));
    let full_log_probs = log_softmax(full_logits, 2);

    // Narrow: skip the first `warmup` frames (simulating warmup slice).
    let narrowed = full_log_probs.clone().narrow(0, warmup, t_eff);

    // Reference: manually extract the same slice as a contiguous tensor.
    // Collect in the backend's native dtype (f32 or f16) to preserve
    // precision when rebuilding the tensor.
    let reference_data: Vec<FloatElem> = full_log_probs
        .clone()
        .slice([warmup..t_full, 0..n, 0..c])
        .into_data()
        .iter::<FloatElem>()
        .collect();
    let reference =
        TestTensor::<3>::from(burn_tensor::TensorData::new(reference_data, [t_eff, n, c]));

    let targets = TestTensorInt::<2>::from([[1, 2]]);
    let input_lengths = TestTensorInt::<1>::from([t_eff as i64]);
    let target_lengths = TestTensorInt::<1>::from([2]);

    let loss_narrowed = ctc_loss(
        narrowed,
        targets.clone(),
        input_lengths.clone(),
        target_lengths.clone(),
        0,
    );
    let loss_reference = ctc_loss(reference, targets, input_lengths, target_lengths, 0);

    let v_narrowed: Vec<f32> = loss_narrowed.into_data().iter::<f32>().collect();
    let v_reference: Vec<f32> = loss_reference.into_data().iter::<f32>().collect();

    assert!(
        (v_narrowed[0] - v_reference[0]).abs() < 1e-3,
        "narrowed loss ({}) diverges from contiguous reference ({})",
        v_narrowed[0],
        v_reference[0],
    );
}

/// Training pipelines typically produce log_probs with a `[B, T, C] ->
/// swap_dims(0, 1) -> [T, B, C] -> narrow(0, warmup, ...)` chain before
/// handing the tensor to CTC. `swap_dims` makes strides non-monotonic,
/// `narrow` introduces a non-zero base offset, and on fused backends the
/// combination has historically produced garbage from manual stride
/// indexing. This guards against that regression with a shape-compatible
/// fixture (B=2 so swap_dims actually permutes strides non-trivially).
#[test]
fn test_ctc_loss_swap_dims_then_narrow() {
    let b: usize = 2;
    let t_full: usize = 8;
    let warmup: usize = 2;
    let t_eff: usize = t_full - warmup;
    let c: usize = 4;

    // Build logits in [B, T, C] layout (as a typical model output would).
    let mut data = Vec::with_capacity(b * t_full * c);
    for bi in 0..b {
        for ti in 0..t_full {
            for ci in 0..c {
                data.push(((bi * 31 + ti * 7 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits_btc = TestTensor::<3>::from(burn_tensor::TensorData::new(data, [b, t_full, c]));
    let log_probs_btc = log_softmax(logits_btc, 2);

    // Permute to [T, B, C] (non-monotonic strides) then narrow away warmup.
    let log_probs_tbc = log_probs_btc.clone().swap_dims(0, 1);
    let narrowed = log_probs_tbc.narrow(0, warmup, t_eff);

    // Reference: materialize the same slice as a contiguous [T, B, C] tensor
    // in the backend's native dtype.
    let reference_data: Vec<FloatElem> = log_probs_btc
        .swap_dims(0, 1)
        .slice([warmup..t_full, 0..b, 0..c])
        .into_data()
        .iter::<FloatElem>()
        .collect();
    let reference =
        TestTensor::<3>::from(burn_tensor::TensorData::new(reference_data, [t_eff, b, c]));

    let targets = TestTensorInt::<2>::from([[1_i64, 2], [2, 3]]);
    let input_lengths = TestTensorInt::<1>::from([t_eff as i64, t_eff as i64]);
    let target_lengths = TestTensorInt::<1>::from([2_i64, 2]);

    let loss_narrowed = ctc_loss(
        narrowed,
        targets.clone(),
        input_lengths.clone(),
        target_lengths.clone(),
        0,
    );
    let loss_reference = ctc_loss(reference, targets, input_lengths, target_lengths, 0);

    let v_narrowed: Vec<f32> = loss_narrowed.into_data().iter::<f32>().collect();
    let v_reference: Vec<f32> = loss_reference.into_data().iter::<f32>().collect();

    for i in 0..b {
        assert!(
            (v_narrowed[i] - v_reference[i]).abs() < 1e-3,
            "sample {i}: swap_dims+narrow loss ({}) diverges from contiguous reference ({})",
            v_narrowed[i],
            v_reference[i],
        );
    }
}

#[test]
fn test_ctc_loss_matches_pytorch() {
    // T=5, N=3, C=4, deterministic logits via sin((t*7 + n*13 + c*3) * 0.1).
    // Expected per-sample losses computed by PyTorch's nn.functional.ctc_loss.
    let t_size: usize = 5;
    let n_size: usize = 3;
    let c_size: usize = 4;

    let mut data = Vec::with_capacity(t_size * n_size * c_size);
    for t in 0..t_size {
        for n in 0..n_size {
            for c in 0..c_size {
                data.push(((t * 7 + n * 13 + c * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits =
        TestTensor::<3>::from(burn_tensor::TensorData::new(data, [t_size, n_size, c_size]));
    let log_probs = log_softmax(logits, 2);

    let targets = TestTensorInt::<2>::from([[1, 2, 0], [1, 0, 0], [3, 2, 1]]);
    let input_lengths = TestTensorInt::<1>::from([5, 5, 5]);
    let target_lengths = TestTensorInt::<1>::from([2, 1, 3]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([
        3.5236570835113525f32,
        3.495313882827759,
        4.262677192687988,
    ]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}
