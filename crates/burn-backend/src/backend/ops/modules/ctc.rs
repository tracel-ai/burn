use burn_std::{Shape, Slice};

use crate::{
    Backend, TensorMetadata, get_device_settings,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};

/// Default CTC loss implementation using the forward (alpha) algorithm.
///
/// Computes the Connectionist Temporal Classification loss by summing over
/// all valid alignments between the input and target sequences.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of shape `[T, N, C]`
/// * `targets` - Target indices of shape `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `target_lengths` - Actual target lengths per batch element `[N]`
/// * `blank` - Index of the blank label
///
/// # Returns
///
/// Per-sample loss of shape `[N]`
pub fn ctc_loss_default<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    target_lengths: IntTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let alpha = AlphaCtx::<B>::compute(
        log_probs,
        &targets,
        input_lengths,
        target_lengths.clone(),
        blank,
    );
    extract_loss::<B>(&alpha, target_lengths)
}

/// Compose the CTC gradient w.r.t. `log_probs` from pre-computed alpha, beta, and nll.
///
/// The T-iteration alpha and beta recursions are the dominant cost of the backward
/// pass. Backends that fuse those recursions into a single kernel launch can call
/// this helper to reuse the gradient composition.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities `[T, N, C]`
/// * `targets` - Target label indices `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `grad_loss` - Upstream gradient w.r.t. the per-sample loss `[N]`
/// * `log_alpha_full` - Alpha recursion output `[T, N, 2S+1]`
/// * `log_beta_full` - Beta recursion output `[T, N, 2S+1]`
/// * `nll` - Per-sample negative log-likelihood (forward loss) `[N]`
/// * `blank` - Index of the blank label
#[allow(clippy::too_many_arguments)]
pub fn ctc_grad_from_alpha_beta_default<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    grad_loss: FloatTensor<B>,
    log_alpha_full: FloatTensor<B>,
    log_beta_full: FloatTensor<B>,
    nll: FloatTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime_len = 2 * max_target_len + 1;
    let device = B::float_device(&log_probs);
    let int_dtype: burn_std::IntDType = targets.dtype().into();
    let settings = get_device_settings::<B>(&device);

    let blank_inserted_targets = insert_blanks::<B>(
        &targets,
        batch_size,
        max_target_len,
        max_l_prime_len,
        blank,
        &device,
        int_dtype,
    );

    // Both log_alpha[t, n, s] and log_beta[t, n, s] include a factor of
    // log_probs[t, n, l'[s]] (added on every recursion step). The CTC paper's
    // alpha_hat * beta_hat product divides one of those factors out, so we
    // subtract log_probs[t, n, l'[s]] when forming log_post.
    //
    // We then divide by total_prob = exp(-nll) to obtain the alignment
    // posterior, which in log space means *adding* nll (since nll = -log P,
    // dividing by P is adding nll). Per PyTorch's CTC backward kernel:
    //   log_post[t, n, s] = log_alpha + log_beta - log_probs[t, n, l'[s]] - log P
    //                     = log_alpha + log_beta - log_probs[t, n, l'[s]] + nll
    let indices_3d = B::int_reshape(
        blank_inserted_targets,
        Shape::new([1, batch_size, max_l_prime_len]),
    );
    let indices_3d = B::int_expand(
        indices_3d,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let log_probs_at_l = B::float_gather(2, log_probs.clone(), indices_3d.clone());

    // Samples with an unreachable target yield nll = +inf. For those, log_alpha
    // stays at -inf at many (t, s) while log_beta is finite at the boundary, so
    // log_post = (-inf) + finite - finite + (+inf) = NaN and -exp(NaN) = NaN
    // contaminates the gradient. `NaN * 0 = NaN` under IEEE 754, so zero_infinity
    // masking on the outer grad_loss can't clear it. Capture the mask now and
    // zero the gradient for those samples at the end.
    let nll_is_inf = B::float_is_inf(nll.clone(), settings.bool_dtype);

    let nll_b = B::float_reshape(nll, Shape::new([1, batch_size, 1]));
    let nll_b = B::float_expand(
        nll_b,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let log_post = B::float_add(
        B::float_sub(B::float_add(log_alpha_full, log_beta_full), log_probs_at_l),
        nll_b,
    );

    // grad starts as exp(log_probs) * grad_loss[None, :, None].
    let grad_loss_3d = B::float_reshape(grad_loss, Shape::new([1, batch_size, 1]));
    let grad_loss_b = B::float_expand(
        grad_loss_3d.clone(),
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let mut grad = B::float_mul(B::float_exp(log_probs), grad_loss_b);

    // Subtract sum over s of grad_loss[n] * exp(log_post[t, n, s]) at index l'[n, s].
    let grad_loss_post = B::float_expand(
        grad_loss_3d,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let scatter_value = B::float_neg(B::float_mul(B::float_exp(log_post), grad_loss_post));

    grad = B::float_scatter_add(2, grad, indices_3d, scatter_value);

    // Mask out timesteps where t >= input_lengths[n].
    let t_indices = B::int_arange(0..max_input_length as i64, &device, int_dtype);
    let t_indices = B::int_reshape(t_indices, Shape::new([max_input_length, 1, 1]));
    let t_indices = B::int_expand(
        t_indices,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let il_b = B::int_reshape(input_lengths, Shape::new([1, batch_size, 1]));
    let il_b = B::int_expand(
        il_b,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let oob_mask = B::int_greater_equal(t_indices, il_b, settings.bool_dtype);

    // Broadcast the nll-is-inf mask across [T, N, C] and OR with oob_mask so a
    // single mask_fill zeros both unreachable samples and out-of-bound timesteps.
    let nll_inf_b = B::bool_reshape(nll_is_inf, Shape::new([1, batch_size, 1]));
    let nll_inf_b = B::bool_expand(
        nll_inf_b,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let mask = B::bool_or(oob_mask, nll_inf_b);
    B::float_mask_fill(grad, mask, 0.0.into())
}

/// Cached state from the alpha recursion. Only `last` is consumed by
/// `ctc_loss_default` (via `extract_loss`); the other fields hold intermediate
/// products that backends with a native backward kernel could reuse if wired
/// up. They are kept here to document the recursion's outputs.
#[allow(dead_code)]
struct AlphaCtx<B: Backend> {
    /// `log_alpha[T, N, 2S+1]` (full history).
    full: FloatTensor<B>,
    /// `log_alpha[T-1, :, :]` (last timestep; used to read out the loss).
    last: FloatTensor<B>,
    /// `l'` after blank insertion `[N, 2S+1]`.
    blank_inserted_targets: IntTensor<B>,
    /// `log_probs[t, n, l'[n, s]]` pre-gathered as `[T, N, 2S+1]`.
    log_probs_at_l_full: FloatTensor<B>,
    max_l_prime_len: usize,
}

impl<B: Backend> AlphaCtx<B> {
    fn compute(
        log_probs: FloatTensor<B>,
        targets: &IntTensor<B>,
        input_lengths: IntTensor<B>,
        target_lengths: IntTensor<B>,
        blank: usize,
    ) -> Self {
        let log_probs_shape = log_probs.shape();
        let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
        let target_shape = targets.shape();
        let max_target_len = target_shape.dims::<2>()[1];
        let device = B::float_device(&log_probs);
        let float_dtype: burn_std::FloatDType = log_probs.dtype().into();
        let int_dtype: burn_std::IntDType = targets.dtype().into();
        let settings = get_device_settings::<B>(&device);

        let max_l_prime_len = 2 * max_target_len + 1;
        let blank_inserted_targets = insert_blanks::<B>(
            targets,
            batch_size,
            max_target_len,
            max_l_prime_len,
            blank,
            &device,
            int_dtype,
        );

        // Pre-allocate the full alpha tensor [T, N, 2S+1] filled with -inf.
        let mut alpha_full = B::float_full(
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );

        // Initialize alpha[0, :, 0] = log_probs[0, :, blank]
        // and alpha[0, :, 1] = log_probs[0, :, l'[1]].
        let log_probs_t0 = B::float_slice(
            log_probs.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        let log_probs_t0 = B::float_reshape(log_probs_t0, Shape::new([batch_size, num_classes]));

        let first_blank = B::int_slice(
            blank_inserted_targets.clone(),
            &[Slice::full(), Slice::new(0, Some(1), 1)],
        );
        let log_prob_blank = B::float_gather(1, log_probs_t0.clone(), first_blank);
        // Broadcast to [1, N, 1] for slice_assign into alpha_full.
        let log_prob_blank_3d = B::float_reshape(log_prob_blank, Shape::new([1, batch_size, 1]));
        alpha_full = B::float_slice_assign(
            alpha_full,
            &[
                Slice::new(0, Some(1), 1),
                Slice::full(),
                Slice::new(0, Some(1), 1),
            ],
            log_prob_blank_3d,
        );

        if max_l_prime_len > 1 {
            let first_label = B::int_slice(
                blank_inserted_targets.clone(),
                &[Slice::full(), Slice::new(1, Some(2), 1)],
            );
            let log_prob_first = B::float_gather(1, log_probs_t0, first_label);
            let log_prob_first_3d =
                B::float_reshape(log_prob_first, Shape::new([1, batch_size, 1]));
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(0, Some(1), 1),
                    Slice::full(),
                    Slice::new(1, Some(2), 1),
                ],
                log_prob_first_3d,
            );
        }

        // Track the latest row separately for the recursion (cheaper than
        // re-slicing alpha_full each iteration).
        let mut log_alpha = B::float_slice(
            alpha_full.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        log_alpha = B::float_reshape(log_alpha, Shape::new([batch_size, max_l_prime_len]));

        let l_prime_mask = create_l_prime_mask::<B>(
            &blank_inserted_targets,
            batch_size,
            max_l_prime_len,
            blank,
            &device,
            int_dtype,
            settings.bool_dtype,
        );
        let s_mask = create_s_mask::<B>(
            &target_lengths,
            batch_size,
            max_l_prime_len,
            &device,
            int_dtype,
            settings.bool_dtype,
        );

        // Hoist out of the T-loop: padding tensors for right_shift (same
        // value/shape at every iteration) and the full `[T, N, 2S+1]`
        // gather of log_probs at l' (one T-sized gather replaces T small
        // gathers).
        let pad_1 = B::float_full(
            Shape::new([batch_size, 1]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let pad_2 = B::float_full(
            Shape::new([batch_size, 2]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let indices_3d = B::int_expand(
            B::int_reshape(
                blank_inserted_targets.clone(),
                Shape::new([1, batch_size, max_l_prime_len]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let log_probs_at_l_full = B::float_gather(2, log_probs.clone(), indices_3d);

        // Precompute `combined_mask_all[t, n, s] = (input_lengths[n] > t) AND
        // s_mask[n, s]` for every t in one shot. The T-loop reads its row via
        // a metadata-only slice instead of recomputing the `int_greater_elem`
        // + bool_and per iteration.
        let t_indices_2d = B::int_expand(
            B::int_reshape(
                B::int_arange(0..max_input_length as i64, &device, int_dtype),
                Shape::new([max_input_length, 1]),
            ),
            Shape::new([max_input_length, batch_size]),
        );
        let il_tn = B::int_expand(
            B::int_reshape(input_lengths.clone(), Shape::new([1, batch_size])),
            Shape::new([max_input_length, batch_size]),
        );
        let t_mask_all = B::bool_expand(
            B::bool_reshape(
                B::int_greater(il_tn, t_indices_2d, settings.bool_dtype),
                Shape::new([max_input_length, batch_size, 1]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let s_mask_bcast = B::bool_expand(
            B::bool_reshape(s_mask.clone(), Shape::new([1, batch_size, max_l_prime_len])),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let combined_mask_all = B::bool_and(t_mask_all, s_mask_bcast);

        for t in 1..max_input_length {
            let combined_mask = B::bool_reshape(
                B::bool_slice(
                    combined_mask_all.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );

            let log_alpha_s = log_alpha.clone();
            let log_alpha_s_m1 = right_shift::<B>(&log_alpha, &pad_1, max_l_prime_len, 1);
            let log_alpha_s_m2 = right_shift::<B>(&log_alpha, &pad_2, max_l_prime_len, 2);

            let bar = log_sum_exp::<B>(log_alpha_s, log_alpha_s_m1, settings.bool_dtype);
            let bar_with_skip = log_sum_exp::<B>(bar.clone(), log_alpha_s_m2, settings.bool_dtype);
            let log_alpha_combined = B::float_mask_where(bar, l_prime_mask.clone(), bar_with_skip);

            // Slice row t from the pre-gathered `[T, N, 2S+1]` tensor.
            let log_probs_at_l = B::float_reshape(
                B::float_slice(
                    log_probs_at_l_full.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );
            let new_alpha = B::float_add(log_alpha_combined, log_probs_at_l);
            log_alpha = B::float_mask_where(log_alpha, combined_mask, new_alpha);

            let log_alpha_3d = B::float_reshape(
                log_alpha.clone(),
                Shape::new([1, batch_size, max_l_prime_len]),
            );
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(t as isize, Some(t as isize + 1), 1),
                    Slice::full(),
                    Slice::full(),
                ],
                log_alpha_3d,
            );
        }

        Self {
            full: alpha_full,
            last: log_alpha,
            blank_inserted_targets,
            log_probs_at_l_full,
            max_l_prime_len,
        }
    }
}

/// Extract the per-sample loss from the last alpha row.
fn extract_loss<B: Backend>(alpha: &AlphaCtx<B>, target_lengths: IntTensor<B>) -> FloatTensor<B> {
    let log_alpha_shape = alpha.last.shape();
    let [batch_size, _] = log_alpha_shape.dims::<2>();
    let device = B::float_device(&alpha.last);
    let settings = get_device_settings::<B>(&device);

    let last_blank_idx = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let last_blank_idx = B::int_reshape(last_blank_idx, Shape::new([batch_size, 1]));
    let last_label_idx = B::int_clamp_min(
        B::int_sub_scalar(last_blank_idx.clone(), 1.into()),
        0.into(),
    );

    let log_alpha_last_blank = B::float_gather(1, alpha.last.clone(), last_blank_idx);
    let log_alpha_last_blank = B::float_reshape(log_alpha_last_blank, Shape::new([batch_size]));

    let log_alpha_last_label = B::float_gather(1, alpha.last.clone(), last_label_idx);
    let log_alpha_last_label = B::float_reshape(log_alpha_last_label, Shape::new([batch_size]));

    // For target_lengths == 0, last_label is meaningless: substitute -inf.
    let target_len_zero = B::int_equal_elem(target_lengths, 0.into(), settings.bool_dtype);
    let log_alpha_last_label = B::float_mask_fill(
        log_alpha_last_label,
        target_len_zero,
        f32::NEG_INFINITY.into(),
    );

    let log_likelihood = log_sum_exp::<B>(
        log_alpha_last_blank,
        log_alpha_last_label,
        settings.bool_dtype,
    );
    B::float_neg(log_likelihood)
}

/// Insert blank labels between each target label: [b, l1, b, l2, ..., b]
fn insert_blanks<B: Backend>(
    targets: &IntTensor<B>,
    batch_size: usize,
    max_target_len: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
) -> IntTensor<B> {
    let result = B::int_full(
        Shape::new([batch_size, max_l_prime_len]),
        (blank as i64).into(),
        device,
        int_dtype,
    );

    if max_target_len == 0 {
        return result;
    }

    // Place every target label at odd columns {1, 3, 5, ...} in one
    // strided slice_assign, equivalent to `result[:, 1::2] = targets`.
    B::int_slice_assign(
        result,
        &[Slice::full(), Slice::new(1, None, 2)],
        targets.clone(),
    )
}

/// Right-shift a 2D float tensor by `shift` positions, prepending the
/// pre-allocated `padding` tensor (shape `[batch_size, shift]`, value
/// `-inf`) instead of materializing it each call.
///
/// Called inside the T-loop of the alpha recursion; hoisting the padding
/// out of the loop eliminates `O(T)` `float_full` allocations.
fn right_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    padding: &FloatTensor<B>,
    cols: usize,
    shift: usize,
) -> FloatTensor<B> {
    // Shifting by more than the column count pushes every data slot off
    // the right. Avoid the `cols - shift` usize underflow when
    // `max_target_len == 0` (so `max_l_prime_len == 1`) by narrowing the
    // all-`-inf` padding down to `cols`.
    if cols < shift {
        return B::float_slice(
            padding.clone(),
            &[Slice::full(), Slice::new(0, Some(cols as isize), 1)],
        );
    }
    let shortened = B::float_slice(
        tensor.clone(),
        &[
            Slice::full(),
            Slice::new(0, Some((cols - shift) as isize), 1),
        ],
    );
    B::float_cat(alloc::vec![padding.clone(), shortened], 1)
}

/// Compute `log(exp(a) + exp(b))` in a numerically stable way.
///
/// `log_sum_exp(a, b) = max(a, b) + log1p(exp(-|a - b|))`. The edge case is
/// `a = b = -inf`, where `-|(-inf) - (-inf)| = NaN`; we detect `max == -inf`
/// and substitute a `-inf` diff so the final sum stays `-inf` (both via the
/// mask and because `log1p(exp(-inf)) = 0`). Gradient-safe: no `NaN` flows
/// through the forward intermediates when inputs are `-inf`.
///
/// Precondition: inputs must be `<= 0` (log-probabilities). `+inf` inputs are
/// not guarded and produce `NaN`; callers outside the CTC recursion should
/// validate this themselves.
fn log_sum_exp<B: Backend>(
    a: FloatTensor<B>,
    b: FloatTensor<B>,
    bool_dtype: burn_std::BoolDType,
) -> FloatTensor<B> {
    // `-inf` values in `a` or `b` would make `a - b` evaluate to `NaN`
    // (when both are `-inf`) and the backward pass through that `NaN`
    // intermediate propagates `NaN` into the gradient even when the
    // forward mask discards it (`0 * NaN = NaN` in IEEE). Clamp `-inf`
    // to `0` on safe copies used only for the diff computation; compute
    // `max` on the original values so its output is correct in the
    // `-inf` cases.
    let a_is_neg_inf = B::float_equal_elem(a.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let b_is_neg_inf = B::float_equal_elem(b.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let either_neg_inf = B::bool_or(a_is_neg_inf.clone(), b_is_neg_inf.clone());

    let a_safe = B::float_mask_fill(a.clone(), a_is_neg_inf, 0.0.into());
    let b_safe = B::float_mask_fill(b.clone(), b_is_neg_inf, 0.0.into());

    let lt_mask = B::float_lower(a.clone(), b.clone(), bool_dtype);
    let mx = B::float_mask_where(a, lt_mask, b);

    // diff_safe = -|a_safe - b_safe|. Finite by construction. When either
    // input was `-inf`, force it to `-inf` so `exp(diff) == 0` and the
    // `log1p` term contributes nothing (`result = mx`). When both were
    // `-inf`, `mx = -inf` so `result = -inf + 0 = -inf`.
    let diff_safe = B::float_neg(B::float_abs(B::float_sub(a_safe, b_safe)));
    let diff_final = B::float_mask_fill(diff_safe, either_neg_inf, f32::NEG_INFINITY.into());

    B::float_add(mx, B::float_log1p(B::float_exp(diff_final)))
}

/// Mask for the alpha skip transition: `l'[s] != blank AND l'[s] != l'[s-2] AND s >= 2`.
fn create_l_prime_mask<B: Backend>(
    blank_inserted_targets: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
    bool_dtype: burn_std::BoolDType,
) -> BoolTensor<B> {
    // The mask requires `s >= 2`, which is unsatisfiable when max_l_prime_len < 2
    // (i.e. targets have shape [N, 0]). Bail out before the `max_l_prime_len - 2`
    // usize subtraction underflows.
    if max_l_prime_len < 2 {
        return B::bool_zeros(
            Shape::new([batch_size, max_l_prime_len]),
            device,
            bool_dtype,
        );
    }
    let l_prime = blank_inserted_targets.clone();

    let not_blank = B::int_not_equal_elem(l_prime.clone(), (blank as i64).into(), bool_dtype);

    let l_prime_shifted = {
        let padding = B::int_full(
            Shape::new([batch_size, 2]),
            (blank as i64).into(),
            device,
            int_dtype,
        );
        let shortened = B::int_slice(
            l_prime.clone(),
            &[
                Slice::full(),
                Slice::new(0, Some((max_l_prime_len - 2) as isize), 1),
            ],
        );
        B::int_cat(alloc::vec![padding, shortened], 1)
    };
    let not_equal_s_m2 = B::int_not_equal(l_prime, l_prime_shifted, bool_dtype);

    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));
    let s_ge_2 = B::int_greater_equal_elem(col_indices, 2.into(), bool_dtype);

    B::bool_and(B::bool_and(not_blank, not_equal_s_m2), s_ge_2)
}

/// Create a mask for valid s positions: s < 2 * target_length + 1
fn create_s_mask<B: Backend>(
    target_lengths: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
    bool_dtype: burn_std::BoolDType,
) -> BoolTensor<B> {
    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));

    let lengths = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let lengths = B::int_add_scalar(lengths, 1.into());
    let lengths = B::int_reshape(lengths, Shape::new([batch_size, 1]));
    let lengths = B::int_expand(lengths, Shape::new([batch_size, max_l_prime_len]));

    B::int_lower(col_indices, lengths, bool_dtype)
}
