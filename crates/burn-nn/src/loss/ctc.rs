use super::Reduction;
use alloc::vec;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Bool, Element, Int, Tensor, backend::Backend, s};
use burn_core as burn;
use burn_core::tensor::Numeric;
use core::f32;

/// Configuration for the [CTC Loss](CTCLoss) module.
#[derive(Config, Debug)]
pub struct CTCLossConfig {
    /// The index number used to represent the blank label. Default value is `0`.
    #[config(default = 0)]
    pub blank: usize,
    /// Whether to zero infinite losses and the associated gradients. Default value is `false`.
    #[config(default = false)]
    pub zero_infinity: bool,
}

impl CTCLossConfig {
    /// Initialize a new [CTC Loss](CTCLoss) module
    pub fn init(&self) -> CTCLoss {
        CTCLoss {
            blank: self.blank,
            zero_infinity: self.zero_infinity,
        }
    }
}

/// Computes the Connectionist Temporal Classification (CTC) loss.
///
/// Calculates the loss between a continuous (unsegmented) time series and a target sequence.
/// CTC sums over the probability of all possible alignments of the input to the target,
/// producing a loss value that is differentiable with respect to each input node.
///
/// The input to this loss is expected to be **log-probabilities** (e.g,, via `log_softmax`),
/// not raw logits.
///
/// # References
///
/// - [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
///
/// # Example
///
/// ```rust,ignore
/// use burn::tensor::{Tensor, Int};
/// use burn::tensor::activation::log_softmax;
/// use burn::nn::loss::{CTCLossConfig, CTCLoss};
///
/// let device = Default::default();
///
/// // Initialize CTC Loss with default configuration
/// let ctc_loss = CTCLossConfig::new().init();
///
/// // Initialize CTC Loss with custom configuration
/// let ctc_loss = CTCLossConfig::new()
///     .with_blank(1)
///     .with_zero_infinity(true)
///     .init();
///
/// // Prepare inputs (Logits shape: [Time, Batch, Class])
/// // In your actual code, the logits would be the output of your model
/// let logits = Tensor::<B, 3>::ones([10, 2, 5], &device);
/// let log_probs = log_softmax(logits, 2);
///
/// // Targets shape: [Batch, Max_Target_Len]
/// // Note: Targets should not contain the blank index (1).
/// let targets = Tensor::<B, 2, Int>::from_data([[0, 2], [3, 4]], &device);
///
/// // Lengths shape: [Batch]
/// let input_lengths = Tensor::<B, 1, Int>::from_data([10, 8], &device);
/// let target_lengths = Tensor::<B, 1, Int>::from_data([2, 2], &device);
///
/// // Compute loss
/// let loss = ctc_loss.forward(log_probs, targets, input_lengths, target_lengths);
/// ```
#[derive(Module, Clone, Debug)]
pub struct CTCLoss {
    blank: usize,
    zero_infinity: bool,
}

impl CTCLoss {
    /// Computes the CTC loss for the input log-probabilities and targets with no reduction applied.
    ///
    /// # Arguments
    ///
    /// - `log_probs`: The log-probabilities of the outputs (e.g., from `log_softmax`).
    /// - `targets`: A 2D tensor containing the target class indices. These indices should not
    ///   include the blank index used in CTC loss. The targets are padded to the length of the longest sequence.
    /// - `input_lengths`: A 1D tensor containing the actual length of the input sequence for each batch. This
    ///   allows retrieving the actual sequence of log-probabilities from `log_probs` if the batch contains
    ///   sequences of varying lengths.
    /// - `target_lengths`: A 1D tensor containing the actual length of the target sequence for each target
    ///   sequence in `targets`.
    ///
    /// # Returns
    ///
    /// - A 1D tensor of shape `[batch_size]` containing the loss for each sample.
    ///
    /// # Shapes
    ///
    /// - `log_probs`: `[time_steps, batch_size, num_classes]` where `num_classes` includes blank.
    /// - `targets`: `[batch_size, max_target_length]`
    /// - `input_lengths`: `[batch_size]`
    /// - `target_lengths`: `[batch_size]`
    pub fn forward<B: Backend>(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = log_probs.device();
        let [max_input_length, batch_size, num_classes] = log_probs.dims(); // [T, N, C]
        let max_target_len = targets.dims()[1];
        let input_lengths_len = input_lengths.dims()[0];
        let target_lengths_len = target_lengths.dims()[0];
        self.assertions(
            batch_size,
            num_classes,
            targets.clone(),
            input_lengths_len,
            target_lengths_len,
        );

        // Build the modified label sequence l' by inserting blanks around every label
        let blank_inserted_targets =
            self.insert_blanks::<B>(&targets, batch_size, max_target_len, &device);

        // Initialize the forward variable alpha
        let max_l_prime_len = 2 * max_target_len + 1;
        let mut log_alpha_t_s =
            Tensor::<B, 2>::full([batch_size, max_l_prime_len], f32::NEG_INFINITY, &device);
        log_alpha_t_s = self.initialize_log_alpha(
            log_probs.clone(),
            blank_inserted_targets.clone(),
            log_alpha_t_s,
        );

        let l_prime_combined_mask =
            self.create_l_prime_mask(blank_inserted_targets.clone(), &device);
        let s_mask =
            self.create_s_mask(max_l_prime_len, batch_size, target_lengths.clone(), &device);

        // Loop over time steps since an arbitrary time step t depends on t - 1
        for t in 1..max_input_length {
            let combined_s_t_mask = self.create_combined_s_t_mask(
                input_lengths.clone(),
                t,
                batch_size,
                max_l_prime_len,
                s_mask.clone(),
            );
            log_alpha_t_s = self.compute_log_alpha_t_s(
                t,
                combined_s_t_mask,
                log_alpha_t_s,
                l_prime_combined_mask.clone(),
                log_probs.clone(),
                blank_inserted_targets.clone(),
            );
        }

        let last_blank_indices = target_lengths.mul_scalar(2).reshape([batch_size, 1]);
        let last_label_indices = last_blank_indices.clone().sub_scalar(1);
        let log_alpha_last_blank = log_alpha_t_s
            .clone()
            .gather(1, last_blank_indices)
            .squeeze_dim::<1>(1);
        let log_alpha_last_label = log_alpha_t_s
            .clone()
            .gather(1, last_label_indices)
            .squeeze_dim::<1>(1);
        let log_likelihood = self.log_sum_exp(log_alpha_last_blank, log_alpha_last_label, &device);
        let mut ctc_loss_tensor = log_likelihood.neg();

        if self.zero_infinity {
            let inf_mask = ctc_loss_tensor.clone().is_inf();
            ctc_loss_tensor = ctc_loss_tensor
                .clone()
                .mask_where(inf_mask, ctc_loss_tensor.clone().zeros_like());
        }

        ctc_loss_tensor
    }

    /// Computes the CTC loss for the input log-probabilities and targets with reduction.
    ///
    /// # Arguments
    ///
    /// - `log_probs`: The log-probabilities of the outputs (e.g., from `log_softmax`).
    /// - `targets`: A 2D tensor containing the target class indices. These indices should not
    ///   include the blank index used in CTC loss. The targets are padded to the length of the longest sequence.
    /// - `input_lengths`: A 1D tensor containing the actual length of the input sequence for each batch. This
    ///   allows retrieving the actual sequence of log-probabilities from `log_probs` if the batch contains
    ///   sequences of varying lengths.
    /// - `target_lengths`: A 1D tensor containing the actual length of the target sequence for each target
    ///   sequence in `targets`.
    /// - `reduction`: The reduction stratey to apply to the loss tensor containing the CTC loss values for
    ///   each sample (e.g., mean, sum).
    ///
    /// # Returns
    ///
    /// - A 1D tensor of shape `[1]` containing the reduced loss value.
    ///
    /// # Shapes
    ///
    /// - `log_probs`: `[time_steps, batch_size, num_classes]` where `num_classes` includes blank.
    /// - `targets`: `[batch_size, max_target_length]`
    /// - `input_lengths`: `[batch_size]`
    /// - `target_lengths`: `[batch_size]`
    ///
    /// # Panics
    /// - If `reduction` is not one of `Reduction::Auto`, `Reduction::Mean`, and `Reduction::Sum`.
    /// - If `blank` index is greater than or equal to `num_classes`.
    /// - If the batch dimension of `log_probs`, `targets`, `input_lengths`, and `target_lengths` do not match.
    pub fn forward_with_reduction<B: Backend>(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let ctc_loss_tensor =
            self.forward(log_probs, targets, input_lengths, target_lengths.clone());

        match reduction {
            Reduction::Auto | Reduction::Mean => {
                // Following PyTorch's behavior where the output losses are divided
                // by the target lengths and then the mean over the batch is taken
                let target_lengths_float = target_lengths.float();
                ctc_loss_tensor.div(target_lengths_float).mean()
            }
            Reduction::Sum => ctc_loss_tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    fn assertions<B: Backend>(
        &self,
        batch_size: usize,
        num_classes: usize,
        targets: Tensor<B, 2, Int>,
        input_lengths_len: usize,
        target_lengths_len: usize,
    ) {
        assert!(
            self.blank < num_classes,
            "blank index {} must be less than num_classes {}",
            self.blank,
            num_classes
        );
        assert_eq!(
            targets.dims()[0],
            batch_size,
            "targets batch dimension {} must equal batch_size {}",
            targets.dims()[0],
            batch_size
        );
        assert_eq!(
            input_lengths_len, batch_size,
            "input_lengths length {} must equal batch_size {}",
            input_lengths_len, batch_size
        );
        assert_eq!(
            target_lengths_len, batch_size,
            "target_lengths length {} must equal batch_size {}",
            target_lengths_len, batch_size
        );
    }

    fn insert_blanks<B: Backend>(
        &self,
        targets: &Tensor<B, 2, Int>,
        batch_size: usize,
        max_target_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        // The modified label sequences have (max_target_len + 1) blank labels
        let blank_tensor = Tensor::<B, 2, Int>::full(
            [batch_size, 2 * max_target_len + 1],
            self.blank as i64,
            device,
        );

        blank_tensor.slice_assign(s![.., 1..;2], targets.clone())
    }

    fn initialize_log_alpha<B: Backend>(
        &self,
        log_probs: Tensor<B, 3>,
        blank_inserted_targets: Tensor<B, 2, Int>,
        log_alpha_t_s: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Given alpha_t(s), we have:
        // alpha_1(1) = (y_blank)^1  => log_alpha_1(1) = ln(y_blank)^1
        // alpha_1(2) = (y_l1)^1  => log_alpha_1(2) = ln(y_l1)^1
        // alpha_1(s) = 0 (for every s > 2)  => log_alpha_1(s) = -neg_inf
        let log_probs_t0 = log_probs
            .clone()
            .slice(s![0..1, .., ..])
            .squeeze_dim::<2>(0); // shape: [N, C]

        // log_alpha shape: [N, 2*S+1]
        // log_probs shape: [T, N, C]
        // log_alpha[:, 0] = log_probs[0, :, blank]
        let first_blank = blank_inserted_targets.clone().slice(s![.., 0..1]); // [N, 1]
        // log_probs_t0 have C columns where each represents a unique class (includes blank)
        let log_prob_blank = log_probs_t0.clone().gather(1, first_blank); // [N, 1]
        let temp_log_alpha_t_s = log_alpha_t_s.slice_assign(s![.., 0..1], log_prob_blank);

        // log_alpha[:, 1] = log_probs[0, :, targets[:, 0]]
        let first_label = blank_inserted_targets.clone().slice(s![.., 1..2]); // [N, 1]
        let log_prob_first_label = log_probs_t0.gather(1, first_label); // [N, 1]
        temp_log_alpha_t_s.slice_assign(s![.., 1..2], log_prob_first_label)
    }

    fn right_shift_2d_tensor<B: Backend, K>(
        &self,
        org_2d_tensor: Tensor<B, 2, K>,
        shift_by: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, K>
    where
        K: Numeric<B>,
        K::Elem: Element,
    {
        assert!(
            shift_by == 1 || shift_by == 2,
            "The parameter shift_by must 1 or 2"
        );

        let [rows, cols] = org_2d_tensor.dims();
        let padding_shape = [rows, shift_by];
        let padding_tensor = if org_2d_tensor.dtype().is_float() {
            Tensor::<B, 2, K>::full(padding_shape, f32::NEG_INFINITY, device)
        } else {
            Tensor::<B, 2, K>::full(padding_shape, 0, device)
        };
        let org_tensor_shortened = org_2d_tensor.slice(s![.., ..cols - shift_by]);

        Tensor::cat(vec![padding_tensor, org_tensor_shortened], 1)
    }

    fn create_l_prime_mask<B: Backend>(
        &self,
        blank_inserted_targets: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 2, Bool> {
        let l_prime_s = blank_inserted_targets.clone();
        let l_prime_s_minus_2 =
            self.right_shift_2d_tensor(blank_inserted_targets.clone(), 2, device);

        // Create a single mask that is true for entries where alpha_{t-1}(s - 2) should also
        // be added to compute alpha_{t}(s)
        let s_is_not_blank_mask = l_prime_s.clone().not_equal_elem(self.blank as i64);
        let s_not_equal_s_minus_2_mask = l_prime_s.not_equal(l_prime_s_minus_2);

        s_is_not_blank_mask.bool_and(s_not_equal_s_minus_2_mask)
    }

    fn create_s_mask<B: Backend>(
        &self,
        max_l_prime_len: usize,
        batch_size: usize,
        target_lengths: Tensor<B, 1, Int>,
        device: &B::Device,
    ) -> Tensor<B, 2, Bool> {
        let col_indices = Tensor::<B, 1, Int>::arange(0..max_l_prime_len as i64, device)
            .reshape([1, max_l_prime_len]);
        let col_indices_expanded = col_indices.expand([batch_size, max_l_prime_len]);
        let blank_inserted_target_lengths = target_lengths
            .mul_scalar(2)
            .add_scalar(1)
            .reshape([batch_size, 1]);
        let target_lengths_expanded =
            blank_inserted_target_lengths.expand([batch_size, max_l_prime_len]);

        col_indices_expanded.lower(target_lengths_expanded)
    }

    fn log_sum_exp<const D: usize, B: Backend>(
        &self,
        log_tensor1: Tensor<B, D>,
        log_tensor2: Tensor<B, D>,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let shape = log_tensor1.dims();
        let ones_tensor = Tensor::<B, D>::ones(shape, device);

        // Let A and B represent parameters tensor1 and tensor2 respectively.
        // Let C be the tensor this method returns.
        // If an entry in both A and B are neg_inf, then the same entry
        // in C should also contain neg_inf.
        // If an entry in only one of A or B is neg_inf, then the same entry in
        // C should contain the value of the other tensor entry which is not neg_inf.
        let tensor1_is_neg_inf = log_tensor1.clone().equal_elem(f32::NEG_INFINITY);
        let tensor2_is_neg_inf = log_tensor2.clone().equal_elem(f32::NEG_INFINITY);
        let temp_tensor1 = ones_tensor
            .clone()
            .mask_where(tensor1_is_neg_inf.clone(), log_tensor2.clone());
        let neg_inf_lse_tensor =
            temp_tensor1.mask_where(tensor2_is_neg_inf.clone(), log_tensor1.clone());

        // Create sanitized tensors for math operations to prevent NaN. Replace neg_inf
        // with 0.0. The tensor neg_inf_lse_tensor contains correct values for entries
        // where at least one of the corresponding entries in log_tensor1 or log_tensor2
        // is neg_inf. Hence, the math operations below is computing the values for entries
        // that are not already filled with their actual/correct values. Thus, result for
        // these positions (where we sanitize) are not used anyway since the
        // unfilled_entries_mask is applied at the end.
        let tensor1_safe = log_tensor1
            .clone()
            .mask_fill(tensor1_is_neg_inf.clone(), 0.0);
        let tensor2_safe = log_tensor2
            .clone()
            .mask_fill(tensor2_is_neg_inf.clone(), 0.0);

        // Create a mask which contains true for entries whose values were not
        // set by operations above
        let filled_entries_mask = tensor1_is_neg_inf.bool_or(tensor2_is_neg_inf);
        let unfilled_entries_mask = filled_entries_mask.bool_not();

        let max_tensor = tensor1_safe.clone().max_pair(tensor2_safe.clone());
        let diff_tensor = tensor1_safe.sub(tensor2_safe);
        let exp_tensor = diff_tensor.abs().neg().exp();
        let ln_tensor = ones_tensor.add(exp_tensor).log();
        let lse_tensor = max_tensor.add(ln_tensor);
        neg_inf_lse_tensor.mask_where(unfilled_entries_mask, lse_tensor)
    }

    fn create_combined_s_t_mask<B: Backend>(
        &self,
        input_lengths: Tensor<B, 1, Int>,
        t: usize,
        batch_size: usize,
        max_l_prime_len: usize,
        s_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 2, Bool> {
        // Create masks for valid t and s
        let t_mask_1d = input_lengths
            .clone()
            .greater_elem(t as i64)
            .reshape([batch_size, 1]);
        let t_mask = t_mask_1d.expand([batch_size, max_l_prime_len]);

        t_mask.bool_and(s_mask.clone())
    }

    fn compute_log_alpha_t_s<B: Backend>(
        &self,
        t: usize,
        combined_s_t_mask: Tensor<B, 2, Bool>,
        log_alpha_t_s: Tensor<B, 2>,
        l_prime_combined_mask: Tensor<B, 2, Bool>,
        log_probs: Tensor<B, 3>,
        blank_inserted_targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let device = log_probs.device();
        let log_alpha_t_minus_1 = log_alpha_t_s.clone();

        // No move from last time step: alpha_{t-1}(s)
        let log_alpha_s = log_alpha_t_minus_1.clone();

        // Single move from last time step: alpha_{t-1}(s - 1)
        let log_alpha_s_minus_1 =
            self.right_shift_2d_tensor(log_alpha_t_minus_1.clone(), 1, &device);

        // A skip move (moving 2 positions) from last time step: alpha_{t-1}(s - 2)
        let log_alpha_s_minus_2 =
            self.right_shift_2d_tensor(log_alpha_t_minus_1.clone(), 2, &device);

        // Compute alpha_{t}(s) using recursion, corresponding to equation 6 of the paper.
        let log_alpha_bar = self.log_sum_exp(log_alpha_s, log_alpha_s_minus_1, &device);
        let log_alpha_bar_plus_log_alpha_s_minus_2 =
            self.log_sum_exp(log_alpha_bar.clone(), log_alpha_s_minus_2, &device);
        let log_alpha_s_to_s_minus_2 = log_alpha_bar.mask_where(
            l_prime_combined_mask.clone(),
            log_alpha_bar_plus_log_alpha_s_minus_2,
        ); // [N, 2 * U + 1]
        let log_probs_t = log_probs.clone().slice(s![t, .., ..]).squeeze_dim::<2>(0); // [N, C]
        let log_probs_l_prime_s = log_probs_t.gather(1, blank_inserted_targets.clone());
        let temp_log_alpha_t_s = log_alpha_s_to_s_minus_2.add(log_probs_l_prime_s);
        log_alpha_t_s.mask_where(combined_s_t_mask, temp_log_alpha_t_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn assert_approx_equal(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "Mismatch at index {}: expected {:.6}, got {:.6} (diff: {:.6})",
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    // ---------------------------------------------------------------
    // insert_blanks tests
    // ---------------------------------------------------------------

    #[test]
    fn test_insert_blanks_single_sample() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64, 2, 3]], &device);
        let result = ctc.insert_blanks::<TestBackend>(&targets, 1, 3, &device);
        let result_data = result.into_data().to_vec::<i64>().unwrap();
        assert_eq!(result_data, vec![0, 1, 0, 2, 0, 3, 0]);
    }

    #[test]
    fn test_insert_blanks_batch() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64, 2], [3, 4]], &device);
        let result = ctc.insert_blanks::<TestBackend>(&targets, 2, 2, &device);
        let result_data = result.into_data().to_vec::<i64>().unwrap();
        assert_eq!(result_data, vec![0, 1, 0, 2, 0, 0, 3, 0, 4, 0]);
    }

    #[test]
    fn test_insert_blanks_custom_blank() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_blank(2).init();

        let targets = Tensor::<TestBackend, 2, Int>::from_data([[0_i64, 1]], &device);
        let result = ctc.insert_blanks::<TestBackend>(&targets, 1, 2, &device);
        let result_data = result.into_data().to_vec::<i64>().unwrap();
        // l' = [blank=2, 0, blank=2, 1, blank=2]
        assert_eq!(result_data, vec![2, 0, 2, 1, 2]);
    }

    // ---------------------------------------------------------------
    // Assertions
    // ---------------------------------------------------------------

    #[test]
    #[should_panic(expected = "blank index")]
    fn test_ctc_loss_panics_invalid_blank_index() {
        let device = NdArrayDevice::Cpu;
        // blank=5 is out of bounds for num_classes=3
        let ctc = CTCLossConfig::new().with_blank(5).init();

        let log_probs = Tensor::<TestBackend, 3>::zeros([2, 1, 3], &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1], &device);

        ctc.forward(log_probs, targets, input_lengths, target_lengths);
    }

    #[test]
    #[should_panic(expected = "must equal batch_size")]
    fn test_ctc_loss_panics_mismatched_batch_size() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        // Logits batch size = 2
        let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
        // Targets batch size = 1 (Mismatch)
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2, 2], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1, 1], &device);

        ctc.forward(log_probs, targets, input_lengths, target_lengths);
    }

    #[test]
    #[should_panic(expected = "input_lengths length")]
    fn test_ctc_loss_panics_input_lengths_mismatch() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        // Logits batch size = 2
        let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1], [2]], &device);

        // Input lengths size = 1 (Mismatch)
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1, 1], &device);

        ctc.forward(log_probs, targets, input_lengths, target_lengths);
    }

    #[test]
    #[should_panic(expected = "target_lengths length")]
    fn test_ctc_loss_panics_target_lengths_mismatch() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        // Logits batch size = 2
        let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1], [2]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2, 2], &device);

        // Target lengths size = 1 (Mismatch)
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1], &device);

        ctc.forward(log_probs, targets, input_lengths, target_lengths);
    }

    // ---------------------------------------------------------------
    // Edge Case & Config Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ctc_loss_repeated_labels_minimum_input_length() {
        // T=3, N=1, C=2, blank=0, target=[1, 1], uniform P = 1/2.
        //
        // The minimum T for target [1, 1] is 3: the only valid path is (1, 0, 1).
        // prob = (1/2)^3 = 1/8
        // Loss = -ln(1/8) = 3 * ln(2)
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        let log_probs = Tensor::<TestBackend, 3>::full([3, 1, 2], 0.5_f32.ln(), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64, 1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([3_i64], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.into_data().to_vec::<f32>().unwrap();
        let expected = 3.0 * 2.0_f32.ln();
        assert_approx_equal(&loss_data, &[expected], 1e-3);
    }

    #[test]
    fn test_ctc_loss_custom_blank_uniform() {
        // T=3, N=1, C=3, blank=2, target=[0, 1], uniform P = 1/3.
        //
        // Two distinct labels, 3 classes, 3 time steps, just with
        // blank=2 instead of 0.
        // 5 valid paths → total = 5/27
        // Loss = -ln(5/27)
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_blank(2).init();

        let log_probs = Tensor::<TestBackend, 3>::full([3, 1, 3], (1.0_f32 / 3.0).ln(), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[0_i64, 1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([3_i64], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.into_data().to_vec::<f32>().unwrap();
        let expected = -(5.0_f32 / 27.0).ln();
        assert_approx_equal(&loss_data, &[expected], 1e-3);
    }

    // ---------------------------------------------------------------
    // zero_infinity tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ctc_loss_zero_infinity_produces_inf_when_disabled() {
        // T=2, N=1, C=3, blank=0, target=[1, 1], input_length=2
        // Target [1, 1] requires at least 3 time steps → no valid paths → loss = +inf
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_zero_infinity(false).init();

        let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 3], (1.0_f32 / 3.0).ln(), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64, 1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.into_data().to_vec::<f32>().unwrap();
        assert!(
            loss_data[0].is_infinite() && loss_data[0] > 0.0,
            "Expected +inf, got {}",
            loss_data[0]
        );
    }

    #[test]
    fn test_ctc_loss_zero_infinity_masks_inf_when_enabled() {
        // Same inputs as above, but zero_infinity=true → loss should be 0.0
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_zero_infinity(true).init();

        let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 3], (1.0_f32 / 3.0).ln(), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64, 1]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.into_data().to_vec::<f32>().unwrap();
        assert_approx_equal(&loss_data, &[0.0], 1e-6);
    }

    #[test]
    fn test_ctc_loss_zero_infinity_does_not_affect_finite_loss() {
        // Verify that zero_infinity=true does not change a finite loss value.
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_zero_infinity(true).init();

        let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 2], 0.5_f32.ln(), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i64]], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i64], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1_i64], &device);

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.into_data().to_vec::<f32>().unwrap();
        let expected = -(0.75_f32).ln();
        assert_approx_equal(&loss_data, &[expected], 1e-3);
    }
}

#[cfg(test)]
mod pytorch_comparison_tests {
    use super::*;
    use burn::tensor::activation::log_softmax;
    use burn_autodiff::Autodiff;
    use burn_core::tensor::TensorData;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type InnerBackend = NdArray<f32>;
    type TestBackend = Autodiff<InnerBackend>;

    fn assert_approx_equal(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "Mismatch at index {}: expected {:.6}, got {:.6} (diff: {:.6})",
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    /// Deterministic logits: sin((t*7 + n*13 + c*3) * 0.1).
    fn generate_logits(
        t_size: usize,
        n_size: usize,
        c_size: usize,
        device: &NdArrayDevice,
    ) -> Tensor<TestBackend, 3> {
        let mut data = Vec::with_capacity(t_size * n_size * c_size);
        for t in 0..t_size {
            for n in 0..n_size {
                for c in 0..c_size {
                    data.push(((t * 7 + n * 13 + c * 3) as f32 * 0.1).sin());
                }
            }
        }
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [t_size, n_size, c_size]), device)
    }

    /// Runs a CTC forward + backward test and asserts against expected values from PyTorch.
    ///
    /// This helper performs the following steps:
    /// 1. Generates deterministic logits using a sine-wave formula.
    /// 2. Computes the CTC loss (forward pass).
    /// 3. Asserts the computed loss matches `expected_losses`.
    /// 4. Backpropagates the sum of the loss.
    /// 5. Asserts the resulting gradients w.r.t. logits match `expected_grad_flat`.
    ///
    /// # Arguments
    ///
    /// - `expected_losses`: per-sample loss values from PyTorch (reduction='none').
    /// - `expected_grad_flat`: flattened gradient of sum(loss) w.r.t. logits.
    fn run_comparison(
        label: &str,
        t_size: usize,
        n_size: usize,
        c_size: usize,
        targets_flat: Vec<i64>,
        target_shape: [usize; 2],
        input_lengths: Vec<i64>,
        target_lengths: Vec<i64>,
        blank: usize,
        expected_losses: &[f32],
        expected_grad_flat: &[f32],
        loss_tol: f32,
        grad_tol: f32,
    ) {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().with_blank(blank).init();

        let logits = generate_logits(t_size, n_size, c_size, &device).require_grad();
        let log_probs = log_softmax(logits.clone(), 2);

        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(targets_flat, target_shape),
            &device,
        );
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(input_lengths, [n_size]),
            &device,
        );
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(target_lengths, [n_size]),
            &device,
        );

        let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
        let loss_data = loss.clone().into_data().to_vec::<f32>().unwrap();

        println!("=== {} ===", label);
        println!("  Loss: {:?}", loss_data);
        assert_approx_equal(&loss_data, expected_losses, loss_tol);

        let loss_sum = loss.sum();
        let grads = loss_sum.backward();
        let logits_grad = logits.grad(&grads).unwrap();
        let grad_data = logits_grad.into_data().to_vec::<f32>().unwrap();
        assert_approx_equal(&grad_data, expected_grad_flat, grad_tol);
    }

    #[test]
    fn test_ctc_loss_uniform_input_lengths() {
        // T=5, N=3, C=4, all input_lengths = 5
        // Expected losses and gradient from PyTorch
        let expected_losses = [3.5236570835113525_f32, 3.495313882827759, 4.262677192687988];
        let expected_grad_flat = [
            -0.1679008007_f32,
            -0.4595540464,
            0.2795598209,
            0.3478950262,
            -0.3913056254,
            -0.0832268298,
            0.2535884976,
            0.2209439576,
            -0.0502742566,
            0.2766197622,
            0.2054125518,
            -0.4317580462,
            -0.0544800088,
            -0.3144550920,
            0.0847885981,
            0.2841464877,
            -0.1844545156,
            -0.2063435912,
            0.2222184092,
            0.1685796976,
            0.0278018005,
            0.2657383382,
            -0.0336986706,
            -0.2598414719,
            -0.0482986756,
            -0.0098767160,
            -0.1533526182,
            0.2115280181,
            -0.1380317956,
            -0.2198686600,
            0.2042596638,
            0.1536407918,
            0.0534787849,
            0.1819230020,
            -0.2805589139,
            0.0451571345,
            -0.0895631388,
            0.1996460557,
            -0.2741115987,
            0.1640286744,
            -0.2200077325,
            -0.1693530381,
            0.2101601064,
            0.1792006642,
            0.0398471877,
            -0.1131042913,
            -0.2363226712,
            0.3095797896,
            -0.2163617164,
            0.2740726173,
            -0.2124865055,
            0.1547756046,
            -0.4312027395,
            -0.0446923785,
            0.2330704331,
            0.2428246588,
            -0.0050083841,
            -0.6256869435,
            0.2689785957,
            0.3617166877,
        ];
        run_comparison(
            "T=5, N=3, C=4 (uniform input lengths)",
            5,
            3,
            4,
            vec![1, 2, 0, 1, 0, 0, 3, 2, 1],
            [3, 3],
            vec![5, 5, 5],
            vec![2, 1, 3],
            0,
            &expected_losses,
            &expected_grad_flat,
            1e-3,
            1e-3,
        );
    }

    #[test]
    fn test_ctc_loss_repeated_labels() {
        // T=8, N=4, C=6, includes consecutive repeated label [1,1,2]
        // Expected losses and gradient from PyTorch
        let expected_losses = [
            8.84203052520752_f32,
            9.023029327392578,
            9.398024559020996,
            9.008068084716797,
        ];
        let expected_grad_flat = [
            -0.2766432464,
            -0.5202965736,
            0.1523768753,
            0.1896236390,
            0.2200277001,
            0.2349116206,
            -0.1854365915,
            0.2031330466,
            -0.4260218740,
            0.1678018719,
            0.1360142529,
            0.1045092493,
            -0.6603536606,
            0.2278252542,
            0.1691786796,
            0.1262856424,
            0.0972681716,
            0.0397959016,
            -0.0894432291,
            -0.5457318425,
            0.1490373611,
            0.1462858170,
            0.1569476575,
            0.1829041988,
            -0.2842915654,
            -0.4220107496,
            0.1822281033,
            0.1889107376,
            0.1791101843,
            0.1560532600,
            -0.1155678406,
            0.2295538932,
            -0.2645366490,
            -0.0288553704,
            0.1027252972,
            0.0766806602,
            -0.5448347330,
            0.2031028718,
            0.1589304954,
            0.1322451383,
            0.1189499870,
            -0.0683937520,
            -0.0873993114,
            -0.3051757514,
            -0.2355299890,
            0.1586059481,
            0.2018169016,
            0.2676822543,
            -0.3225219846,
            -0.2611543834,
            0.1922984123,
            0.1632783115,
            0.1297036558,
            0.0983960181,
            -0.1507159024,
            0.2256962359,
            -0.1040333956,
            -0.1514528394,
            0.0985243544,
            0.0819815546,
            -0.2940836251,
            0.1586865336,
            0.1468491107,
            0.1485087872,
            0.1639631987,
            -0.3239239752,
            -0.0767390430,
            -0.0434846729,
            -0.4023587406,
            -0.0052628326,
            0.2273432612,
            0.3005020022,
            -0.2598774135,
            -0.2188862711,
            0.1678501070,
            0.1352078766,
            0.1002781317,
            0.0754275694,
            -0.1502914876,
            0.1930875033,
            -0.0709601715,
            -0.2219523191,
            0.1243555173,
            0.1257609427,
            -0.0574148744,
            0.1152269915,
            0.1307857931,
            0.1599020809,
            0.2068412602,
            -0.5553412437,
            -0.0536844917,
            0.0758557543,
            -0.2106334567,
            -0.2509877980,
            0.1757438034,
            0.2637061775,
            -0.1759711355,
            -0.2431350052,
            0.1071053818,
            0.1259848624,
            0.1004033238,
            0.0856125653,
            -0.1173698306,
            0.1213828772,
            -0.1768893301,
            -0.2070008069,
            0.1709136516,
            0.2089634240,
            0.0153109450,
            0.0967332721,
            0.1268781722,
            0.1706230640,
            0.2291058898,
            -0.6386513710,
            -0.0536664203,
            0.1378114969,
            0.0360041447,
            -0.2989685237,
            -0.0084722806,
            0.1872915775,
            -0.1523490399,
            -0.2111770809,
            -0.0390694551,
            0.1366800815,
            0.1302325875,
            0.1356829405,
            -0.0982905105,
            -0.0127884001,
            -0.3586881459,
            -0.0259541404,
            0.2114149332,
            0.2843062580,
            -0.0324133746,
            0.1084750593,
            0.1447229236,
            0.1862253845,
            0.2259712219,
            -0.6329812407,
            -0.1173689738,
            0.1914442331,
            0.1654772907,
            -0.1376858056,
            -0.2194855511,
            0.1176188141,
            -0.1529908478,
            -0.0606661662,
            -0.3384291232,
            0.1524862647,
            0.1777049750,
            0.2218948901,
            -0.0923086405,
            -0.2855934799,
            -0.3215619624,
            0.1726681292,
            0.2303666323,
            0.2964293361,
            -0.2508065701,
            0.1479703039,
            0.1753441393,
            0.1917535067,
            0.1919818372,
            -0.4562432170,
            -0.2350299209,
            0.2257601619,
            0.1863904297,
            0.0388212129,
            -0.2966264784,
            0.0806845874,
            -0.1992894858,
            0.1068909168,
            -0.5761897564,
            0.1624972969,
            0.2155302167,
            0.2905607820,
            -0.1168124676,
            -0.6870660186,
            0.1488010883,
            0.1881926507,
            0.2230074406,
            0.2438773215,
            -0.5771554708,
            0.1980127096,
            0.1924194694,
            0.1714663208,
            0.1415647417,
            -0.1263078004,
            -0.3408652246,
            0.2292248607,
            0.1707807332,
            0.1269564927,
            -0.2634142637,
            0.0773174241,
        ];
        run_comparison(
            "T=8, N=4, C=6 (repeated labels)",
            8,
            4,
            6,
            vec![1, 1, 2, 0, 2, 3, 2, 1, 5, 0, 0, 0, 1, 2, 3, 4],
            [4, 4],
            vec![8, 8, 8, 8],
            vec![3, 4, 1, 4],
            0,
            &expected_losses,
            &expected_grad_flat,
            1e-3,
            1e-3,
        );
    }

    #[test]
    fn test_ctc_loss_long_sequence() {
        // T=10, N=2, C=8
        // Expected losses and gradient from PyTorch
        let expected_losses = [12.629399299621582, 12.298524856567383];
        let expected_grad_flat = [
            -0.2570972741,
            -0.6013792753,
            0.1061997041,
            0.1321590245,
            0.1533492655,
            0.1637226790,
            0.1598964781,
            0.1431493312,
            -0.2540431321,
            0.1788398325,
            -0.4038805366,
            0.1477340311,
            0.1197479516,
            0.0920107216,
            0.0686140805,
            0.0509770736,
            -0.1364373565,
            -0.3724762201,
            0.1489177048,
            -0.0966964588,
            0.1463697106,
            0.1275274903,
            0.1033692732,
            0.0794258416,
            -0.1771971881,
            0.2073454857,
            -0.3109439015,
            0.1249521226,
            -0.0101635465,
            0.0692621097,
            0.0533472970,
            0.0433975980,
            -0.1398337185,
            -0.0874802172,
            0.1705365479,
            -0.2174201906,
            0.1150254831,
            0.0460043959,
            0.0647982135,
            0.0483694859,
            -0.2332949787,
            0.1969220787,
            -0.1270586401,
            0.1098557115,
            -0.1364655048,
            0.0715296715,
            0.0553609394,
            0.0631506816,
            -0.2169117928,
            0.0929956511,
            0.1624538749,
            -0.2009791434,
            0.0904926360,
            -0.0248185843,
            0.0532633252,
            0.0435040221,
            -0.2313277274,
            0.1497355998,
            -0.0024202778,
            0.1029939279,
            -0.2776987851,
            0.0963881761,
            0.0351882279,
            0.1271408647,
            -0.2590557337,
            0.1577988416,
            0.1429322213,
            -0.1401246637,
            0.0866033062,
            -0.1151762009,
            0.0683368817,
            0.0586853735,
            -0.1322475076,
            0.0806737095,
            0.0528722852,
            0.0920089707,
            -0.3037962914,
            0.1280544847,
            -0.1391123086,
            0.2215466499,
            -0.1918463260,
            0.1376975775,
            0.1160097718,
            -0.0549413785,
            0.0970225409,
            -0.2708687484,
            0.1147320047,
            0.0521945432,
            -0.0504456684,
            -0.0012221609,
            0.0644332916,
            0.0818370953,
            -0.1036835983,
            0.1512031406,
            -0.4072600305,
            0.2651379406,
            -0.0681083873,
            0.0860663429,
            0.0810486302,
            0.0434282124,
            0.1056238264,
            -0.2994530201,
            0.1729898751,
            -0.1215954795,
            -0.0481944978,
            -0.1697723418,
            0.0725984722,
            0.0692019314,
            0.0859903544,
            0.1680216491,
            -0.4071443677,
            0.2292988002,
            -0.0205532499,
            0.0566616580,
            0.0326749459,
            0.0861379728,
            0.1142501161,
            -0.0448331088,
            0.2054910213,
            -0.4298293889,
            -0.0647637174,
            -0.4240962267,
            0.1013666242,
            -0.0110451467,
            0.1519176364,
            0.1661346704,
            -0.0719586164,
            0.1524447650,
            -0.0496110357,
            0.0562372655,
            -0.1889088154,
            0.1013496071,
            0.1339637935,
            0.1694275290,
            0.2007708699,
            -0.4232292175,
            -0.0401752405,
            -0.2951072752,
            0.1443216652,
            -0.2857291698,
            0.1489982456,
            0.1327733696,
            0.1096193567,
            0.0852990299,
            -0.0413062274,
            0.0820900649,
            -0.7903561592,
            0.1329460591,
            0.1535883099,
            0.1631743014,
            0.1585651338,
            0.1412984729,
            -0.1033771932,
            0.1799504310,
            0.1697744429,
            -0.5749052763,
            0.1189445183,
            0.0911802500,
            0.0679325759,
            0.0505003072,
        ];
        run_comparison(
            "T=10, N=2, C=8",
            10,
            2,
            8,
            vec![1, 3, 5, 7, 2, 2, 4, 6, 1, 3],
            [2, 5],
            vec![10, 10],
            vec![5, 5],
            0,
            &expected_losses,
            &expected_grad_flat,
            1e-3,
            1e-3,
        );
    }

    #[test]
    fn test_ctc_loss_mixed_input_lengths() {
        // T=12, N=3, C=5, input_lengths=[12, 7, 10]
        // Expected losses and gradient from PyTorch
        let expected_losses = [10.595505714416504, 6.8078508377075195, 7.705057144165039];
        let expected_grad_flat = [
            -0.4790987670,
            -0.2554937005,
            0.1991624236,
            0.2478453964,
            0.2875846624,
            -0.3495813310,
            0.2268397957,
            0.2150714993,
            -0.2442178279,
            0.1518878639,
            -0.2764556706,
            0.2474014312,
            -0.2137086987,
            0.1371368915,
            0.1056260392,
            -0.2729502618,
            -0.3609606028,
            0.2159237266,
            0.2238420397,
            0.1941450834,
            -0.2953839302,
            0.1920599341,
            0.1974952668,
            -0.2054278404,
            0.1112565696,
            -0.1719199270,
            0.2299505472,
            -0.2864859998,
            0.1497263014,
            0.0787290633,
            -0.2035763413,
            -0.3042884767,
            0.2126964629,
            0.1810975969,
            0.1140707731,
            -0.2759391963,
            0.0975771844,
            0.1823379993,
            -0.1112988219,
            0.1073228419,
            -0.1336459517,
            0.1869296581,
            -0.1996247321,
            0.1846873760,
            -0.0383463502,
            -0.2254105806,
            -0.1834360659,
            0.1925925612,
            0.1462381780,
            0.0700158924,
            -0.2259973884,
            -0.0393539183,
            0.1802661419,
            -0.0571591072,
            0.1422442794,
            -0.0609069727,
            0.1089282706,
            -0.0313654318,
            0.2186669111,
            -0.2353227735,
            -0.2840364873,
            -0.0632198900,
            0.1755636632,
            0.1377806067,
            0.0339120962,
            -0.1904856712,
            -0.2139032930,
            0.1827126741,
            0.0056131603,
            0.2160631120,
            -0.0243270602,
            -0.0070458520,
            0.1070247591,
            0.2239368409,
            -0.2995886803,
            -0.2955487072,
            0.0309870224,
            0.1654911339,
            0.1581364125,
            -0.0590658709,
            -0.2191396207,
            -0.3791662455,
            0.1803640425,
            0.1225430891,
            0.2953987718,
            -0.0436352938,
            -0.1575258970,
            0.1785279512,
            0.1756918877,
            -0.1530586481,
            -0.1834939867,
            0.0909025446,
            0.1423641294,
            0.1959712654,
            -0.2457439601,
            -0.3619639874,
            -0.3929221630,
            0.1820438206,
            0.2454170734,
            0.3274252713,
            -0.0628800318,
            -0.2567180395,
            0.2112283260,
            0.0507859327,
            0.0575838275,
            -0.0587697029,
            0.1174769849,
            0.0783569664,
            0.2290501744,
            -0.3661144078,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0725664943,
            -0.1532069892,
            0.2162397504,
            -0.1248963475,
            0.1344300956,
            -0.0362483934,
            0.1295878887,
            -0.0502482466,
            0.2470482886,
            -0.2901395261,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.1349253207,
            0.0867646411,
            0.1998746395,
            -0.2658679783,
            0.1141540110,
            -0.0705668628,
            0.1519546807,
            -0.2509805560,
            0.2475892603,
            -0.0779965296,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.2338010073,
            0.2471641302,
            0.1834627241,
            -0.3026831448,
            0.1058573127,
            -0.1155209392,
            0.1921830922,
            -0.4129956067,
            0.2229512781,
            0.1133821756,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.2636392713,
            0.2323469073,
            -0.2913427949,
            0.1800564528,
            0.1425786912,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
        ];
        run_comparison(
            "T=12, N=3, C=5 (mixed input lengths)",
            12,
            3,
            5,
            vec![1, 4, 2, 0, 3, 1, 0, 0, 2, 4, 1, 3],
            [3, 4],
            vec![12, 7, 10],
            vec![3, 2, 4],
            0,
            &expected_losses,
            &expected_grad_flat,
            1e-3,
            1e-3,
        );
    }

    #[test]
    fn test_ctc_loss_sum_reduction() {
        // Same inputs as comparison_uniform_input_lengths, sum reduction
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        let logits = generate_logits(5, 3, 4, &device).require_grad();
        let log_probs = log_softmax(logits.clone(), 2);
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 0, 1, 0, 0, 3, 2, 1], [3, 3]),
            &device,
        );
        let il = Tensor::<TestBackend, 1, Int>::from_data([5_i64, 5, 5], &device);
        let tl = Tensor::<TestBackend, 1, Int>::from_data([2_i64, 1, 3], &device);

        let loss = ctc.forward_with_reduction(log_probs, targets, il, tl, Reduction::Sum);
        let loss_data = loss.clone().into_data().to_vec::<f32>().unwrap();

        let expected_sum = 11.2816486359_f32; // Expected value from PyTorch
        assert_approx_equal(&loss_data, &[expected_sum], 1e-3);

        let grads = loss.backward();
        let logits_grad = logits.grad(&grads).unwrap();
        let grad_data = logits_grad.into_data().to_vec::<f32>().unwrap();
        // Expected gradient from PyTorch
        let expected_grad = [
            -0.1679008007_f32,
            -0.4595540464,
            0.2795598209,
            0.3478950262,
            -0.3913056254,
            -0.0832268298,
            0.2535884976,
            0.2209439576,
            -0.0502742566,
            0.2766197622,
            0.2054125518,
            -0.4317580462,
            -0.0544800088,
            -0.3144550920,
            0.0847885981,
            0.2841464877,
            -0.1844545156,
            -0.2063435912,
            0.2222184092,
            0.1685796976,
            0.0278018005,
            0.2657383382,
            -0.0336986706,
            -0.2598414719,
            -0.0482986756,
            -0.0098767160,
            -0.1533526182,
            0.2115280181,
            -0.1380317956,
            -0.2198686600,
            0.2042596638,
            0.1536407918,
            0.0534787849,
            0.1819230020,
            -0.2805589139,
            0.0451571345,
            -0.0895631388,
            0.1996460557,
            -0.2741115987,
            0.1640286744,
            -0.2200077325,
            -0.1693530381,
            0.2101601064,
            0.1792006642,
            0.0398471877,
            -0.1131042913,
            -0.2363226712,
            0.3095797896,
            -0.2163617164,
            0.2740726173,
            -0.2124865055,
            0.1547756046,
            -0.4312027395,
            -0.0446923785,
            0.2330704331,
            0.2428246588,
            -0.0050083841,
            -0.6256869435,
            0.2689785957,
            0.3617166877,
        ];
        assert_approx_equal(&grad_data, &expected_grad, 1e-3);
    }

    #[test]
    fn test_ctc_loss_mean_reduction() {
        let device = NdArrayDevice::Cpu;
        let ctc = CTCLossConfig::new().init();

        let logits = generate_logits(5, 3, 4, &device).require_grad();
        let log_probs = log_softmax(logits.clone(), 2);
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 0, 1, 0, 0, 3, 2, 1], [3, 3]),
            &device,
        );
        let il = Tensor::<TestBackend, 1, Int>::from_data([5_i64, 5, 5], &device);
        let tl = Tensor::<TestBackend, 1, Int>::from_data([2_i64, 1, 3], &device);

        let loss = ctc.forward_with_reduction(log_probs, targets, il, tl, Reduction::Mean);
        let loss_data = loss.clone().into_data().to_vec::<f32>().unwrap();

        let expected_mean = 2.2260115147_f32; // Expected value from PyTorch
        assert_approx_equal(&loss_data, &[expected_mean], 1e-3);

        let grads = loss.backward();
        let logits_grad = logits.grad(&grads).unwrap();
        let grad_data = logits_grad.into_data().to_vec::<f32>().unwrap();
        // Expected gradient from PyTorch
        let expected_grad = [
            -0.0279834662_f32,
            -0.0765923411,
            0.0465933047,
            0.0579825081,
            -0.1304352134,
            -0.0277422778,
            0.0845294967,
            0.0736479908,
            -0.0055860290,
            0.0307355281,
            0.0228236169,
            -0.0479731150,
            -0.0090800021,
            -0.0524091832,
            0.0141314333,
            0.0473577492,
            -0.0614848398,
            -0.0687812045,
            0.0740728080,
            0.0561932363,
            0.0030890885,
            0.0295264814,
            -0.0037442972,
            -0.0288712755,
            -0.0080497796,
            -0.0016461194,
            -0.0255587716,
            0.0352546684,
            -0.0460105985,
            -0.0732895583,
            0.0680865571,
            0.0512135960,
            0.0059420872,
            0.0202136654,
            -0.0311732125,
            0.0050174589,
            -0.0149271907,
            0.0332743451,
            -0.0456852652,
            0.0273381118,
            -0.0733359158,
            -0.0564510152,
            0.0700533763,
            0.0597335547,
            0.0044274656,
            -0.0125671430,
            -0.0262580756,
            0.0343977548,
            -0.0360602848,
            0.0456787720,
            -0.0354144201,
            0.0257959347,
            -0.1437342465,
            -0.0148974592,
            0.0776901469,
            0.0809415579,
            -0.0005564869,
            -0.0695207715,
            0.0298865121,
            0.0401907414,
        ];
        assert_approx_equal(&grad_data, &expected_grad, 1e-3);
    }
}
