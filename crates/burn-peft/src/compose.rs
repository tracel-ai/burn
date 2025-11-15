use burn::tensor::{Tensor, backend::Backend};
use burn_core as burn;

/// Compose (merge) multiple LoRA adapters into a single adapter.
///
/// This is useful for multi-task learning where you train separate LoRA adapters
/// for different tasks and want to combine them for inference.
///
/// # Methods
///
/// ## Weighted Sum
/// ```text
/// A_merged = Σᵢ wᵢ·Aᵢ
/// B_merged = Σᵢ wᵢ·Bᵢ
/// ```
///
/// ## Concatenation (rank-wise)
/// ```text
/// A_merged = [A₁; A₂; ...; Aₙ]  (stack along rank dimension)
/// B_merged = [B₁; B₂; ...; Bₙ]
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // Train adapters for different tasks
/// let lora_task1 = train_lora_for_task1();
/// let lora_task2 = train_lora_for_task2();
///
/// // Compose with equal weights
/// let merged = AdapterComposer::new()
///     .add_adapter(&lora_task1, 0.5)
///     .add_adapter(&lora_task2, 0.5)
///     .compose();
/// ```
pub struct AdapterComposer;

impl AdapterComposer {
    /// Merge multiple LoRA adapters using weighted sum.
    ///
    /// # Arguments
    ///
    /// * `adapters_a` - List of A matrices (each [r, d_input])
    /// * `adapters_b` - List of B matrices (each [d_output, r])
    /// * `weights` - Weight for each adapter (should sum to ~1.0)
    ///
    /// # Returns
    ///
    /// Tuple of (merged_a, merged_b)
    pub fn weighted_sum<B: Backend>(
        adapters_a: &[Tensor<B, 2>],
        adapters_b: &[Tensor<B, 2>],
        weights: &[f32],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        assert_eq!(
            adapters_a.len(),
            adapters_b.len(),
            "Number of A and B adapters must match"
        );
        assert_eq!(
            adapters_a.len(),
            weights.len(),
            "Number of adapters and weights must match"
        );
        assert!(!adapters_a.is_empty(), "At least one adapter required");

        // Weighted sum of A matrices: A_merged = Σ wᵢ·Aᵢ
        let mut a_merged = adapters_a[0].clone() * weights[0];
        for (adapter, &weight) in adapters_a[1..].iter().zip(&weights[1..]) {
            a_merged = a_merged + adapter.clone() * weight;
        }

        // Weighted sum of B matrices: B_merged = Σ wᵢ·Bᵢ
        let mut b_merged = adapters_b[0].clone() * weights[0];
        for (adapter, &weight) in adapters_b[1..].iter().zip(&weights[1..]) {
            b_merged = b_merged + adapter.clone() * weight;
        }

        (a_merged, b_merged)
    }

    /// Concatenate multiple LoRA adapters to increase effective rank.
    ///
    /// This stacks adapters along the rank dimension, creating a higher-rank adapter.
    ///
    /// # Arguments
    ///
    /// * `adapters_a` - List of A matrices (each [rᵢ, d_input])
    /// * `adapters_b` - List of B matrices (each [d_output, rᵢ])
    ///
    /// # Returns
    ///
    /// Tuple of (concatenated_a, concatenated_b) with effective rank = Σrᵢ
    ///
    /// # Example
    ///
    /// If you have 3 adapters each with rank 8, concatenation gives rank 24.
    pub fn concatenate<B: Backend>(
        adapters_a: &[Tensor<B, 2>],
        adapters_b: &[Tensor<B, 2>],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        assert_eq!(
            adapters_a.len(),
            adapters_b.len(),
            "Number of A and B adapters must match"
        );
        assert!(!adapters_a.is_empty(), "At least one adapter required");

        // Concatenate A matrices along dimension 0 (rank dimension)
        // [r1, d] + [r2, d] + ... -> [r1+r2+..., d]
        let a_merged = Tensor::cat(adapters_a.to_vec(), 0);

        // Concatenate B matrices along dimension 1 (rank dimension)
        // [d, r1] + [d, r2] + ... -> [d, r1+r2+...]
        let b_merged = Tensor::cat(adapters_b.to_vec(), 1);

        (a_merged, b_merged)
    }

    // TODO: SVD-based adapter merging
    //
    // Once cubecl supports SVD operations, we can add an SVD-based merging method:
    //
    // pub fn svd_merge<B: Backend>(
    //     adapters_a: &[Tensor<B, 2>],
    //     adapters_b: &[Tensor<B, 2>],
    //     target_rank: usize,
    // ) -> (Tensor<B, 2>, Tensor<B, 2>) {
    //     // Compute full ΔW = Σᵢ (Aᵢᵀ @ Bᵢᵀ)
    //     // Decompose via SVD: ΔW = U·Σ·Vᵀ
    //     // Truncate to target_rank: U_r, Σ_r, V_r
    //     // Return (V_r·√Σ_r, U_r·√Σ_r)
    // }
    //
    // This would enable optimal rank reduction when merging multiple adapters.
    // Reference: "LoRA Composition" techniques from PEFT literature.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Distribution;

    #[test]
    fn test_weighted_sum_composition() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create two adapters
        let a1 = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let b1 = Tensor::<TestBackend, 2>::random([64, 8], Distribution::Default, &device);

        let a2 = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let b2 = Tensor::<TestBackend, 2>::random([64, 8], Distribution::Default, &device);

        // Merge with equal weights
        let (a_merged, b_merged) = AdapterComposer::weighted_sum(
            &[a1.clone(), a2.clone()],
            &[b1.clone(), b2.clone()],
            &[0.5, 0.5],
        );

        // Check shapes
        assert_eq!(a_merged.dims(), [8, 32]);
        assert_eq!(b_merged.dims(), [64, 8]);

        // Verify: merged should equal 0.5*adapter1 + 0.5*adapter2
        let expected_a = (a1 * 0.5) + (a2 * 0.5);
        let diff_a = (a_merged - expected_a).abs().mean();
        assert!(diff_a.into_scalar() < 1e-6);
    }

    #[test]
    fn test_concatenate_composition() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create two adapters with different ranks
        let a1 = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let b1 = Tensor::<TestBackend, 2>::random([64, 8], Distribution::Default, &device);

        let a2 = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let b2 = Tensor::<TestBackend, 2>::random([64, 4], Distribution::Default, &device);

        // Concatenate
        let (a_merged, b_merged) = AdapterComposer::concatenate(&[a1, a2], &[b1, b2]);

        // Check shapes: ranks should sum (8 + 4 = 12)
        assert_eq!(a_merged.dims(), [12, 32]);
        assert_eq!(b_merged.dims(), [64, 12]);
    }

    #[test]
    fn test_single_adapter_composition() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let a = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let b = Tensor::<TestBackend, 2>::random([64, 8], Distribution::Default, &device);

        // Single adapter with weight 1.0 should return itself
        let (a_merged, b_merged) = AdapterComposer::weighted_sum(
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            &[1.0],
        );

        let diff_a = (a_merged - a).abs().mean();
        let diff_b = (b_merged - b).abs().mean();

        assert!(diff_a.into_scalar() < 1e-6);
        assert!(diff_b.into_scalar() < 1e-6);
    }
}
