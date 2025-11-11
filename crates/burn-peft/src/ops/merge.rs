use burn_tensor::{Tensor, backend::Backend};

/// Merge LoRA adapters into base weight: W' = W0 + A^T @ B^T
///
/// This is used for inference optimization where we want to fold the low-rank
/// update directly into the base weight for faster forward passes.
///
/// # Arguments
///
/// * `base` - Base weight W0 of shape [k, d] (Burn Linear convention: [d_in, d_out])
/// * `lora_b` - LoRA B matrix of shape [d, r] where d=d_out
/// * `lora_a` - LoRA A matrix of shape [r, k] where k=d_in
/// * `alpha_over_r` - Scaling factor α/r
///
/// # Returns
///
/// Merged weight W' = W0 + (α/r) * A^T @ B^T of shape [k, d]
pub fn merge_lora<B: Backend>(
    base: &Tensor<B, 2>,
    lora_b: &Tensor<B, 2>,
    lora_a: &Tensor<B, 2>,
    alpha_over_r: f32,
) -> Tensor<B, 2> {
    // Compute rank-r update: A^T @ B^T
    // A: [r, k] -> A^T: [k, r]
    // B: [d, r] -> B^T: [r, d]
    // A^T @ B^T = [k, r] @ [r, d] = [k, d]
    let delta = lora_a
        .clone()
        .transpose()
        .matmul(lora_b.clone().transpose());

    // Scale and add to base: W0 + (α/r) * (A^T @ B^T)
    base.clone() + delta * alpha_over_r
}

/// Unmerge LoRA adapters from merged weight: W0 = W' - A^T @ B^T
///
/// This is the inverse operation of merge_lora, useful when you want to
/// extract the original base weight from a merged version.
///
/// # Arguments
///
/// * `merged` - Merged weight W' of shape [k, d]
/// * `lora_b` - LoRA B matrix of shape [d, r]
/// * `lora_a` - LoRA A matrix of shape [r, k]
/// * `alpha_over_r` - Scaling factor α/r
///
/// # Returns
///
/// Base weight W0 = W' - (α/r) * A^T @ B^T of shape [k, d]
pub fn unmerge_lora<B: Backend>(
    merged: &Tensor<B, 2>,
    lora_b: &Tensor<B, 2>,
    lora_a: &Tensor<B, 2>,
    alpha_over_r: f32,
) -> Tensor<B, 2> {
    // Compute rank-r update: A^T @ B^T
    let delta = lora_a
        .clone()
        .transpose()
        .matmul(lora_b.clone().transpose());

    // Subtract from merged: W' - (α/r) * (A^T @ B^T)
    merged.clone() - delta * alpha_over_r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_merge_unmerge_roundtrip() {
        let device = Default::default();

        // Create base weight [k, d] = [d_in, d_out] = [2, 3]
        let base = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        // Create LoRA matrices with rank 1
        // lora_a: [r, k] = [1, 2]
        let lora_a = Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.1, 0.2]]), &device);
        // lora_b: [d, r] = [3, 1]
        let lora_b =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0], [2.0], [3.0]]), &device);

        let alpha_over_r = 1.0;

        // Merge
        let merged = merge_lora(&base, &lora_b, &lora_a, alpha_over_r);

        // Unmerge
        let restored = unmerge_lora(&merged, &lora_b, &lora_a, alpha_over_r);

        // Should recover original base weight
        let base_data = base.into_data().to_vec::<f32>().unwrap();
        let restored_data = restored.into_data().to_vec::<f32>().unwrap();

        for (b, r) in base_data.iter().zip(restored_data.iter()) {
            assert!((b - r).abs() < 1e-5, "Expected {}, got {}", b, r);
        }
    }
}
