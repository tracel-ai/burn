use burn_tensor::{Tensor, backend::Backend, linalg::l2_norm};

/// Compute column-wise L2 norms with numerical stability.
///
/// For a matrix M ∈ ℝ^{d×k}, computes norms ∈ ℝ^{1×k} where:
/// norms[j] = sqrt(sum_i(M[i,j]^2) + ε)
///
/// # Numerical Stability
///
/// - Uses Burn's optimized L2 norm implementation
/// - Applies epsilon via clamp_min to prevent division by zero
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [d, k]
/// * `epsilon` - Small constant to clamp minimum norm value (default: 1e-8)
///
/// # Returns
///
/// Tensor of shape [1, k] containing column-wise norms
///
/// # Example
///
/// ```rust,ignore
/// let matrix = Tensor::random([128, 64], Distribution::Default, &device);
/// let norms = col_norm(&matrix, 1e-8);
/// assert_eq!(norms.dims(), [1, 64]);
/// ```
pub fn col_norm<B: Backend>(tensor: &Tensor<B, 2>, epsilon: f32) -> Tensor<B, 2> {
    // Compute L2 norm along dimension 0 (rows) to get column-wise norms
    // Result shape: [1, k]
    let norms = l2_norm(tensor.clone(), 0);

    // Clamp to minimum epsilon to prevent division by zero
    norms.clamp_min(epsilon)
}

/// Compute column-wise L2 norms and return detached version (stops gradient).
///
/// This is critical for DoRA/QDoRA where we need to prevent gradients from flowing
/// through the norm computation to save memory and stabilize training.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [d, k]
/// * `epsilon` - Small constant to add before sqrt (default: 1e-8)
///
/// # Returns
///
/// Detached tensor of shape [1, k] containing column-wise norms
pub fn col_norm_detached<B: Backend>(tensor: &Tensor<B, 2>, epsilon: f32) -> Tensor<B, 2> {
    col_norm(tensor, epsilon).detach()
}

/// Normalize columns to unit length.
///
/// For a matrix M ∈ ℝ^{d×k}, computes M_normalized where each column j is:
/// M_normalized[:, j] = M[:, j] / ||M[:, j]||_2
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [d, k]
/// * `epsilon` - Small constant to add before division (default: 1e-8)
///
/// # Returns
///
/// Tensor of shape [d, k] with unit-length columns
pub fn normalize_cols<B: Backend>(tensor: &Tensor<B, 2>, epsilon: f32) -> Tensor<B, 2> {
    let norms = col_norm(tensor, epsilon);
    // Broadcast division over rows
    tensor.clone() / norms
}

/// Normalize columns to unit length with detached norms (stops gradient through norm).
///
/// This is the key operation for DoRA: V' / detach(||V'||_c)
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [d, k]
/// * `epsilon` - Small constant to add before division (default: 1e-8)
///
/// # Returns
///
/// Tensor of shape [d, k] with unit-length columns (gradients don't flow through norms)
pub fn normalize_cols_detached<B: Backend>(tensor: &Tensor<B, 2>, epsilon: f32) -> Tensor<B, 2> {
    let norms = col_norm_detached(tensor, epsilon);
    // Broadcast division over rows
    // Gradients will flow through tensor but NOT through norms (detached)
    tensor.clone() / norms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_col_norm_simple() {
        let device = Default::default();

        // Create a simple 2x2 matrix:
        // [[3, 0],
        //  [4, 1]]
        // Column norms should be: [5, 1]
        let data = TensorData::from([[3.0, 0.0], [4.0, 1.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);

        let norms = col_norm(&tensor, 1e-8);
        let norms_data = norms.into_data();

        // Check shape is [1, 2]
        assert_eq!(norms_data.shape, vec![1, 2]);

        // Check values (sqrt(9+16) = 5, sqrt(0+1) = 1)
        let values = norms_data.to_vec::<f32>().unwrap();
        assert!((values[0] - 5.0).abs() < 1e-5);
        assert!((values[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_cols() {
        let device = Default::default();

        // Create a matrix with known column norms
        let data = TensorData::from([[3.0, 0.0], [4.0, 1.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);

        let normalized = normalize_cols(&tensor, 1e-8);

        // Verify columns have unit norm
        let norms = col_norm(&normalized, 1e-8);
        let norms_data = norms.into_data().to_vec::<f32>().unwrap();

        assert!((norms_data[0] - 1.0).abs() < 1e-5);
        assert!((norms_data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_col_norm_epsilon_prevents_nan() {
        let device = Default::default();

        // Zero tensor should not produce NaN with epsilon
        let tensor = Tensor::<TestBackend, 2>::zeros([10, 5], &device);
        let norms = col_norm(&tensor, 1e-8);
        let norms_data = norms.into_data().to_vec::<f32>().unwrap();

        // All norms should be clamped to epsilon (not NaN)
        // Note: clamp_min clamps to epsilon, not sqrt(epsilon)
        for &norm in &norms_data {
            assert!(!norm.is_nan());
            assert!((norm - 1e-8_f32).abs() < 1e-10);
        }
    }
}
