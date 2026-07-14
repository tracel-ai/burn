/// Helper function for cumulative operations in Candle backend
///
/// This function reduces code duplication for cumulative operations (cumprod, cummin, cummax)
/// which all follow the same pattern of slicing, applying an operation, and concatenating.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `dim` - The dimension along which to apply the cumulative operation
/// * `op` - A closure that takes two tensor references and produces a result tensor
pub fn cumulative_with_op<F>(tensor: &candle_core::Tensor, dim: usize, op: F) -> candle_core::Tensor
where
    F: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>,
{
    let dim_size = tensor.dims()[dim];
    let mut slices = Vec::with_capacity(dim_size);

    // First slice is the initial value
    slices.push(tensor.narrow(dim, 0, 1).unwrap());

    // Apply cumulative operation
    for i in 1..dim_size {
        let curr = tensor.narrow(dim, i, 1).unwrap();
        let result = op(&slices[i - 1], &curr).unwrap();
        slices.push(result);
    }

    candle_core::Tensor::cat(&slices, dim).unwrap()
}
