/// Macro to generate cumulative operation functions for Candle backend
///
/// This macro reduces code duplication for cumulative operations (cumprod, cummin, cummax)
/// which all follow the same pattern of slicing, applying an operation, and concatenating.
macro_rules! cumulative_op {
    ($tensor:expr, $dim:expr, $init_slice:expr, $op:expr) => {{
        let dim_size = $tensor.dims()[$dim];
        let mut slices = Vec::with_capacity(dim_size);

        // First slice is the initial value
        slices.push($init_slice);

        // Apply cumulative operation
        for i in 1..dim_size {
            let curr = $tensor.narrow($dim, i, 1).unwrap();
            let result = $op(&slices[i - 1], &curr);
            slices.push(result);
        }

        candle_core::Tensor::cat(&slices, $dim).unwrap()
    }};
}

pub(crate) use cumulative_op;
