use crate::{ElementConversion, Tensor, check, check::TensorCheck};
use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{
    Backend, Slice,
    tensor::{Bool, IndexingUpdateOp, Int},
};

/// Computes the LU decomposition of a square or rectangular matrix with partial pivoting.
///
/// This function decomposes the input tensor A into three tensors P, L, and U
/// such that A = PLU.
///
/// # Arguments
/// - `tensor` - The input tensor of shape `[..., n_rows, n_cols]`.
///
/// # Returns
/// A tuple of three tensors `(P, L, U)`:
/// - `P` - The permutation tensor of shape `[..., n_rows, n_rows]`.
/// - `L` - The lower triangular tensor of shape `[..., n_rows, min(n_rows, n_cols)]`
///   with unit diagonal elements.
/// - `U` - The upper triangular tensor of shape `[..., min(n_rows, n_cols), n_cols]`.
///
/// # Generic Parameters
///
/// - `D`: The number of dimensions of the input tensor.
/// - `D1`: The number of dimensions of the 1D pivot tensor. Must be exactly `D - 1`.
///
/// # Panics
/// This function will panic if the tensor checks fail:
/// - The input tensor has less than 2 dimensions (`D < 2`).
/// - The generic parameters do not satisfy `D - 1 == D1`.
///
/// # Performance Note
/// The current implementation of LU decomposition is not fully optimized. It will not
/// be as fast as highly tuned specialized libraries, especially for very large
/// matrices or large batch sizes.
///
/// # Example
/// ```rust,ignore
/// use burn::tensor::Tensor;
/// use burn::backend::Flex;
/// use burn::tensor::linalg;
///
/// fn example<B: Backend>() {
///     let device = Default::default();
///     let tensor = Tensor::<B, 2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
///
///     // Compute P, L, U
///     let (p, l, u) = linalg::lu::<B, 2, 1>(tensor);
///
///     // Expected Output:
///     // p: [[0.0, 1.0],
///     //     [1.0, 0.0]]
///     //
///     // l: [[1.0,       0.0],
///     //     [0.6666667, 1.0]]
///     //
///     // u: [[6.0, 3.0],
///     //     [0.0, 1.0]]
/// }
/// ```
pub fn lu<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>,
) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {
    let dims = tensor.dims();

    check!(TensorCheck::lu_generic_param::<D, D1>("linalg::lu"));
    check!(TensorCheck::lu_input_tensor::<D>("linalg::lu", &dims));

    let device = tensor.device();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];

    let (lu_tensor, p_compact) = compute_lu_decomposition::<B, D, D1>(tensor);

    let u;
    let temp_l;
    if n_rows < n_cols {
        temp_l = lu_tensor.clone().slice_dim(D - 1, 0..n_rows).tril(0);
        u = lu_tensor.triu(0);
    } else {
        temp_l = lu_tensor.clone().tril(0);
        u = lu_tensor.slice_dim(D - 2, 0..n_cols).triu(0);
    }
    let mask = Tensor::<B, D, Bool>::diag_mask(temp_l.shape(), 0, &device).bool_not();
    let l = temp_l.mask_fill(mask, 1.0);
    let p = construct_full_permutation_tensor(p_compact, n_rows, &device).transpose();

    (p, l, u)
}

/// Dispatches the LU decomposition to either the block or standard algorithm based on
/// the size of the matrix.
fn compute_lu_decomposition<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let device = tensor.device();
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    let size = n_rows.min(n_cols);
    if size < 256 {
        return standard_lu_with_partial_piv::<B, D, D1>(tensor, &device);
    }

    block_lu_with_partial_piv::<B, D, D1>(tensor)
}

/// Performs block LU decomposition with partial pivoting.
///
/// This algorithm divides the matrix into blocks to maximize matrix-matrix multiplications (GEMM),
/// which are highly optimized on modern hardware, compared to vector-vector operations.
fn block_lu_with_partial_piv<B: Backend, const D: usize, const D1: usize>(
    mut tensor: Tensor<B, D>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let device = tensor.device();
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    let piv_nums = n_rows.min(n_cols);
    let mut global_piv = create_permutation_tensor::<B, D>(piv_nums, dims, &device);
    let block_size = 128;

    // Computes the total number of blocks including incomplete blocks
    // E.g., piv_nums = 100 & block_size = 32 -> n_blocks = 4 (not 3) where
    // the 4th block is smaller than other blocks
    let n_blocks = piv_nums.div_ceil(block_size);
    let mut slices = vec![Slice::full(); D]; // For updating the original tensor in-place

    // k is the current block number
    for block_k in 0..n_blocks {
        // Determine the index range for the current block column
        let k_start = block_k * block_size;
        let k_end = (k_start + block_size).min(piv_nums);
        let current_block_size = k_end - k_start;

        // Apply standard LU decomposition with partial pivoting to the current block column
        let sub_tensor = tensor
            .clone()
            .slice_dim(D - 2, k_start..)
            .slice_dim(D - 1, k_start..k_end);
        let (block_column, local_piv) =
            standard_lu_with_partial_piv::<B, D, D1>(sub_tensor, &device);
        slices[D - 2] = Slice::from(k_start..);
        slices[D - 1] = Slice::from(k_start..k_end);
        tensor = tensor.slice_assign(&slices, block_column);

        // Update `permutations` to global indices
        global_piv =
            update_permutations_to_global_idx(global_piv.clone(), local_piv.clone(), k_start);

        // Apply `local_piv` permutations to the left sub-tensor
        if block_k != 0 {
            let left_sub_tensor = tensor
                .clone()
                .slice_dim(D - 2, k_start..)
                .slice_dim(D - 1, ..k_start);
            let permutated_left_sub_tensor =
                apply_permutations_to_tensor(left_sub_tensor, local_piv.clone(), &device);
            slices[D - 2] = Slice::from(k_start..);
            slices[D - 1] = Slice::from(..k_start);
            tensor = tensor.slice_assign(&slices, permutated_left_sub_tensor);
        }

        // Only update the right side if there are columns left
        if k_end < n_cols {
            // Apply `local_piv` permutations to the remaining right sub-tensor
            let right_sub_tensor = tensor
                .clone()
                .slice_dim(D - 2, k_start..)
                .slice_dim(D - 1, k_end..);
            let permutated_right_sub_tensor =
                apply_permutations_to_tensor(right_sub_tensor, local_piv, &device);
            slices[D - 2] = Slice::from(k_start..);
            slices[D - 1] = Slice::from(k_end..);
            tensor = tensor.slice_assign(&slices, permutated_right_sub_tensor);

            // Update the cols to the right of the current diagonal block.
            // Triangular solve for U blocks.
            let diagonal_l_block = tensor
                .clone()
                .slice_dim(D - 2, k_start..k_end)
                .slice_dim(D - 1, k_start..k_end);
            let row_blocks = tensor
                .clone()
                .slice_dim(D - 2, k_start..k_end)
                .slice_dim(D - 1, k_end..);
            let updated_row_blocks =
                solve_for_u_blocks(diagonal_l_block, row_blocks, current_block_size);
            slices[D - 2] = Slice::from(k_start..k_end);
            slices[D - 1] = Slice::from(k_end..);
            tensor = tensor.slice_assign(&slices, updated_row_blocks.clone());

            // Only update trailing A blocks if there are rows left below
            if k_end < n_rows {
                // Update the trailing A blocks
                let trailing_a_blocks = tensor
                    .clone()
                    .slice_dim(D - 2, k_end..)
                    .slice_dim(D - 1, k_end..);
                let l_col_blocks = tensor
                    .clone()
                    .slice_dim(D - 2, k_end..)
                    .slice_dim(D - 1, k_start..k_end);
                let outer_prod = l_col_blocks.matmul(updated_row_blocks);
                let new_trailing_a_blocks = trailing_a_blocks - outer_prod;

                // Overwrite part of the tensor with new_trailing_a_blocks
                slices[D - 2] = Slice::from(k_end..);
                slices[D - 1] = Slice::from(k_end..);
                tensor = tensor.slice_assign(&slices, new_trailing_a_blocks);
            }
        }
    }

    (tensor, global_piv)
}

/// Performs standard LU decomposition (outer product LU) with partial pivoting.
///
/// This is an iterative, unblocked algorithm that processes the matrix column by column.
fn standard_lu_with_partial_piv<B: Backend, const D: usize, const D1: usize>(
    mut tensor: Tensor<B, D>,
    device: &B::Device,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    let piv_nums = n_rows.min(n_cols);
    let mut permutations = create_permutation_tensor::<B, D>(piv_nums, dims, device);

    for k in 0..piv_nums {
        // Find the index of the maximum absolute value in the k-th column (from row k downwards)
        // Shape: [B1, ..., BN, 1, 1]
        let max_row_indices = tensor
            .clone()
            .slice_dim(D - 2, k..)
            .slice_dim(D - 1, k)
            .abs()
            .argmax(D - 2)
            + (k as i64);

        // Swap current row (k-th row) with the row with maximum absolute value
        tensor = swap_tensor_rows(tensor, max_row_indices.clone(), k, device);
        // Store the max row index in the k-th entry of the permutations vector/tensor
        permutations = update_permutations(permutations, max_row_indices, k);

        // If there are rows left under the k-th pivot
        if k < n_rows - 1 {
            // Update k-th column under the diagonal
            tensor = update_kth_column(tensor, k);

            // If there still exists columns to right of the k-th pivot
            if k < piv_nums - 1 {
                tensor = update_trailing_submatrix::<B, D, D1>(tensor, k);
            }
        }
    }

    (tensor, permutations)
}

/// Constructs a full square permutation matrix \( P \) from a compact pivot tensor.
fn construct_full_permutation_tensor<B: Backend, const D: usize>(
    piv: Tensor<B, D>,
    n_rows: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    let dims = piv.dims();
    let identity_2d = Tensor::eye(n_rows, device);

    // Reshape the `identity` tensor from 2 dims to D dims
    let mut reshape_dims = [1; D];
    reshape_dims[D - 2] = n_rows;
    reshape_dims[D - 1] = n_rows;
    let reshaped_identity = identity_2d.reshape(reshape_dims);

    // Expand the batch dimensions to match the original input tensor's shape
    let mut expand_dims = [n_rows; D];
    expand_dims[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let identity = reshaped_identity.expand(expand_dims);

    // Iterate through `piv` and apply rows swap to the `identity` tensor
    // to construct the full permutation tensor

    apply_permutations_to_tensor(identity, piv, device)
}

/// Initializes a permutation tensor representing the identity permutation `[0, 1, 2, ..., piv_nums - 1]`.
fn create_permutation_tensor<B: Backend, const D: usize>(
    piv_nums: usize,
    dims: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    let piv = Tensor::arange(0..piv_nums as i64, device).float();

    // Reshape the piv tensor from 1 dim to D dims
    let mut reshape_dims = [1; D];
    reshape_dims[D - 2] = piv_nums;
    let reshaped = piv.reshape(reshape_dims);

    // Expand the batch dimensions to match the original input tensor's shape
    let mut expand_dims = [piv_nums; D];
    expand_dims[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    expand_dims[D - 1] = 1;

    reshaped.expand(expand_dims)
}

/// Swaps the `k`-th row with the rows specified in `swap_target_row_tensor`.
fn swap_tensor_rows<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mut swap_target_row_tensor: Tensor<B, D, Int>,
    k: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    let mut expand_dims = tensor.dims();
    expand_dims[D - 2] = 1;
    swap_target_row_tensor = swap_target_row_tensor.expand(expand_dims);

    let k_index_tensor =
        Tensor::<B, D, Int>::full(swap_target_row_tensor.shape(), k as i32, device);

    let val_k = tensor.clone().gather(D - 2, k_index_tensor.clone());
    let val_r = tensor.clone().gather(D - 2, swap_target_row_tensor.clone());
    let val_k_minus_r = val_k.clone() - val_r.clone();
    let val_r_minus_k = val_r - val_k;

    let tensor = tensor.scatter(D - 2, k_index_tensor, val_r_minus_k, IndexingUpdateOp::Add);

    tensor.scatter(
        D - 2,
        swap_target_row_tensor,
        val_k_minus_r,
        IndexingUpdateOp::Add,
    )
}

/// Updates the permutation tensor by recording the swap at step `k`.
fn update_permutations<B: Backend, const D: usize>(
    mut permutations: Tensor<B, D>,
    max_row_index_tensor: Tensor<B, D, Int>,
    k: usize,
) -> Tensor<B, D> {
    // Store the max row index in the k-th index of the permutations vector/tensor
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from(k);
    let float_max_row_indices = max_row_index_tensor.float();
    permutations = permutations.slice_assign(&slices, float_max_row_indices);

    permutations
}

/// Scales the `k`-th column below the diagonal by the pivot element A_{kk}.
fn update_kth_column<B: Backend, const D: usize>(tensor: Tensor<B, D>, k: usize) -> Tensor<B, D> {
    let a_kk = tensor.clone().slice_dim(D - 2, k).slice_dim(D - 1, k);
    let a_rho_k = tensor.clone().slice_dim(D - 2, k + 1..).slice_dim(D - 1, k);

    // A singular matrix will have a pivot of exactly 0.
    // Due to partial pivoting, if the pivot is 0, all elements below it are also 0.
    // We replace 0 with 1.0 to avoid NaN when dividing 0.0 / 0.0.
    let is_zero_mask = a_kk.clone().equal_elem(0.0);
    let safe_a_kk = a_kk.mask_fill(is_zero_mask, 1.0);
    let updated_column = a_rho_k / safe_a_kk;

    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from((k + 1)..); // Rows k+1 to the end
    slices[D - 1] = Slice::from(k..(k + 1)); // Column k

    tensor.slice_assign(&slices, updated_column)
}

/// Updates the trailing submatrix: A_{k+1:, k+1:} -= A_{k+1:, k} * A_{k, k+1:}.
fn update_trailing_submatrix<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>,
    k: usize,
) -> Tensor<B, D> {
    let a_rho_k = tensor.clone().slice_dim(D - 2, k + 1..).slice_dim(D - 1, k);
    let a_k_rho = tensor.clone().slice_dim(D - 2, k).slice_dim(D - 1, k + 1..);
    let outer_product = a_rho_k.matmul(a_k_rho);

    let a_rho_rho = tensor
        .clone()
        .slice_dim(D - 2, k + 1..)
        .slice_dim(D - 1, k + 1..);
    let updated_a_rho_rho = a_rho_rho - outer_product;

    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from((k + 1)..); // Rows k+1 to the end
    slices[D - 1] = Slice::from((k + 1)..); // Cols k+1 to the end
    tensor.slice_assign(&slices, updated_a_rho_rho)
}

/// Shifts local pivot indices from a block factorization to global indices.
fn update_permutations_to_global_idx<B: Backend, const D: usize>(
    global_piv: Tensor<B, D>,
    local_piv: Tensor<B, D>,
    k_start: usize,
) -> Tensor<B, D> {
    let n = local_piv.dims()[D - 2];
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from(k_start..(n + k_start));

    let global_val = local_piv.add_scalar(k_start as f32);

    global_piv.slice_assign(&slices, global_val)
}

/// Applies the permutations to the entire width of the tensor.
fn apply_permutations_to_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    piv: Tensor<B, D>,
    device: &B::Device,
) -> Tensor<B, D> {
    let tensor_dims = tensor.dims();
    let n_rows = tensor_dims[D - 2];
    let n_pivots = piv.dims()[D - 2];
    let piv_data: Vec<f32> = piv.into_data().convert::<f32>().into_vec::<f32>().unwrap();

    // Compute total batch size (product of all batch dimensions)
    let batch_size: usize = tensor_dims[..D - 2].iter().product();
    if batch_size <= 1 {
        // No batch dims (or batch size 1)
        let mut perm: Vec<i64> = (0..n_rows as i64).collect();
        for (i, piv_val) in piv_data.iter().enumerate().take(n_pivots) {
            let j = piv_val.elem::<u32>() as usize;
            perm.swap(i, j);
        }
        let perm_tensor = Tensor::<B, 1, Int>::from_data(&perm[..], device);
        return tensor.select(D - 2, perm_tensor);
    }

    // If input tensor has batch dimensions, then flatten batch dims,
    // iterate, then reshape back.
    // Reshape tensor: [b1, b2, ..., bN, rows, cols] -> [B, rows, cols]
    let n_cols = tensor_dims[D - 1];
    let flat_tensor: Tensor<B, 3> = tensor.reshape([batch_size, n_rows, n_cols]);
    // Reshape pivot: [b1, b2, ..., bN, n_pivots, 1] -> [B * n_pivots]
    let mut results: Vec<Tensor<B, 3>> = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        // Build permutation for this batch element
        let mut perm: Vec<i64> = (0..n_rows as i64).collect();
        let offset = b * n_pivots;
        for i in 0..n_pivots {
            let j = piv_data[offset + i].elem::<u32>() as usize;
            perm.swap(i, j);
        }
        let perm_tensor = Tensor::<B, 1, Int>::from_data(&perm[..], device);

        // Extract this batch element [1, rows, cols], select rows, collect
        let batch_elem = flat_tensor.clone().slice_dim(0, b); // [1, rows, cols]  
        let permuted = batch_elem.select(1, perm_tensor); // [1, rows, cols]  
        results.push(permuted);
    }

    // Concatenate along batch dim and reshape back to original shape
    let concatenated: Tensor<B, 3> = Tensor::cat(results, 0); // [B, rows, cols]  
    concatenated.reshape(tensor_dims)
}

/// Solves for the U blocks using forward substitution.
///
/// Solves the equation L_{kk} U_{k, k+1:} = A_{k, k+1:}  for  U_{k, k+1:}.
///
/// # Arguments
/// - `diagonal_l_block`: The L block L_{k, k}, [k_start..k_end, k_start..k_end]
/// - `row_blocks`: The row blocks A_{k, k+1}, ..., A_{k, N}, [k_start..k_end, k_end..]
/// - `block_size`: The size of the current block
fn solve_for_u_blocks<B: Backend, const D: usize>(
    diagonal_l_block: Tensor<B, D>,
    mut a_row_blocks: Tensor<B, D>,
    block_size: usize,
) -> Tensor<B, D> {
    // The first row requires no computation since the first row of
    // diagonal_l_block is [1, 0, 0, ..., 0]
    let mut slices = vec![Slice::full(); D];

    for i in 1..block_size {
        // Shape of each matrix: 1 by i
        let l_multipliers = diagonal_l_block
            .clone()
            .slice_dim(D - 2, i)
            .slice_dim(D - 1, 0..i);
        // Shape of each matrix: i by c where c is the number of cols in a_row_blocks
        let u_computed = a_row_blocks.clone().slice_dim(D - 2, 0..i);
        let prod = l_multipliers.matmul(u_computed);

        let current_rows = a_row_blocks.clone().slice_dim(D - 2, i);
        let updated_rows = current_rows - prod;

        // Update the i-th row of a_row_blocks with the solved values of U blocks
        slices[D - 2] = Slice::from(i);
        a_row_blocks = a_row_blocks.slice_assign(&slices, updated_rows);
    }

    a_row_blocks.clone()
}
