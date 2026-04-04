use burn_backend::{Backend, Slice, tensor::{Bool, IndexingUpdateOp, Int}};
use crate::Tensor;

/// 
pub fn lu<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>, use_block_lu: bool
) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {
    let device = tensor.device();
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    
    let (lu_tensor, p_compact) = compute_lu_decomposition::<B, D, D1>(tensor, use_block_lu);
    
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

/// Checks:
/// - Input must have at least 2 dimensions
/// - Singularity: zero pivots
/// 
/// # Generic Parameters
/// - `D` is the input dimension
/// - `D1` must be set to D - 1
pub fn lu_factor<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>, use_block_lu: bool
) -> (Tensor<B, D>, Tensor<B, D1>) {
    let (lu, p) = compute_lu_decomposition::<B, D, D1>(tensor, use_block_lu);
    (lu, p.squeeze_dim(D - 1))
}

fn compute_lu_decomposition<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>, use_block_lu: bool
) -> (Tensor<B, D>, Tensor<B, D>) {
    let device = tensor.device();
    if !use_block_lu {
        return standard_lu_with_partial_piv::<B, D, D1>(tensor, &device)
    }
    
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    let size = n_rows.min(n_cols);
    if size < 256 {
        return standard_lu_with_partial_piv::<B, D, D1>(tensor, &device)
    }
        
    block_lu_with_partial_piv::<B, D, D1>(tensor)
}

/// Performs block LU decomposition with partial pivoting.
fn block_lu_with_partial_piv<B: Backend, const D: usize, const D1: usize>(
    mut tensor: Tensor<B, D>
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
    let n_blocks = (piv_nums + block_size - 1) / block_size;
    let mut slices = vec![Slice::full(); D];  // For updating the original tensor in-place
    
    // k is the current block number
    for block_k in 0..n_blocks {
        // Determine the index range for the current block column
        let k_start = block_k * block_size;
        let k_end = (k_start + block_size).min(piv_nums);
        let current_block_size = k_end - k_start;
        
        // Apply LU decomposition with partial pivoting to the first block column
        let sub_tensor = tensor
            .clone()
            .slice_dim(D - 2, k_start..)
            .slice_dim(D - 1, k_start..k_end);
        let (block_column, local_piv) = standard_lu_with_partial_piv::<B, D, D1>(
            sub_tensor,
            &device);
        // Update `tensor` with `block_column`
        slices[D - 2] = Slice::from(k_start..);
        slices[D - 1] = Slice::from(k_start..k_end);
        tensor = tensor.slice_assign(&slices, block_column);
        
        // Update `permutations` to global indices
        global_piv = update_permutations_to_global_idx(global_piv.clone(), local_piv.clone(), k_start);
        
        // Apply `permutations` to the remaining left sub-tensor
        if block_k != 0 {
            let left_sub_tensor = tensor.clone().slice_dim(D - 2, k_start..).slice_dim(D - 1, ..k_start);
            let permutated_left_sub_tensor = apply_permutations_to_tensor(left_sub_tensor, local_piv.clone(), &device);
            slices[D - 2] = Slice::from(k_start..);
            slices[D - 1] = Slice::from(..k_start);
            tensor = tensor.slice_assign(&slices, permutated_left_sub_tensor);
        }
        
        // Only update the right side if there are columns left
        if k_end < n_cols {
            // Apply `permutations` to the remaining right sub-tensor
            let right_sub_tensor = tensor.clone().slice_dim(D - 2, k_start..).slice_dim(D - 1, k_end..);
            let permutated_right_sub_tensor = apply_permutations_to_tensor(right_sub_tensor, local_piv, &device);
            slices[D - 2] = Slice::from(k_start..);
            slices[D - 1] = Slice::from(k_end..);
            tensor = tensor.slice_assign(&slices, permutated_right_sub_tensor);
            
            // Update the cols to the right of the current diagonal block
            let diagonal_l_block = tensor
                .clone()
                .slice_dim(D - 2, k_start..k_end)
                .slice_dim(D - 1, k_start..k_end);
            let row_blocks = tensor.clone().slice_dim(D - 2, k_start..k_end).slice_dim(D - 1, k_end..);
            let updated_row_blocks = solve_for_u_blocks(diagonal_l_block, row_blocks, current_block_size);
            
            slices[D - 2] = Slice::from(k_start..k_end);
            slices[D - 1] = Slice::from(k_end..);
            tensor = tensor.slice_assign(&slices, updated_row_blocks.clone());
            
            // Only update trailing A blocks if there are rows left below
            if k_end < n_rows {
                // Update the trailing A blocks
                let trailing_a_blocks = tensor.clone().slice_dim(D - 2, k_end..).slice_dim(D - 1, k_end..);
                let l_col_blocks = tensor.clone().slice_dim(D - 2, k_end..).slice_dim(D - 1, k_start..k_end);
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
fn standard_lu_with_partial_piv<B: Backend, const D: usize, const D1: usize>(
    mut tensor: Tensor<B, D>, device: &B::Device
) -> (Tensor<B, D>, Tensor<B, D>) {
    let dims = tensor.dims();
    let n_rows = dims[D - 2];
    let n_cols = dims[D - 1];
    let piv_nums = if n_rows == n_cols { n_rows } else { n_rows.min(n_cols) };
    let mut permutations = create_permutation_tensor::<B, D>(piv_nums, dims, &device);
    
    for k in 0..piv_nums {
        // Shape: [B1, ..., BN, 1, 1]
        let max_row_indices = tensor
            .clone()
            .slice_dim(D - 2, k..)
            .slice_dim(D - 1, k)
            .abs()
            .argmax(D - 2) + (k as i64);
        
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
                tensor = update_trailing_submatirx::<B, D, D1>(tensor, k);
            }
        }
    }
    
    (tensor, permutations)
}

fn construct_full_permutation_tensor<B: Backend, const D: usize>(piv: Tensor<B, D>, n_rows: usize, device: &B::Device) -> Tensor<B, D> {
    let dims = piv.dims();
    let identity_2d = Tensor::eye(n_rows, device);
    
    // Reshape the `identity` tensor from 2 dims to D dims
    let mut reshape_dims = [1; D];
    reshape_dims[D - 2] = n_rows;
    reshape_dims[D - 1] = n_rows;
    let reshaped_identity = identity_2d.reshape(reshape_dims);
    
    // Expand the batch dimensions to match the original input tensor's shape
    let mut expand_dims = [n_rows; D];
    for i in 0..(D - 2) {
        expand_dims[i] = dims[i];
    }
    let identity = reshaped_identity.expand(expand_dims);
    
    // Iterate through `piv` and apply rows swap to the `identity` tensor 
    // to construct the full permutation tensor
    let permutation_tensor = apply_permutations_to_tensor(identity, piv, device);
    permutation_tensor
}

fn create_permutation_tensor<B: Backend, const D: usize>(
    piv_nums: usize, dims: [usize; D], device: &B::Device
) -> Tensor<B, D> {
    let piv = Tensor::arange(0..piv_nums as i64, device).float();
    
    // Reshape the piv tensor from 1 dim to D dims
    let mut reshape_dims = [1; D];
    reshape_dims[D - 2] = piv_nums;
    let reshaped = piv.reshape(reshape_dims);
    
    // Expand the batch dimensions to match the original input tensor's shape
    let mut expand_dims = [piv_nums; D];
    for i in 0..(D - 2) {
        expand_dims[i] = dims[i];
    }
    expand_dims[D - 1] = 1;
    
    reshaped.expand(expand_dims)
}

/// Swaps k-th rows with the rows in `swap_target_row_tensor`
fn swap_tensor_rows<B: Backend, const D: usize>(
    tensor: Tensor<B, D>, mut swap_target_row_tensor: Tensor<B, D, Int>, k: usize, device: &B::Device
) -> Tensor<B, D> {
    let mut expand_dims = tensor.dims();
    expand_dims[D - 2] = 1;
    swap_target_row_tensor= swap_target_row_tensor.expand(expand_dims);
    
    let k_index_tensor = Tensor::<B, D, Int>::full(
        swap_target_row_tensor.shape(),
        k as i32,
        device
    );
    
    let val_k = tensor.clone().gather(D - 2, k_index_tensor.clone());
    let val_r = tensor.clone().gather(D - 2, swap_target_row_tensor.clone());
    let val_k_minus_r = val_k.clone() - val_r.clone();
    let val_r_minus_k = val_r - val_k;
    
    let tensor = tensor.scatter(D - 2, k_index_tensor, val_r_minus_k, IndexingUpdateOp::Add);
    let tensor = tensor.scatter(D - 2, swap_target_row_tensor, val_k_minus_r, IndexingUpdateOp::Add);
    
    tensor
}

fn update_permutations<B: Backend, const D: usize>(
    mut permutations: Tensor<B, D>, max_row_index_tensor: Tensor<B, D, Int>, k: usize
) -> Tensor<B, D> {
    // TODO: Update to not swap when k-th row is the max row
    
    // Store the max row index in the k-th index of the permutations vector/tensor
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from(k);
    let float_max_row_indices = max_row_index_tensor.float();
    permutations = permutations.slice_assign(&slices, float_max_row_indices);
    
    permutations
}

fn update_kth_column<B: Backend, const D: usize>(tensor: Tensor<B, D>, k: usize) -> Tensor<B, D> {
    let a_kk = tensor.clone().slice_dim(D - 2, k).slice_dim(D - 1, k);
    let a_rho_k = tensor.clone().slice_dim(D - 2, k+1..).slice_dim(D - 1, k);
    // TODO: Skip scaling and update steps
    let updated_column = a_rho_k / a_kk;
    
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from((k + 1)..);  // Rows k+1 to the end
    slices[D - 1] = Slice::from(k..(k + 1));  // Column k
    
    tensor.slice_assign(&slices, updated_column)
}

fn update_trailing_submatirx<B: Backend, const D: usize, const D1: usize>(
    tensor: Tensor<B, D>, k: usize
) -> Tensor<B, D> {
    let a_rho_k = tensor.clone().slice_dim(D - 2, k+1..).slice_dim(D - 1, k);
    let a_k_rho = tensor.clone().slice_dim(D - 2, k).slice_dim(D - 1, k+1..);
    let outer_product = a_rho_k.matmul(a_k_rho);
    
    let a_rho_rho = tensor.clone().slice_dim(D - 2, k+1..).slice_dim(D - 1, k+1..);
    let updated_a_rho_rho = a_rho_rho - outer_product;
    
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from((k + 1)..);  // Rows k+1 to the end
    slices[D - 1] = Slice::from((k + 1)..);  // Cols k+1 to the end
    tensor.slice_assign(&slices, updated_a_rho_rho)
}

fn update_permutations_to_global_idx<B: Backend, const D: usize>(
    global_piv: Tensor<B, D>, local_piv: Tensor<B, D>, k_start: usize
) -> Tensor<B, D> {
    let n = local_piv.dims()[D - 2];
    let mut slices = vec![Slice::full(); D];
    slices[D - 2] = Slice::from(k_start..(n + k_start));
    
    let global_val = local_piv.add_scalar(k_start as f32);
    
    global_piv.slice_assign(&slices, global_val)
}

/// Applies the permutations to the entire width of the tensor.
/// This updates the past L blocks and also permutes the unfactor A blocks.
fn apply_permutations_to_tensor<B: Backend, const D: usize>(
    mut tensor: Tensor<B, D>, global_piv: Tensor<B, D>, device: &B::Device
) -> Tensor<B, D> {
    // println!("tensor> {}", tensor);
    for (i, swap_target_row_tensor) in global_piv.iter_dim(D - 2).enumerate() {
        // swap i-th rows with the `swap_target_row_tensor` rows
        tensor = swap_tensor_rows(tensor, swap_target_row_tensor.int(), i, device);
    }
    
    tensor
}

/// # Arguments
/// - `diagonal_l_block`: The L block L_{k, k}, [k_start..k_end, k_start..k_end]
/// - `row_blocks`: The row blocks A_{k, k+1}, ..., A_{k, N}, [k_start..k_end, k_end..]
fn solve_for_u_blocks<B: Backend, const D: usize>(
    diagonal_l_block: Tensor<B, D>, mut a_row_blocks: Tensor<B, D>, block_size: usize
) -> Tensor<B, D> {
    // The first row requires no computation since the first row of 
    // diagonal_l_block is [1, 0, 0, ..., 0]
    let mut slices = vec![Slice::full(); D];
    
    for i in 1..block_size {
        // Shape of each matrix: 1 by i
        let l_multipliers = diagonal_l_block.clone().slice_dim(D - 2, i).slice_dim(D - 1, 0..i);
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


