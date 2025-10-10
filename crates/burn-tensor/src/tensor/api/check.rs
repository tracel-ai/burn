use crate::ops::FloatElem;
use crate::{BasicOps, Numeric, Shape, Slice, Tensor, backend::Backend, cast::ToElement};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// The struct should always be used with the [check](crate::check) macro.
///
/// This is a simple pub(crate) data structure that efficiently checks tensor operations and
/// formats clear error messages. It's crucial that the checks are really fast, but it doesn't matter
/// when a failed check is discovered since the program will panic.
///
/// # Notes
///
/// Failing tensor checks will always result in a panic.
/// As mentioned in [The Rust Programming Language book](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html),
/// when there is no way to recover, panic should be used instead of a result.
///
/// Most users will unwrap the results anyway, which will worsen the clarity of the code. Almost
/// all checks highlight programming errors, which means invalid programs that should be fixed.
/// Checks are not the ideal way to help users write correct programs, but they are still better
/// than backend errors. Other forms of compile-time validation could be developed, such as named
/// tensors, but we have to carefully evaluate the ease of use of the Tensor API. Adding overly
/// complex type validation checks might drastically worsen the API and result in harder-to-maintain
/// programs.
///
/// # Design
///
/// Maybe the Backend API should return a result for each operation, which would allow handling
/// all checks, even the ones that can't be efficiently checked before performing an operation,
/// such as the `index_select` operation. The downside of that approach is that all backend
/// implementation might re-implement the same checks, which may result in unnecessary code
/// duplication. Maybe a combination of both strategies could help to cover all use cases.
pub(crate) enum TensorCheck {
    Ok,
    Failed(FailedTensorCheck),
}

impl TensorCheck {
    /// Checks device and shape compatibility for element wise binary operations.
    pub(crate) fn binary_ops_ew<B: Backend, const D: usize, K: BasicOps<B>>(
        ops: &str,
        lhs: &Tensor<B, D, K>,
        rhs: &Tensor<B, D, K>,
    ) -> Self {
        Self::Ok
            .binary_ops_device(ops, &lhs.device(), &rhs.device())
            .binary_ops_ew_shape::<D>(ops, &lhs.shape(), &rhs.shape())
    }

    pub(crate) fn into_scalar<const D: usize>(shape: &Shape) -> Self {
        let mut check = Self::Ok;

        if shape.num_elements() != 1 {
            check = check.register(
                "Into Scalar",
                TensorError::new("Only tensors with 1 element can be converted into scalar.")
                    .details(format!(
                        "Current tensor has {} elements",
                        shape.num_elements()
                    )),
            );
        }

        check
    }

    pub(crate) fn dim_ops<const D: usize>(ops: &str, dim: usize) -> Self {
        let mut check = Self::Ok;

        if dim >= D {
            check = check.register(
                ops,
                TensorError::new("Given dimension is higher than the tensor rank.")
                    .details(format!("Tensor rank: '{D}', given dimension: '{dim}'.")),
            );
        }

        check
    }

    pub(crate) fn creation_ops<const D: usize>(ops: &str, dims: &[usize]) -> Self {
        let mut check = Self::Ok;

        if D == 0 {
            check = check.register(
                ops,
                TensorError::new("Tried to create a 0-dim tensor, which is invalid.")
                    .details(format!("Tensor rank: '{D}', given dimensions: '{dims:?}'.")),
            );
        }

        if dims.len() != D {
            check = check.register(
                ops,
                TensorError::new("Given dimensions differ from the tensor rank.")
                    .details(format!("Tensor rank: '{D}', given dimensions: '{dims:?}'.")),
            );
        }

        check
    }

    pub(crate) fn narrow<B: Backend, const D: usize, K: BasicOps<B>>(
        tensor: &Tensor<B, D, K>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Self {
        let mut check = Self::Ok;

        if length == 0 {
            check = check.register(
                "Narrow",
                TensorError::new(format!(
                    "Can't narrow at dimension {dim}, length must be greater than 0",
                )),
            );
        }

        if start >= tensor.shape().dims[dim] {
            check = check.register(
                "Narrow",
                TensorError::new(format!(
                    "Can't narrow at dimension {dim}, start exceeds the size of the tensor along \
                     this dimension (Size={})",
                    tensor.shape().dims[dim]
                )),
            );
        }

        if start + length > tensor.shape().dims[dim] {
            check = check.register(
                "Narrow",
                TensorError::new(format!(
                    "Can't narrow at dimension {dim}, start + length exceeds the size of the tensor \
                     along this dimension (Size={})",
                    tensor.shape().dims[dim]
                )),
            );
        }

        check
    }

    pub(crate) fn reshape_args_usize<const D1: usize, const D2: usize>(
        original: &Shape,
        target: &Shape,
    ) -> Self {
        let mut check = Self::Ok;

        if original.num_elements() != target.num_elements() {
            check = check.register(
                "Reshape",
                TensorError::new(
                    "The given shape doesn't have the same number of elements as the current \
                     tensor.",
                )
                .details(format!(
                    "Current shape: {:?}, target shape: {:?}.",
                    original.dims, target.dims
                )),
            );
        }

        check
    }

    pub(crate) fn reshape_args_i64<const D: usize>(target: &[i64; D]) -> Self {
        let mut check = Self::Ok;

        if target.iter().any(|&dim| dim < -1) {
            check = check.register(
                "Reshape",
                TensorError::new(
                    "The given shape cannot contain negative dimensions (other than -1).",
                )
                .details(format!("Target shape: {target:?}.")),
            );
        }

        if target.iter().filter(|&x| x == &-1).count() > 1 {
            check = check.register(
                "Reshape",
                TensorError::new("The given shape cannot contain more than one -1.")
                    .details(format!("Target shape: {target:?}.")),
            );
        }

        check
    }

    pub(crate) fn movedim_args_usize<const D: usize>(dim: usize) -> Self {
        let mut check = Self::Ok;

        if dim >= D {
            check = check.register(
                "Movedim",
                TensorError::new(
                    "The given dimension exceeds the number of dimensions of the current tensor.",
                )
                .details(format!(
                    "Current tensor has {D} dimensions, but the given dimension is {dim}.",
                )),
            );
        }

        check
    }

    pub(crate) fn movedim_args_i32<const D: usize>(dim: i32) -> Self {
        let mut check = Self::Ok;

        if dim < -(D as i32) || dim >= D as i32 {
            check = check.register(
                "Movedim",
                TensorError::new(
                    "The given dimension is out of bounds for the current tensor dimensions.",
                )
                .details(format!(
                    "Current tensor has {D} dimensions, but the given dimension is {dim}.",
                )),
            );
        }

        check
    }

    pub(crate) fn movedim_args_vec<const D: usize>(dims: &Vec<usize>) -> Self {
        let mut check = Self::Ok;

        // Check out of bounds
        if dims.iter().any(|&x| x >= D) {
            check = check.register(
                "Movedim",
                TensorError::new("The given dimensions are out of bounds.").details(format!(
                    "Current tensor has {D} dimensions, but the given dimensions are {dims:?}.",
                )),
            );
        }

        // Check there are no duplicates
        for (i, &dim_i) in dims.iter().enumerate() {
            for &dim_j in dims.iter().skip(i + 1) {
                if dim_i == dim_j {
                    check = check.register(
                        "Movedim",
                        TensorError::new("The given dimensions contain duplicates.").details(
                            format!(
                            "The dimension {dim_i} is duplicated in the given dimensions {dims:?}.",
                        ),
                        ),
                    );
                }
            }
        }

        check
    }

    pub(crate) fn movedim_args_length(
        source_dims: &Vec<usize>,
        destination_dims: &Vec<usize>,
    ) -> Self {
        let mut check = Self::Ok;

        if source_dims.len() != destination_dims.len() {
            check = check.register(
                "Movedim",
                TensorError::new(
                    "The number of dimensions in source and destination must be equal.",
                )
                .details(format!(
                    "Source dimensions: {source_dims:?}, Destination dimensions: {destination_dims:?}.",
                )),
            )
        }

        check
    }

    pub(crate) fn flatten<const D1: usize, const D2: usize>(
        start_dim: usize,
        end_dim: usize,
    ) -> Self {
        let mut check = Self::Ok;

        if start_dim > end_dim {
            check = check.register(
                "Flatten",
                TensorError::new(format!(
                    "The start dim ({start_dim}) must be smaller than or equal to the end dim ({end_dim})"
                )),
            );
        }

        if D2 > D1 {
            check = check.register(
                "Flatten",
                TensorError::new(format!(
                    "Result dim ({D2}) must be smaller than or equal to ({D1})"
                )),
            );
        }

        if D1 < end_dim + 1 {
            check = check.register(
                "Flatten",
                TensorError::new(format!(
                    "The end dim ({end_dim}) must be smaller than the tensor dim ({D1})"
                )),
            );
        }

        if (D2 as i32) < (D1 as i32 - (end_dim as i32 - start_dim as i32)) {
            check = check.register(
                "Flatten",
                TensorError::new(format!(
                    "The destination dimension ({D2}) must be large enough to accommodate the \
                     flattening operation."
                )),
            );
        }

        check
    }

    pub(crate) fn tri<const D: usize>() -> Self {
        let mut check = Self::Ok;

        if D < 2 {
            check = check.register(
                "Tri",
                TensorError::new(format!(
                    "The input tensor must have at least 2 dimensions, got {D}"
                )),
            );
        }

        check
    }

    pub(crate) fn squeeze<const D2: usize>(dim: usize, tensor_dims: &[usize]) -> Self {
        let mut check = Self::Ok;
        // This should actually be to check that the dimension to squeeze
        // has a size of 1
        if tensor_dims[dim] != 1 {
            check = check.register(
                "Squeeze",
                TensorError::new(format!(
                    "Can't squeeze dimension {dim} because its size is not 1",
                )),
            );
        }

        if dim >= tensor_dims.len() {
            check = check.register(
                "Squeeze",
                TensorError::new(format!(
                    "Dimension index {dim} is out of bounds for tensor dimensions {tensor_dims:?}.",
                )),
            );
        }

        check
    }

    pub(crate) fn squeeze_dims_input<const D2: usize>(
        dim_indices: &[usize],
        current_dims: &[usize],
    ) -> Self {
        let mut check = Self::Ok;
        if dim_indices.len() >= current_dims.len() {
            check = check.register(
                "Squeeze",
                TensorError::new("Attempted to squeeze too many dimensions!"),
            );
        }

        check
    }

    pub(crate) fn squeeze_dims_len<const D2: usize>(new_dims_len: usize) -> Self {
        let mut check = Self::Ok;
        if new_dims_len != D2 {
            check = check.register(
                "Squeeze",
                TensorError::new(format!(
                    "Resulting dimensions {new_dims_len} do not match the required D2 size {D2}.",
                )),
            );
        }

        check
    }

    pub(crate) fn unsqueeze<const D1: usize, const D2: usize>() -> Self {
        let mut check = Self::Ok;
        if D2 < D1 {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "Can't unsqueeze smaller tensor, got dim {D2}, expected > {D1}",
                )),
            );
        }

        check
    }

    pub(crate) fn unsqueeze_dim<const D1: usize, const D2: usize>(dim: usize) -> Self {
        let mut check = Self::Ok;
        if D2 <= D1 {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "The unsqueezed rank must be greater than the input rank (D={D1}; D2={D2})",
                )),
            );
        }

        if dim > D1 {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "Can't unsqueeze at dimension {dim}, exceeds tensor dimensions (D={D1})",
                )),
            );
        }

        if dim >= D2 {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "Can't unsqueeze at dimension {dim}, exceeds output tensor dimensions (D2={D2})",
                )),
            );
        }

        check
    }

    pub(crate) fn unsqueeze_dims<const D: usize>(dim: isize) -> Self {
        let mut check = Self::Ok;
        let output_rank = D as isize;
        //contains is right exclusive, so this is to spec
        if !(-output_rank..output_rank).contains(&dim) {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "unsqueeze arg {dim} is out of range for the output tensor of rank {output_rank}",
                )),
            );
        }
        check
    }

    pub(crate) fn one_hot_tensor<B: Backend, const D: usize, K: Numeric<B>>(
        index_tensor: Tensor<B, D, K>,
        num_classes: usize,
    ) -> Self {
        let mut check = Self::Ok;
        if index_tensor
            .clone()
            .greater_equal_elem(num_classes as i32)
            .any()
            .into_scalar()
            .to_bool()
        {
            check = check.register(
                "One Hot",
                TensorError::new(format!(
                    "Can't create a one hot tensor from ({index_tensor:?}) containing indexes greater or equal to the number of classes ({num_classes})",
                )),
            );
        } else if num_classes <= 1 {
            check = check.register(
                "One Hot",
                TensorError::new("Can't create a one hot tensor with less then 2 classes"),
            )
        }
        check
    }

    pub(crate) fn one_hot_tensor_rank<const D: usize, const D2: usize>() -> Self {
        let mut check = Self::Ok;
        if D + 1 != D2 {
            check = check.register(
                "One Hot",
                TensorError::new(
                    "The one-hot tensor rank must correspond to the rank of the tensor + 1",
                )
                .details(format!("Expected D2={}, got {D2}", D + 1)),
            );
        }
        check
    }

    pub(crate) fn swap_dims<const D: usize>(dim1: usize, dim2: usize) -> Self {
        let mut check = Self::Ok;

        if dim1 > D || dim2 > D {
            check = check.register(
                "Swap Dims",
                TensorError::new("The swap dimensions must be smaller than the tensor dimension")
                    .details(format!(
                        "Swap dims ({dim1}, {dim2}) on tensor with ({D}) dimensions."
                    )),
            );
        }

        check
    }

    pub(crate) fn permute<const D: usize>(axes: [usize; D]) -> Self {
        let check = Self::Ok;

        // Check if the axes are within the tensor dimensions
        if let Some(axis) = axes.iter().find(|&x| *x >= D) {
            return check.register(
                "permute",
                TensorError::new("The axes must be smaller than the tensor dimension.")
                    .details(format!("The '{axis}' axis is greater than {D} dimensions.")),
            );
        }

        // Check if the axes are unique
        let mut seen = [false; D];
        axes.iter().for_each(|&x| seen[x] = true);
        if seen.iter().any(|&x| !x) {
            return check.register(
                "permute",
                TensorError::new("The axes must be unique.")
                    .details(format!("The axes '{axes:?}' are not unique.")),
            );
        }

        check
    }

    pub(crate) fn flip(rank: usize, axes: &[usize]) -> Self {
        let check = Self::Ok;

        // Check if the axes are within the tensor dimensions
        if let Some(axis) = axes.iter().find(|&x| *x >= rank) {
            return check.register(
                "flip",
                TensorError::new("The axes must be smaller than the tensor dimension.").details(
                    format!("The '{axis}' axis is greater than {rank} dimensions."),
                ),
            );
        }

        // Check if the axes are unique
        let mut dedup = axes.to_vec();
        dedup.sort_unstable();
        dedup.dedup();
        if dedup.len() != axes.len() {
            return check.register(
                "flip",
                TensorError::new("The axes must be unique.")
                    .details(format!("The axes '{axes:?}' are not unique.")),
            );
        }

        check
    }

    pub(crate) fn matmul<B: Backend, const D: usize, K>(
        lhs: &Tensor<B, D, K>,
        rhs: &Tensor<B, D, K>,
    ) -> Self
    where
        K: BasicOps<B>,
    {
        let mut check = Self::Ok;

        check = check.binary_ops_device("Matmul", &lhs.device(), &rhs.device());

        if D < 2 {
            return check;
        }

        let shape_lhs = lhs.shape();
        let shape_rhs = rhs.shape();

        let dim_lhs = shape_lhs.dims[D - 1];
        let dim_rhs = shape_rhs.dims[D - 2];

        if dim_lhs != dim_rhs {
            check = check.register(
                "Matmul",
                TensorError::new(format!(
                    "The inner dimension of matmul should be the same, but got {dim_lhs} and \
                     {dim_rhs}."
                ))
                .details(format!(
                    "Lhs shape {:?}, rhs shape {:?}.",
                    shape_lhs.dims, shape_rhs.dims
                )),
            );
        }

        check
    }

    pub(crate) fn cross<B: Backend, const D: usize, K>(
        lhs: &Tensor<B, D, K>,
        rhs: &Tensor<B, D, K>,
        dim: usize,
    ) -> Self
    where
        K: BasicOps<B>,
    {
        let mut check = Self::Ok;

        check = check.binary_ops_device("Cross", &lhs.device(), &rhs.device());

        let shape_lhs = lhs.shape();
        let shape_rhs = rhs.shape();

        if dim >= D {
            check = check.register(
                "Cross",
                TensorError::new(format!(
                    "Dimension {dim} is out of bounds for tensors with {D} dimensions."
                )),
            );
            return check;
        }

        let dim_size_lhs = shape_lhs.dims[dim];
        let dim_size_rhs = shape_rhs.dims[dim];

        if dim_size_lhs != 3 || dim_size_rhs != 3 {
            check = check.register(
                "Cross",
                TensorError::new(format!(
                    "Cross product requires dimension {dim} to have size 3, but got {dim_size_lhs} and {dim_size_rhs}."
                )),
            );
        }

        // Check broadcastability of other dimensions
        for i in 0..D {
            if i != dim {
                let l = shape_lhs.dims[i];
                let r = shape_rhs.dims[i];
                if l != r && l != 1 && r != 1 {
                    check = check.register(
                        "Cross",
                        TensorError::new(format!(
                            "Tensors are not broadcastable along dimension {i}: {l} and {r}."
                        )),
                    );
                }
            }
        }

        check
    }

    pub(crate) fn stack<B: Backend, const D1: usize, K: BasicOps<B>, const D2: usize>(
        tensors: &[Tensor<B, D1, K>],
        dim: usize,
    ) -> Self {
        let mut check = Self::Ok;

        if dim > D1 {
            check = check.register(
                "Stack",
                TensorError::new(
                    "Can't stack tensors on a dim that exceeds the tensors dimension (inclusive)",
                )
                .details(format!(
                    "Trying to concatenate tensors with {D1} dimensions on axis {dim}."
                )),
            );
        }

        if D1 == D2 {
            check = check.register(
                "Stack",
                TensorError::new(format!(
                    "Can't stack tensors on existing dimension {dim}, the input and output ranks are the same (D={D1}; D2={D2}).\
                    If you want to concatenate the tensors along the specified dimension ({dim}), use `Tensor::cat` instead.",
                )),
            );
        }

        if tensors.is_empty() {
            return check.register(
                "Stack",
                TensorError::new("Can't stack an empty list of tensors."),
            );
        }

        let shape_reference = tensors.first().unwrap().shape();

        for tensor in tensors {
            let shape = tensor.shape();

            if shape_reference != shape {
                return check.register(
                    "Stack",
                    TensorError::new("Can't stack tensors with different shapes").details(format!(
                        "Provided dimension ({dim}), tensors shapes: {:?}",
                        tensors.iter().map(Tensor::shape).collect::<Vec<_>>()
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn cat<B: Backend, const D: usize, K: BasicOps<B>>(
        tensors: &[Tensor<B, D, K>],
        dim: usize,
    ) -> Self {
        let mut check = Self::Ok;

        if dim >= D {
            check = check.register(
                "Cat",
                TensorError::new(
                    "Can't concatenate tensors on a dim that exceeds the tensors dimension",
                )
                .details(format!(
                    "Trying to concatenate tensors with {D} dimensions on axis {dim}."
                )),
            );
        }

        if tensors.is_empty() {
            return check.register(
                "Cat",
                TensorError::new("Can't concatenate an empty list of tensors."),
            );
        }

        let mut shape_reference = tensors.first().unwrap().shape();
        shape_reference.dims[dim] = 1; // We want to check every dims except the one where the
        // concatenation happens.

        for tensor in tensors {
            let mut shape = tensor.shape();
            shape[dim] = 1; // Ignore the concatenate dim.

            if shape_reference != shape {
                return check.register(
                    "Cat",
                    TensorError::new(
                        "Can't concatenate tensors with different shapes, except for the provided \
                         dimension",
                    )
                    .details(format!(
                        "Provided dimension ({dim}), tensors shapes: {:?}",
                        tensors.iter().map(Tensor::shape).collect::<Vec<_>>()
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn slice<const D1: usize, const D2: usize>(shape: &Shape, slices: &[Slice]) -> Self {
        let mut check = Self::Ok;
        let n_dims_tensor = D1;
        let n_dims_slices = slices.len();

        if n_dims_tensor < n_dims_slices {
            check = check.register(
                "Slice",
                TensorError::new(
                    "The provided slices array has a higher number of dimensions than the current \
                     tensor.",
                )
                .details(format!(
                    "The slices array must be smaller or equal to the tensor number of \
                     dimensions. Tensor number of dimensions: {n_dims_tensor}, slices array \
                     length {n_dims_slices}."
                )),
            );
        }

        for (i, slice) in slices.iter().enumerate().take(D1) {
            let d_tensor = shape[i];

            // Check the raw end value before conversion
            if let Some(end) = slice.end
                && end > 0
                && end as usize > d_tensor
            {
                check = check.register(
                        "Slice",
                        TensorError::new(
                            "The provided slice has a range that exceeds the current tensor \
                             size.",
                        )
                        .details(format!(
                            "The slice end index {} exceeds the size of the tensor ({}) at dimension {}. \
                             Tensor shape {:?}.",
                            end, d_tensor, i, shape.dims,
                        )),
                    );
            }

            let range = slice.to_range(d_tensor);

            if range.start >= range.end {
                check = check.register(
                    "Slice",
                    TensorError::new(
                        "The provided slice has a range where the start index is bigger or \
                         equal to its end.",
                    )
                    .details(format!(
                        "The range at dimension '{}' starts at '{}' and is greater or equal to \
                         its end '{}'. Tensor shape {:?}.",
                        i, range.start, range.end, shape.dims,
                    )),
                );
            }

            if slice.step() == 0 {
                check = check.register(
                    "Slice",
                    TensorError::new("The provided slice has a step of 0.").details(format!(
                        "The slice at dimension '{i}' has a step of 0. Step must be non-zero.",
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn slice_assign<const D1: usize, const D2: usize>(
        shape: &Shape,
        shape_value: &Shape,
        slices: &[crate::Slice],
    ) -> Self {
        let mut check = Self::Ok;

        if D1 < D2 {
            check = check.register(
                "Slice Assign",
                TensorError::new(
                    "The provided slices array has a higher number of dimensions than the current \
                     tensor.",
                )
                .details(format!(
                    "The slices array must be smaller or equal to the tensor number of \
                     dimensions. Tensor number of dimensions: {D1}, slices array length {D2}."
                )),
            );
        }

        for (i, slice) in slices.iter().enumerate().take(usize::min(D1, D2)) {
            let d_tensor = shape[i];
            let d_tensor_value = shape_value.dims[i];
            let range = slice.to_range(d_tensor);

            if range.end > d_tensor {
                check = check.register(
                    "Range Assign",
                    TensorError::new(
                        "The provided slice has a range that exceeds the current tensor \
                         size.",
                    )
                    .details(format!(
                        "The range ({}..{}) exceeds the size of the tensor ({}) at dimension {}. \
                         Current tensor shape {:?}, value tensor shape {:?}.",
                        range.start, range.end, d_tensor, i, shape.dims, shape_value.dims,
                    )),
                );
            }

            // Calculate the number of elements selected with the given step
            let num_elements = slice.output_size(d_tensor);

            if num_elements != d_tensor_value {
                check = check.register(
                    "Slice Assign",
                    TensorError::new(
                        "The value tensor must match the amount of elements selected with the \
                         slices array",
                    )
                    .details(format!(
                        "The slice with range ({}..{}) and step {} selects {} elements but the value \
                         tensor has {} elements at dimension {}. Current tensor shape {:?}, value tensor \
                         shape {:?}.",
                        range.start,
                        range.end,
                        slice.step,
                        num_elements,
                        d_tensor_value,
                        i,
                        shape.dims,
                        shape_value.dims,
                    )),
                );
            }

            if range.start >= range.end && slice.step > 0 {
                check = check.register(
                    "Slice Assign",
                    TensorError::new(
                        "The provided slice has a range where the start index is bigger or \
                         equal to its end with positive step.",
                    )
                    .details(format!(
                        "The range start ({}) must be smaller than its end ({}) for positive step ({}) at dimension {}",
                        range.start, range.end, slice.step, i
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn check_dim<const D: usize>(dim: usize) -> Self {
        let mut check = Self::Ok;

        if dim >= D {
            check = check.register(
                "Check Dim",
                TensorError::new("The provided dimension exceeds the tensor dimensions.").details(
                    format!("Tensor has {D} dimensions, but the provided dimension is {dim}."),
                ),
            );
        }

        check
    }

    pub(crate) fn gather<const D: usize>(dim: usize, shape: &Shape, shape_indices: &Shape) -> Self {
        Self::check_gather_scatter_indices::<D>(Self::Ok, "Gather", dim, shape, shape_indices)
    }

    pub(crate) fn scatter<const D: usize>(
        dim: usize,
        shape: &Shape,
        shape_indices: &Shape,
        shape_value: &Shape,
    ) -> Self {
        let ops = "Scatter";
        let mut check =
            Self::check_gather_scatter_indices::<D>(Self::Ok, ops, dim, shape, shape_indices);

        if shape_indices != shape_value {
            check = check.register(
                ops,
                TensorError::new(
                    "Indices tensor shape should be the same as the value tensor shape."
                        .to_string(),
                )
                .details(format!(
                    "The shape differs: {:?} != {:?}",
                    shape_indices.dims, shape_value.dims
                )),
            );
        }

        check
    }

    pub(crate) fn select<const D: usize>(dim: usize) -> Self {
        Self::check_select_basic::<D>(Self::Ok, "select", dim)
    }

    pub(crate) fn take<const D: usize, const DI: usize, const DO: usize>(dim: usize) -> Self {
        let mut check = Self::check_select_basic::<D>(Self::Ok, "Take", dim);

        // Calculate expected output dimensions
        // DO = D - 1 + DI (remove 1 dim, add DI dims)
        let expected_do = D + DI - 1;
        if DO != expected_do {
            check = check.register(
                "Take",
                TensorError::new("Output dimension mismatch").details(format!(
                    "Expected output dimension {} (D={} + DI={} - 1) but got DO={}",
                    expected_do, D, DI, DO
                )),
            );
        }

        check
    }

    pub(crate) fn diag<const D: usize, const DO: usize>() -> Self {
        let mut check = Self::Ok;

        if D < 2 {
            check = check.register(
                "Diag",
                TensorError::new(
                    "Diagonal operations require 
                tensors with at least 2 dimensions.",
                )
                .details(format!(
                    "Got tensor with {D} dimensions,
                expected at least 2"
                )),
            );
        }

        if DO != D - 1 {
            check = check.register(
                "Diag",
                TensorError::new("Output rank must be input rank minus 1 for diagonal")
                    .details(format!("Expected output rank {}, got {DO}", D - 1)),
            );
        }

        check
    }

    pub(crate) fn select_assign<const D: usize>(
        dim: usize,
        shape_indices: &Shape,
        shape_value: &Shape,
    ) -> Self {
        let mut check = Self::check_select_basic::<D>(Self::Ok, "Select Assign", dim);

        if shape_value.dims[dim] != shape_indices.dims[0] {
            check = check.register(
                "Select Assign",
                TensorError::new(
                    format!(
                        "Number of indices ({}) should be equal to value tensor dimensions {:?} on axis (dim={dim})",
                        shape_indices.dims[0],
                        shape_value.dims
                    ),
                )
            );
        }

        check
    }

    fn check_select_basic<const D: usize>(mut check: Self, ops: &str, dim: usize) -> Self {
        if dim > D {
            check = check.register(
                ops,
                TensorError::new(format!(
                    "Can't index a tensor with ({D}) dimensions on axis ({dim})"
                )),
            );
        }

        check
    }
    fn check_gather_scatter_indices<const D: usize>(
        mut check: Self,
        ops: &str,
        dim: usize,
        shape: &Shape,
        shape_indices: &Shape,
    ) -> Self {
        if dim > D {
            check = check.register(
                ops,
                TensorError::new(format!(
                    "Can't index a tensor with ({D}) dimensions on axis ({dim})"
                )),
            );
        }

        for i in 0..D {
            if i == dim {
                continue;
            }

            let tensor_dim_i = shape[i];
            let indices_dim_i = shape_indices.dims[i];

            if tensor_dim_i != indices_dim_i {
                check = check.register(
                    ops,
                    TensorError::new(
                        "The tensor shape should be the same as the index tensor shape."
                            .to_string(),
                    )
                    .details(format!(
                        "The shape differs at dimension {i}: {tensor_dim_i} != {indices_dim_i}"
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn check_prelu_shape<const D: usize>(
        shape_tensor: &Shape,
        shape_weight: &Shape,
    ) -> Self {
        let mut check = Self::Ok;
        if shape_weight.dims[0] == 1 {
            check
        } else if D >= 2 {
            let channels = shape_tensor.dims[1];
            let num_weights = shape_weight.dims[0];
            if channels != num_weights {
                check = check.register(
                    "PReLu",
                    TensorError::new(
                        "Number of channels in input tensor and  number of weights must be equal",
                    )
                    .details(format!(
                        "Got no. of channels: {channels}, no. of weights: {num_weights}",
                    )),
                );
                return check;
            }
            check
        } else {
            check = check.register(
                "PReLu",
                TensorError::new(
                    "Number of channels in input tensor and  number of weights must be equal",
                )
                .details(format!(
                    "Got no. of channels: 1, no. of weights: {}",
                    shape_weight.dims[0]
                )),
            );
            check
        }
    }

    /// Checks aggregate dimension such as mean and sum.
    pub(crate) fn aggregate_dim<const D: usize>(ops: &str, dim: usize) -> Self {
        let mut check = Self::Ok;

        if dim > D {
            check = check.register(
                ops,
                TensorError::new(format!(
                    "Can't aggregate a tensor with ({D}) dimensions on axis ({dim})"
                )),
            );
        }

        check
    }

    pub(crate) fn sort_dim<const D: usize>(ops: &str, dim: usize) -> Self {
        let mut check = Self::Ok;

        if dim > D {
            check = check.register(
                ops,
                TensorError::new(format!(
                    "Can't sort a tensor with ({D}) dimensions on axis ({dim})"
                )),
            );
        }

        check
    }

    pub(crate) fn split<const D: usize>(
        tensor_dims: &[usize],
        split_size: usize,
        dim: usize,
    ) -> Self {
        let mut check = Self::Ok;
        let op = "split";
        let tensor_rank = tensor_dims.len();

        if dim >= tensor_rank {
            check = check.register(
                op,
                TensorError::new("Given dimension is greater than or equal to the tensor rank.")
                    .details(format!("Tensor rank: '{D}', given dimension: '{dim}'")),
            );
        } else {
            let tensor_size = tensor_dims[dim];
            if split_size == 0 && tensor_size != 0 {
                check = check.register(
                    op,
                    TensorError::new("split_size must be greater than 0 unless the tensor size along the dimension is 0.")
                        .details(format!("split_size: '{split_size}', tensor size along dim '{dim}': '{tensor_size}'.")),
                );
            }
        }

        check
    }

    pub(crate) fn split_with_sizes<const D: usize>(
        tensor_dims: &[usize],
        split_sizes: &[usize],
        dim: usize,
    ) -> Self {
        let mut check = Self::Ok;
        let op = "split_with_sizes";
        let tensor_rank = tensor_dims.len();

        if dim >= tensor_rank {
            check = check.register(
                op,
                TensorError::new("Given dimension is greater than or equal to the tensor rank.")
                    .details(format!("Tensor rank: '{D}', given dimension: '{dim}'.")),
            );
        } else {
            // Validate split_sizes add up to size of dimension to split along
            let tensor_size = tensor_dims[dim];
            let total_split_size: usize = split_sizes.iter().sum();
            if total_split_size != tensor_size {
                check = check.register(
                    op,
                    TensorError::new("The sum of split_sizes must equal the tensor size along the specified dimension.")
                        .details(format!("Sum of split_sizes: '{total_split_size}', tensor size along dim '{dim}': '{tensor_size}'.")),
                );
            }
        }

        check
    }

    /// The goal is to minimize the cost of checks when there are no error, but it's way less
    /// important when an error occurred, crafting a comprehensive error message is more important
    /// than optimizing string manipulation.
    fn register(self, ops: &str, error: TensorError) -> Self {
        let errors = match self {
            Self::Ok => vec![error],
            Self::Failed(mut failed) => {
                failed.errors.push(error);
                failed.errors
            }
        };

        Self::Failed(FailedTensorCheck {
            ops: ops.to_string(),
            errors,
        })
    }

    /// Checks if shapes are compatible for element wise operations supporting broadcasting.
    pub(crate) fn binary_ops_ew_shape<const D: usize>(
        self,
        ops: &str,
        lhs: &Shape,
        rhs: &Shape,
    ) -> Self {
        let mut check = self;

        for i in 0..D {
            let d_lhs = lhs.dims[i];
            let d_rhs = rhs.dims[i];

            if d_lhs != d_rhs {
                let is_broadcast = d_lhs == 1 || d_rhs == 1;

                if is_broadcast {
                    continue;
                }

                check = check.register(
                    ops,
                    TensorError::new("The provided tensors have incompatible shapes.").details(
                        format!(
                            "Incompatible size at dimension '{}' => '{} != {}', which can't be \
                             broadcasted. Lhs tensor shape {:?}, Rhs tensor shape {:?}.",
                            i, d_lhs, d_rhs, lhs.dims, rhs.dims,
                        ),
                    ),
                );
            }
        }

        check
    }

    /// Checks if tensor devices are equal.
    fn binary_ops_device<Device: PartialEq + core::fmt::Debug>(
        self,
        ops: &str,
        lhs: &Device,
        rhs: &Device,
    ) -> Self {
        match lhs != rhs {
            true => self.register(
                ops,
                TensorError::new("The provided tensors are not on the same device.").details(
                    format!("Lhs tensor device {lhs:?}, Rhs tensor device {rhs:?}.",),
                ),
            ),
            false => self,
        }
    }

    /// Checks if expand operation is possible for the given shapes.
    pub fn expand<const D1: usize, const D2: usize>(ops: &str, shape: &Shape, to: &Shape) -> Self {
        let mut check = TensorCheck::Ok;
        let max_dims = core::cmp::max(D1, D2);

        // Calculate the starting indices for each shape array, ensuring alignment from the right.
        let start_index_shape = max_dims.saturating_sub(D1);
        let start_index_to = max_dims.saturating_sub(D2);

        for i in 0..max_dims {
            // Use 1 as the default dimension size for dimensions beyond the tensor's rank.
            let d_shape = if i >= start_index_shape {
                shape[i - start_index_shape]
            } else {
                1
            };
            let d_to = if i >= start_index_to {
                to.dims[i - start_index_to]
            } else {
                1
            };

            if d_shape != d_to && d_shape != 1 && d_to != 1 {
                // Register an incompatibility error.
                check = check.register(
                    ops,
                    TensorError::new(
                        "The provided tensor can't be broadcasted to the target shape.",
                    )
                    .details(format!(
                        "Incompatible size at dimension '{}' => '{} != {}', which can't be \
                         broadcasted. Tensor shape {:?}, Target shape {:?}.",
                        max_dims - i - 1,
                        d_shape,
                        d_to,
                        shape.dims,
                        to.dims,
                    )),
                );
                break; // Incompatibility found, no need to check further.
            }
        }

        check
    }

    /// Checks if unfold operation is possible for the given shapes.
    pub fn unfold<const D1: usize, const D2: usize>(
        ops: &str,
        _shape: &Shape,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> Self {
        let mut check = TensorCheck::Ok;

        if D2 != D1 + 1 {
            check = check.register(
                ops,
                TensorError::new("The unfold rank is incompatible with the input tensor rank.")
                    .details(format!(
                        "The output rank '{D2}' != the input rank + 1 '{D1}'.",
                    )),
            );
        }

        check
    }

    /// Checks if input is compatible with convolution weights.
    pub fn conv<const D1: usize, const D2: usize>(
        ops: &str,
        x: [usize; D1],
        weight: [usize; D2],
        groups: usize,
    ) -> Self {
        let mut check = TensorCheck::Ok;
        let channels = x[1];
        let expected = weight[1] * groups;
        if channels != expected {
            check = check.register(
                ops,
                TensorError::new("Number of channels in input tensor and input channels of convolution must be equal.")
                .details(format!("got: {channels}, expected: {expected}")),
            );
        }
        check
    }

    /// Checks if input is compatible with transposed convolution weights.
    pub fn conv_transpose<const D1: usize, const D2: usize>(
        ops: &str,
        x: [usize; D1],
        weight: [usize; D2],
    ) -> Self {
        let mut check = TensorCheck::Ok;
        let channels = x[1];
        let expected = weight[0];
        if channels != expected {
            check = check.register(
                ops,
                TensorError::new("Number of channels in input tensor and input channels of convolution must be equal.")
                .details(format!("got: {channels}, expected: {expected}")),
            );
        }
        check
    }

    /// Check if input is compatible with LU decomposition.
    pub fn is_square<const D: usize>(ops: &str, shape: &Shape) -> Self {
        let mut check = TensorCheck::Ok;
        if shape.dims[D - 1] != shape.dims[D - 2] {
            check = check.register(
                ops,
                TensorError::new("The input tensor must be square.").details(format!(
                    "Got tensor with shape {:?}, expected last two dimensions to be equal",
                    shape.dims
                )),
            );
        }
        check
    }

    /// Check pivot is valid for LU decomposition.
    pub fn lu_decomposition_pivot<B: Backend>(pivot: FloatElem<B>) -> Self {
        let mut check = TensorCheck::Ok;
        if pivot.to_f64().abs() <= 1e-6 {
            check = check.register(
                "lu_decomposition",
                TensorError::new("LU decomposition requires a valid pivot.")
                    .details(format!("Got pivot value too close to zero: {}", pivot)),
            );
        }
        check
    }
}

pub(crate) struct FailedTensorCheck {
    ops: String,
    errors: Vec<TensorError>,
}

impl FailedTensorCheck {
    /// Format all the checks into a single message ready to be printed by a [panic](core::panic).
    pub(crate) fn format(self) -> String {
        self.errors.into_iter().enumerate().fold(
            format!(
                "=== Tensor Operation Error ===\n  Operation: '{}'\n  Reason:",
                self.ops
            ),
            |accum, (number, error)| accum + error.format(number + 1).as_str(),
        ) + "\n"
    }
}

struct TensorError {
    description: String,
    details: Option<String>,
}

impl TensorError {
    pub(crate) fn new<S: Into<String>>(description: S) -> Self {
        TensorError {
            description: description.into(),
            details: None,
        }
    }

    pub(crate) fn details<S: Into<String>>(mut self, details: S) -> Self {
        self.details = Some(details.into());
        self
    }

    fn format(self, number: usize) -> String {
        let mut message = format!("\n    {number}. ");
        message += self.description.as_str();
        message += " ";

        if let Some(details) = self.details {
            message += details.as_str();
            message += " ";
        }

        message
    }
}

/// Module where we defined macros that can be used only in the project.
pub(crate) mod macros {
    /// We use a macro for all checks, since the panic message file and line number will match the
    /// function that does the check instead of a generic error.rs crate private unrelated file
    /// and line number.
    macro_rules! check {
        ($check:expr) => {
            if let TensorCheck::Failed(check) = $check {
                core::panic!("{}", check.format());
            }
        };
    }
    pub(crate) use check;
}

#[cfg(test)]
mod tests {
    use super::*;
    use macros::check;

    #[test]
    #[should_panic]
    fn reshape_invalid_shape() {
        check!(TensorCheck::reshape_args_usize::<2, 2>(
            &Shape::new([2, 2]),
            &Shape::new([1, 3])
        ));
    }

    #[test]
    fn reshape_valid_shape() {
        check!(TensorCheck::reshape_args_usize::<2, 2>(
            &Shape::new([2, 2]),
            &Shape::new([1, 4])
        ));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_dimension() {
        let slices = vec![Slice::from(0..2), Slice::from(0..4), Slice::from(1..8)];
        check!(TensorCheck::slice::<3, 3>(&Shape::new([3, 5, 7]), &slices));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_number_of_dimensions() {
        let slices = vec![Slice::from(0..1), Slice::from(0..1), Slice::from(0..1)];
        check!(TensorCheck::slice::<2, 3>(&Shape::new([3, 5]), &slices));
    }

    #[test]
    #[should_panic]
    fn binary_ops_shapes_no_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape::<2>(
            TensorCheck::Ok,
            "TestOps",
            &Shape::new([3, 5]),
            &Shape::new([3, 6])
        ));
    }

    #[test]
    fn binary_ops_shapes_with_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape::<2>(
            TensorCheck::Ok,
            "Test",
            &Shape::new([3, 5]),
            &Shape::new([1, 5])
        ));
    }

    #[test]
    #[should_panic]
    fn binary_ops_devices() {
        check!(TensorCheck::binary_ops_device(
            TensorCheck::Ok,
            "Test",
            &5, // We can pass anything that implements PartialEq as device
            &8
        ));
    }

    #[test]
    #[should_panic]
    fn movedim_args_out_of_bounds() {
        check!(TensorCheck::movedim_args_usize::<3>(5));
    }

    #[test]
    fn movedim_args_i32() {
        check!(TensorCheck::movedim_args_i32::<3>(-3));
    }

    #[test]
    #[should_panic]
    fn movedim_args_too_negative() {
        check!(TensorCheck::movedim_args_i32::<3>(-4));
    }

    #[test]
    #[should_panic]
    fn movedim_args_vec_out_of_bounds() {
        check!(TensorCheck::movedim_args_vec::<3>(&vec![0, 1, 3]));
    }

    #[test]
    #[should_panic]
    fn movedim_args_vec_duplicates() {
        check!(TensorCheck::movedim_args_vec::<3>(&vec![0, 1, 1]));
    }

    #[test]
    #[should_panic]
    fn movedim_args_length() {
        check!(TensorCheck::movedim_args_length(
            &vec![0, 1],
            &vec![0, 1, 2]
        ));
    }

    #[test]
    #[should_panic]
    fn unsqueeze_dim_same_rank() {
        check!(TensorCheck::unsqueeze_dim::<3, 3>(2));
    }
}
