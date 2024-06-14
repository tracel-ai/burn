use crate::{backend::Backend, BasicOps, Shape, Tensor};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

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
/// implementation might re-implement the same checks, which may result in uncessary code
/// duplication. Maybe a combination of both strategies could help to cover all usecases.
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
            .binary_ops_ew_shape(ops, &lhs.shape(), &rhs.shape())
    }

    pub(crate) fn into_scalar<const D: usize>(shape: &Shape<D>) -> Self {
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
                    "Can't narrow at dimension {}, length must be greater than 0",
                    dim
                )),
            );
        }

        if start >= tensor.shape().dims[dim] {
            check = check.register(
                "Narrow",
                TensorError::new(format!(
                    "Can't narrow at dimension {}, start exceeds the size of the tensor along \
                     this dimension (Size={})",
                    dim,
                    tensor.shape().dims[dim]
                )),
            );
        }

        if start + length > tensor.shape().dims[dim] {
            check = check.register(
                "Narrow",
                TensorError::new(format!(
                    "Can't narrow at dimension {}, start + length exceeds the size of the tensor \
                     along this dimension (Size={})",
                    dim,
                    tensor.shape().dims[dim]
                )),
            );
        }

        check
    }

    pub(crate) fn reshape_args_usize<const D1: usize, const D2: usize>(
        original: &Shape<D1>,
        target: &Shape<D2>,
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

    pub(crate) fn reshape_args_i32<const D: usize>(target: &[i32; D]) -> Self {
        let mut check = Self::Ok;

        if target.iter().any(|&dim| dim < -1) {
            check = check.register(
                "Reshape",
                TensorError::new(
                    "The given shape cannot contain negative dimensions (other than -1).",
                )
                .details(format!("Target shape: {:?}.", target)),
            );
        }

        if target.iter().filter(|&x| x == &-1).count() > 1 {
            check = check.register(
                "Reshape",
                TensorError::new("The given shape cannot contain more than one -1.")
                    .details(format!("Target shape: {:?}.", target)),
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
                    "Current tensor has {D} dimensions, but the given dimensions are {:?}.",
                    dims
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
                                "The dimension {} is duplicated in the given dimensions {:?}.",
                                dim_i, dims
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
                    "Source dimensions: {:?}, Destination dimensions: {:?}.",
                    source_dims, destination_dims
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
                    "The start dim ({start_dim}) must be smaller than the end dim ({end_dim})"
                )),
            );
        }

        if D2 > D1 {
            check = check.register(
                "Flatten",
                TensorError::new(format!("Result dim ({D2}) must be smaller than ({D1})")),
            );
        }

        if D1 < end_dim + 1 {
            check = check.register(
                "Flatten",
                TensorError::new(format!(
                    "The end dim ({end_dim}) must be greater than the tensor dim ({D2})"
                )),
            );
        }

        if D2 < D1 - (end_dim - start_dim) {
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
                    "Can't squeeze dimension {} because its size is not 1",
                    dim
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
                    "Resulting dimensions {} do not match the required D2 size {}.",
                    new_dims_len, D2
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
                    "Can't unsqueeze smaller tensor, got dim {D2}, expected > {D1}"
                )),
            );
        }

        check
    }

    pub(crate) fn unsqueeze_dim<const D: usize>(dim: usize) -> Self {
        let mut check = Self::Ok;
        if dim > D {
            check = check.register(
                "Unsqueeze",
                TensorError::new(format!(
                    "Can't unsqueeze at dimension {}, exceeds tensor dimensions (D={})",
                    dim, D
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
                    "unsqueeze arg {} is out of range for the output tensor of rank {}",
                    dim, output_rank
                )),
            );
        }
        check
    }

    pub(crate) fn one_hot(index: usize, num_classes: usize) -> Self {
        let mut check = Self::Ok;
        if index >= num_classes {
            check = check.register(
                "One Hot",
                TensorError::new(format!(
                    "Can't create a one hot tensor with index ({index}) greater or equal to the number of classes ({num_classes})",
                )),
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

    pub(crate) fn matmul<B: Backend, const D: usize>(
        lhs: &Tensor<B, D>,
        rhs: &Tensor<B, D>,
    ) -> Self {
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

    pub(crate) fn stack<B: Backend, const D: usize, K: BasicOps<B>>(
        tensors: &[Tensor<B, D, K>],
        dim: usize,
    ) -> Self {
        let mut check = Self::Ok;

        if dim > D {
            check = check.register(
                "Stack",
                TensorError::new(
                    "Can't stack tensors on a dim that exceeds the tensors dimension (inclusive)",
                )
                .details(format!(
                    "Trying to concatenate tensors with {D} dimensions on axis {dim}."
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
                        "Provided dimension ({}), tensors shapes: {:?}",
                        dim,
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
            shape.dims[dim] = 1; // Ignore the concatenate dim.

            if shape_reference != shape {
                return check.register(
                    "Cat",
                    TensorError::new(
                        "Can't concatenate tensors with different shapes, except for the provided \
                         dimension",
                    )
                    .details(format!(
                        "Provided dimension ({}), tensors shapes: {:?}",
                        dim,
                        tensors.iter().map(Tensor::shape).collect::<Vec<_>>()
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn slice<const D1: usize, const D2: usize>(
        shape: &Shape<D1>,
        ranges: &[Range<usize>; D2],
    ) -> Self {
        let mut check = Self::Ok;
        let n_dims_tensor = D1;
        let n_dims_ranges = D2;

        if n_dims_tensor < n_dims_ranges {
            check = check.register(
                "Slice",
                TensorError::new(
                    "The provided ranges array has a higher number of dimensions than the current \
                     tensor.",
                )
                .details(format!(
                    "The ranges array must be smaller or equal to the tensor number of \
                     dimensions. Tensor number of dimensions: {n_dims_tensor}, ranges array \
                     length {n_dims_ranges}."
                )),
            );
        }

        for i in 0..usize::min(D1, D2) {
            let d_tensor = shape.dims[i];
            let range = ranges.get(i).unwrap();

            if range.end > d_tensor {
                check = check.register(
                    "Slice",
                    TensorError::new(
                        "The provided ranges array has a range that exceeds the current tensor \
                         size.",
                    )
                    .details(format!(
                        "The range ({}..{}) exceeds the size of the tensor ({}) at dimension {}. \
                         Tensor shape {:?}, provided ranges {:?}.",
                        range.start, range.end, d_tensor, i, shape.dims, ranges,
                    )),
                );
            }

            if range.start >= range.end {
                check = check.register(
                    "Slice",
                    TensorError::new(
                        "The provided range array has a range where the start index is bigger or \
                         equal to its end.",
                    )
                    .details(format!(
                        "The range at dimension '{}' starts at '{}' and is greater or equal to \
                         its end '{}'. Tensor shape {:?}, provided ranges {:?}.",
                        i, range.start, range.end, shape.dims, ranges,
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn slice_assign<const D1: usize, const D2: usize>(
        shape: &Shape<D1>,
        shape_value: &Shape<D1>,
        ranges: &[Range<usize>; D2],
    ) -> Self {
        let mut check = Self::Ok;

        if D1 < D2 {
            check = check.register(
                "Slice Assign",
                TensorError::new(
                    "The provided ranges array has a higher number of dimensions than the current \
                     tensor.",
                )
                .details(format!(
                    "The ranges array must be smaller or equal to the tensor number of \
                     dimensions. Tensor number of dimensions: {D1}, ranges array length {D2}."
                )),
            );
        }

        for i in 0..usize::min(D1, D2) {
            let d_tensor = shape.dims[i];
            let d_tensor_value = shape_value.dims[i];
            let range = ranges.get(i).unwrap();

            if range.end > d_tensor {
                check = check.register(
                    "Range Assign",
                    TensorError::new(
                        "The provided ranges array has a range that exceeds the current tensor \
                         size.",
                    )
                    .details(format!(
                        "The range ({}..{}) exceeds the size of the tensor ({}) at dimension {}. \
                         Current tensor shape {:?}, value tensor shape {:?}, provided ranges {:?}.",
                        range.start, range.end, d_tensor, i, shape.dims, shape_value.dims, ranges,
                    )),
                );
            }

            if range.end - range.start != d_tensor_value {
                check = check.register(
                    "Slice Assign",
                    TensorError::new(
                        "The value tensor must match the amount of elements selected with the \
                         ranges array",
                    )
                    .details(format!(
                        "The range ({}..{}) doesn't match the number of elements of the value \
                         tensor ({}) at dimension {}. Current tensor shape {:?}, value tensor \
                         shape {:?}, provided ranges {:?}.",
                        range.start,
                        range.end,
                        d_tensor_value,
                        i,
                        shape.dims,
                        shape_value.dims,
                        ranges,
                    )),
                );
            }

            if range.start >= range.end {
                check = check.register(
                    "Slice Assign",
                    TensorError::new(
                        "The provided ranges array has a range where the start index is bigger or \
                         equal to its end.",
                    )
                    .details(format!(
                        "The range at dimension '{}' starts at '{}' and is greater or equal to \
                         its end '{}'. Current tensor shape {:?}, value tensor shape {:?}, \
                         provided ranges {:?}.",
                        i, range.start, range.end, shape.dims, shape_value.dims, ranges,
                    )),
                );
            }
        }

        check
    }

    pub(crate) fn gather<const D: usize>(
        dim: usize,
        shape: &Shape<D>,
        shape_indices: &Shape<D>,
    ) -> Self {
        Self::check_gather_scatter_indices(Self::Ok, "Gather", dim, shape, shape_indices)
    }

    pub(crate) fn scatter<const D: usize>(
        dim: usize,
        shape: &Shape<D>,
        shape_indices: &Shape<D>,
        shape_value: &Shape<D>,
    ) -> Self {
        let ops = "Scatter";
        let mut check =
            Self::check_gather_scatter_indices(Self::Ok, ops, dim, shape, shape_indices);

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

    pub(crate) fn select_assign<const D: usize>(dim: usize) -> Self {
        Self::check_select_basic::<D>(Self::Ok, "select_assign", dim)
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
        shape: &Shape<D>,
        shape_indices: &Shape<D>,
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

            let tensor_dim_i = shape.dims[i];
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
        shape_tensor: &Shape<D>,
        shape_weight: &Shape<1>,
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
                        "Got no. of channels: {}, no. of weights: {}",
                        channels, num_weights
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
                    "Got no. of channels: {}, no. of weights: {}",
                    1, shape_weight.dims[0]
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
        lhs: &Shape<D>,
        rhs: &Shape<D>,
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
    pub fn expand<const D1: usize, const D2: usize>(
        ops: &str,
        shape: &Shape<D1>,
        to: &Shape<D2>,
    ) -> Self {
        let mut check = TensorCheck::Ok;
        let max_dims = core::cmp::max(D1, D2);

        // Calculate the starting indices for each shape array, ensuring alignment from the right.
        let start_index_shape = max_dims.saturating_sub(D1);
        let start_index_to = max_dims.saturating_sub(D2);

        for i in 0..max_dims {
            // Use 1 as the default dimension size for dimensions beyond the tensor's rank.
            let d_shape = if i >= start_index_shape {
                shape.dims[i - start_index_shape]
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
    /// function that does the check instead of a the generic error.rs crate private unrelated file
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
        check!(TensorCheck::reshape_args_usize(
            &Shape::new([2, 2]),
            &Shape::new([1, 3])
        ));
    }

    #[test]
    fn reshape_valid_shape() {
        check!(TensorCheck::reshape_args_usize(
            &Shape::new([2, 2]),
            &Shape::new([1, 4])
        ));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_dimension() {
        check!(TensorCheck::slice(
            &Shape::new([3, 5, 7]),
            &[0..2, 0..4, 1..8]
        ));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_number_of_dimensions() {
        check!(TensorCheck::slice(&Shape::new([3, 5]), &[0..1, 0..1, 0..1]));
    }

    #[test]
    #[should_panic]
    fn binary_ops_shapes_no_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape(
            TensorCheck::Ok,
            "TestOps",
            &Shape::new([3, 5]),
            &Shape::new([3, 6])
        ));
    }

    #[test]
    fn binary_ops_shapes_with_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape(
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
}
