/// We use a macro for all the checks, since the panic message file and line number will match the
/// function that does the check instead of a the generic error.rs crate private unreleated file
/// and line number.
///
/// For readability, every arguments have a name.
#[macro_export(local_inner_macros)]
macro_rules! tensor_check {
    (ops: add, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Add",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: sub, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Add",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: div, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Div",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: mul, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Mul",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: greater, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Greater",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: greater_equal, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Greater Equal",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: lower, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Lower",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (ops: lower_equal, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        crate::tensor_check!(
            binary_ops,
            ops: "Lower Equal",
            shape_lhs: $lhs,
            shape_rhs: $rhs
        );
    };

    (
        ops: reshape,
        shape_original: $shape_original:expr,
        shape_target: $shape_target:expr
    ) => {
        if $shape_original.num_elements() != $shape_target.num_elements() {
            crate::tensor_check!(
                panic,
                ops: "Reshape",
                description: "The given shape doesn't have the same number of elements as the current tensor.",
                details: alloc::format!(
                    "Current shape: {:?}, target shape: {:?}.",
                    $shape_original.dims,
                    $shape_target.dims
                )
            );

        }
    };

    (ops: index, shape: $shape:expr, indexes: $indexes:expr) => {
        let n_dims_tensor = $shape.dims.len();
        let n_dims_indexes = $indexes.len();

        if n_dims_tensor < n_dims_indexes {
            crate::tensor_check!(
                panic,
                ops: "Index",
                description: "The provided indexes array have a higher number of dimensions than the current tensor.",
                details: alloc::format!(
                    "The indexes array must be smaller or equal to the tensor number of dimensions. \
                    Tensor number of dimensions: {n_dims_tensor}, indexes array lenght {n_dims_indexes}."
                )
            );
        }

        for i in 0..n_dims_indexes {
            let d_tensor = $shape.dims[i];
            let index = $indexes.get(i).unwrap();

            if index.end > d_tensor {
                crate::tensor_check!(
                    panic,
                    ops: "Index",
                    description: "The provided indexes array has a range that exceeds the current tensor size.",
                    details: alloc::format!(
                        "The range ({}..{}) exceeds the size of the tensor ({}) at dimension {}. \
                        Tensor shape {:?}, provided indexes {:?}.",
                        index.start,
                        index.end,
                        d_tensor,
                        i,
                        $shape.dims,
                        $indexes,
                    )
                );
            }

            if index.start >= index.end {
                crate::tensor_check!(
                    panic,
                    ops: "Index",
                    description: "The provided indexes array has a range where the start index is bigger or equal to its end.",
                    details: alloc::format!(
                        "The range at dimension '{}' starts at '{}' and is greater or equal to its end '{}'. \
                        Tensor shape {:?}, provided indexes {:?}.",
                        i,
                        index.start,
                        index.end,
                        $shape.dims,
                        $indexes,
                    )
                );
            }
        }
    };

    (binary_ops, ops: $ops:expr, shape_lhs: $lhs:expr, shape_rhs: $rhs:expr) => {
        for i in 0..$lhs.dims.len() {
            let d_lhs = $lhs.dims[i];
            let d_rhs = $rhs.dims[i];

            if d_lhs != d_rhs {
                let is_broadcast = d_lhs == 1 || d_rhs == 1;

                if is_broadcast {
                    continue;
                }

                crate::tensor_check!(
                    panic,
                    ops: $ops,
                    description: "The provided tensors have incompatible shapes.",
                    details: alloc::format!(
                        "Incompatible size at dimension '{}' => '{} != {}', which can't be broadcasted. \
                        Lhs tensor shape {:?}, Rhs tensor shape {:?}.",
                        i,
                        d_lhs,
                        d_rhs,
                        $lhs.dims,
                        $rhs.dims,
                    )
                );
            }
        }
    };

    (panic, ops: $ops:expr, description: $description:expr, details: $details:expr) => {
        core::panic!(
            "Tensor {} Error: \
            {} \
            {}",
            $ops,
            $description,
            $details,
        );
    };
}

#[cfg(test)]
mod tests {
    use crate::Shape;

    #[test]
    #[should_panic]
    fn reshape_invalid_shape() {
        tensor_check!(
            ops: reshape,
            shape_original: &Shape::new([2, 2]),
            shape_target: &Shape::new([1, 3])
        );
    }

    #[test]
    fn reshape_valid_shape() {
        tensor_check!(
            ops: reshape,
            shape_original: &Shape::new([2, 2]),
            shape_target: &Shape::new([1, 4])
        );
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_dimension() {
        tensor_check!(
            ops: index,
            shape: &Shape::new([3, 5, 7]),
            indexes: &[0..2, 0..4, 1..8]
        );
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_number_of_dimensions() {
        tensor_check!(
            ops: index,
            shape: &Shape::new([3, 5]),
            indexes: &[0..1, 0..1, 0..1]
        );
    }

    #[test]
    #[should_panic]
    fn binary_ops_no_broadcast() {
        tensor_check!(
            ops: add,
            shape_lhs: &Shape::new([3, 5]),
            shape_rhs: &Shape::new([3, 6])
        );
    }

    #[test]
    fn binary_ops_with_broadcast() {
        tensor_check!(
            ops: add,
            shape_lhs: &Shape::new([3, 5]),
            shape_rhs: &Shape::new([1, 5])
        );
    }
}
