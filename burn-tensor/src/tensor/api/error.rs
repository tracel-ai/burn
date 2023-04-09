use crate::{backend::Backend, BasicOps, Shape, Tensor};
use alloc::format;
use core::ops::Range;

pub struct TensorCheck {
    ops: String,
    errors: Vec<TensorError>,
}

impl TensorCheck {
    /// Checks device and shape compatibility for element wise binary operations.
    pub fn binary_ops_ew<B: Backend, const D: usize, K: BasicOps<B>>(
        ops: &str,
        lhs: &Tensor<B, D, K>,
        rhs: &Tensor<B, D, K>,
    ) -> Option<TensorCheck> {
        let mut check = None;
        check = Self::binary_ops_device(ops, check, &lhs.device(), &rhs.device());
        check = Self::binary_ops_ew_shape(ops, check, &lhs.shape(), &rhs.shape());
        check
    }

    /// Checks if the original shape can be reshape to the target one.
    pub fn reshape<const D1: usize, const D2: usize>(
        original: &Shape<D1>,
        target: &Shape<D2>,
    ) -> Option<TensorCheck> {
        let mut check = None;

        if original.num_elements() != target.num_elements() {
            check = TensorCheck::register("Reshape", check, TensorError::new(
                "The given shape doesn't have the same number of elements as the current tensor.",
            )
            .details(format!(
                "Current shape: {:?}, target shape: {:?}.",
                original.dims, target.dims
            )));
        }

        check
    }

    /// Checks if the current tensor shape can be indexed with the given indexes.
    pub fn index<const D1: usize, const D2: usize>(
        shape: &Shape<D1>,
        indexes: &[Range<usize>; D2],
    ) -> Option<TensorCheck> {
        let mut check = None;
        let n_dims_tensor = D1;
        let n_dims_indexes = D2;

        if n_dims_tensor < n_dims_indexes {
            check = TensorCheck::register("Index", check,
                TensorError::new ("The provided indexes array has a higher number of dimensions than the current tensor.")
                .details(
                    format!(
                    "The indexes array must be smaller or equal to the tensor number of dimensions. \
                    Tensor number of dimensions: {n_dims_tensor}, indexes array lenght {n_dims_indexes}."
                )));
        }

        for i in 0..usize::min(D1, D2) {
            let d_tensor = shape.dims[i];
            let index = indexes.get(i).unwrap();

            if index.end > d_tensor {
                check = TensorCheck::register(
                    "Index",
                    check,
                    TensorError::new("The provided indexes array has a range that exceeds the current tensor size.")
                .details(alloc::format!(
                        "The range ({}..{}) exceeds the size of the tensor ({}) at dimension {}. \
                        Tensor shape {:?}, provided indexes {:?}.",
                        index.start,
                        index.end,
                        d_tensor,
                        i,
                        shape.dims,
                        indexes,
                    )));
            }

            if index.start >= index.end {
                check = TensorCheck::register(
                    "Index",
                    check,
                    TensorError::new("The provided indexes array has a range where the start index is bigger or equal to its end.")
                    .details(alloc::format!(
                        "The range at dimension '{}' starts at '{}' and is greater or equal to its end '{}'. \
                        Tensor shape {:?}, provided indexes {:?}.",
                        i,
                        index.start,
                        index.end,
                        shape.dims,
                        indexes,
                    )));
            }
        }

        check
    }

    /// The goal is to minimize the cost of checks when there are no error, but it's way less
    /// important when an error occured, crafting a comprehensive error message is more important
    /// than optimizing string manipulation.
    fn register(ops: &str, check: Option<TensorCheck>, error: TensorError) -> Option<TensorCheck> {
        let mut check = match check {
            Some(check) => check,
            None => TensorCheck {
                ops: ops.to_string(),
                errors: vec![],
            },
        };

        check.errors.push(error);

        return Some(check);
    }

    /// Format all the checks into a single message ready to be printed by a [panic](core::panic).
    pub fn format(self) -> String {
        self.errors.into_iter().enumerate().fold(
            format!(
                "=== Tensor Operation Error ===\n  Operation: '{}'\n  Reason:",
                self.ops
            ),
            |accum, (number, error)| accum + error.format(number + 1).as_str(),
        ) + "\n"
    }

    /// Checks if shapes are compatible for element wise operations supporting broadcasting.
    pub fn binary_ops_ew_shape<const D: usize>(
        ops: &str,
        mut check: Option<TensorCheck>,
        lhs: &Shape<D>,
        rhs: &Shape<D>,
    ) -> Option<TensorCheck> {
        for i in 0..D {
            let d_lhs = lhs.dims[i];
            let d_rhs = rhs.dims[i];

            if d_lhs != d_rhs {
                let is_broadcast = d_lhs == 1 || d_rhs == 1;

                if is_broadcast {
                    continue;
                }

                check =
                 TensorCheck::register(ops, check, 
                 TensorError::new("The provided tensors have incompatible shapes.")
                 .details(format!(
                     "Incompatible size at dimension '{}' => '{} != {}', which can't be broadcasted. \
                     Lhs tensor shape {:?}, Rhs tensor shape {:?}.",
                     i,
                     d_lhs,
                     d_rhs,
                     lhs.dims,
                     rhs.dims,
                 )));
            }
        }

        check
    }

    /// Checks if tensor devices are equal.
    fn binary_ops_device<Device: PartialEq + core::fmt::Debug>(
        ops: &str,
        mut check: Option<TensorCheck>,
        lhs: &Device,
        rhs: &Device,
    ) -> Option<TensorCheck> {
        if lhs != rhs {
            check = TensorCheck::register(
                ops,
                check,
                TensorError::new("The provided tensors are not on the same device.").details(
                    format!("Lhs tensor device {:?}, Rhs tensor device {:?}.", lhs, rhs,),
                ),
            );
        }

        check
    }
}

struct TensorError {
    description: String,
    details: Option<String>,
}

impl TensorError {
    pub fn new(description: &str) -> Self {
        TensorError {
            description: description.into(),
            details: None,
        }
    }

    pub fn details<S: Into<String>>(mut self, details: S) -> Self {
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

/// We use a macro for all checks, since the panic message file and line number will match the
/// function that does the check instead of a the generic error.rs crate private unreleated file
/// and line number.
#[macro_export(local_inner_macros)]
macro_rules! check {
    ($check:expr) => {
        if let Some(check) = $check {
            core::panic!("{}", check.format());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn reshape_invalid_shape() {
        check!(TensorCheck::reshape(
            &Shape::new([2, 2]),
            &Shape::new([1, 3])
        ));
    }

    #[test]
    fn reshape_valid_shape() {
        check!(TensorCheck::reshape(
            &Shape::new([2, 2]),
            &Shape::new([1, 4])
        ));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_dimension() {
        check!(TensorCheck::index(
            &Shape::new([3, 5, 7]),
            &[0..2, 0..4, 1..8]
        ));
    }

    #[test]
    #[should_panic]
    fn index_range_exceed_number_of_dimensions() {
        check!(TensorCheck::index(&Shape::new([3, 5]), &[0..1, 0..1, 0..1]));
    }

    #[test]
    #[should_panic]
    fn binary_ops_shapes_no_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape(
            "TestOps",
            None,
            &Shape::new([3, 5]),
            &Shape::new([3, 6])
        ));
    }

    #[test]
    fn binary_ops_shapes_with_broadcast() {
        check!(TensorCheck::binary_ops_ew_shape(
            "Test",
            None,
            &Shape::new([3, 5]),
            &Shape::new([1, 5])
        ));
    }

    #[test]
    #[should_panic]
    fn binary_ops_devices() {
        check!(TensorCheck::binary_ops_device(
            "Test", None, &5, // We can pass anything that implements PartialEq as device
            &8
        ));
    }
}
