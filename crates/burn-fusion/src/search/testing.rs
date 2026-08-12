//! Shared IR constructors for the search tests.

use burn_backend::{DType, Shape};
use burn_ir::{BinaryOpIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus};

/// A `[32, 32]` float tensor with the given id, read with `ReadOnly` status.
pub(crate) fn tensor(id: u64) -> TensorIr {
    TensorIr {
        id: TensorId::new(id),
        shape: Shape::new([32, 32]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    }
}

/// An add operation reading `lhs` and `rhs` and producing `out`.
pub(crate) fn add(lhs: u64, rhs: u64, out: u64) -> OperationIr {
    OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs: tensor(lhs),
            rhs: tensor(rhs),
            out: tensor(out),
        }),
    )
}

/// Like [add] but reads `lhs` with [ReadWrite](TensorStatus::ReadWrite) status — it is the last
/// use and frees the tensor.
pub(crate) fn add_rw(lhs: u64, rhs: u64, out: u64) -> OperationIr {
    let mut lhs = tensor(lhs);
    lhs.status = TensorStatus::ReadWrite;
    OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs,
            rhs: tensor(rhs),
            out: tensor(out),
        }),
    )
}
