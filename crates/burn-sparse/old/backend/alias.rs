use crate::backend::SparseBackend;

/// Sparse tensor primitive type used by the backend.
pub type SparseTensor<B, const D: usize> = <B as SparseBackend>::SparseTensorPrimitive<D>;
