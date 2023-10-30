use crate::FusionTensor;
use burn_tensor::backend::Backend;
use std::ops::Range;

pub enum FloatOps<B: Backend, E> {
    Unary {
        ops: UnaryOps,
        tensor: FusionTensor<B, E>,
    },
    Binary {
        ops: BinaryOps,
        lhs: FusionTensor<B, E>,
        rhs: FusionTensor<B, E>,
    },
    Index(IndexOps<B, E>),
}

pub enum UnaryOps {
    Add,
    Relu,
}

pub enum BinaryOps {
    Add,
    Matmul,
}

pub enum IndexOps<B: Backend, E> {
    Slice {
        tensor: FusionTensor<B, E>,
        ranges: Vec<Range<usize>>,
    },
    SliceAssign {
        tensor: FusionTensor<B, E>,
        ranges: Vec<Range<usize>>,
        value: FusionTensor<B, E>,
    },
}
