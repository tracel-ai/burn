use std::{ops::Range, sync::atomic::AtomicU64};

const ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TensorId {
    value: u64,
}

#[derive(Clone, Debug)]
pub struct TensorDefinition {
    pub id: TensorId,
    pub shape: Vec<usize>,
}

impl TensorId {
    pub(crate) fn new() -> Self {
        let id = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self { value: id.into() }
    }
}

#[derive(Clone, Debug)]
pub enum FloatOps {
    Unary(UnaryOps),
    Binary(BinaryOps),
    Index(IndexOps),
}

#[derive(Clone, Debug)]
pub enum UnaryOps {
    Add,
    Relu,
}

#[derive(Clone, Debug)]
pub enum BinaryOps {
    Add,
    Matmul,
}

#[derive(Clone, Debug)]
pub enum IndexOps {
    Slice {
        tensor: TensorDefinition,
        ranges: Vec<Range<usize>>,
    },
    SliceAssign {
        tensor: TensorDefinition,
        ranges: Vec<Range<usize>>,
        value: TensorDefinition,
    },
}

#[derive(new)]
pub struct Graph<Handle> {
    operations: Vec<FloatOps>,
    inputs: Vec<(TensorId, Handle)>,
    outputs: Vec<(TensorId, Handle)>,
}
