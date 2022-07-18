use crate::{
    node::{NodeRef, Ones, Zeros},
    ops::InitRecordedOps,
    tape::TapeRef,
    FloatTensor, Shape,
};
use num_traits::Float;

#[derive(Debug)]
pub struct ADTensor<P, const D: usize, T> {
    pub node: NodeRef<T>,
    pub shape: Shape<D>,
    pub kind: ADKind<P>,
    pub tape: TapeRef,
}

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    P: Float + Zeros<P> + Default + 'static,
    T: FloatTensor<P, D> + Clone + Zeros<T> + Ones<T> + 'static,
{
    pub fn new(node: NodeRef<T>, tape: TapeRef) -> Self {
        let tensor = node.borrow().value();
        let shape = tensor.shape().clone();
        let kind = ADKind::new();

        let ops = InitRecordedOps::new(node.clone());
        let ops = Box::new(ops);
        tape.borrow_mut().add(ops);

        Self {
            node,
            shape,
            kind,
            tape,
        }
    }

    pub fn from_existing(&self, node: NodeRef<T>) -> Self {
        let tape = self.tape.clone();
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            node,
            shape,
            kind,
            tape,
        }
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T> {
    pub fn tensor(&self) -> T {
        self.node.borrow().value()
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T> {
    pub fn backprob(&self) {
        let id = self.node.borrow().id();
        self.tape.borrow_mut().backward(id);
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T> {
    pub fn grad(&self) -> T {
        self.node.borrow_mut().grad()
    }
}

#[derive(Clone, Debug)]
pub struct ADKind<P> {
    _p: P,
}

impl<P: Float + Default> ADKind<P> {
    pub fn new() -> Self {
        Self { _p: P::default() }
    }
}
