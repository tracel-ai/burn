pub trait ReLU<E, const D: usize> {
    fn relu(&self) -> Self;
}

pub trait Softmax<E, const D: usize> {
    fn softmax(&self, dim: usize) -> Self;
}
