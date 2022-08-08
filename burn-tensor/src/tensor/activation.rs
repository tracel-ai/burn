pub trait ReLU<E, const D: usize> {
    fn relu(&self) -> Self;
}
