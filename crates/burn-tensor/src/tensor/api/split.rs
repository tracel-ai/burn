use super::narrow::narrow;
use crate::{backend::Backend, BasicOps, TensorKind};
use alloc::vec::Vec;

pub fn split<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    split_size: usize,
    dim: usize,
) -> Vec<K::Primitive> {
    let size = K::shape(&tensor).dims[dim];
    let mut tensors = Vec::new();

    let mut start = 0;
    while start < size {
        let length = usize::min(split_size, size - start);
        tensors.push(narrow::<B, K>(tensor.clone(), dim, start, length));
        start += length;
    }

    tensors
}

pub fn split_with_size<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    split_sizes: Vec<usize>,
    dim: usize,
) -> Vec<K::Primitive> {
    let mut tensors = Vec::new();

    let mut start = 0;
    for length in split_sizes {
        if length == 0 {
            continue;
        }
        tensors.push(narrow::<B, K>(tensor.clone(), dim, start, length));
        start += length;
    }

    tensors
}
