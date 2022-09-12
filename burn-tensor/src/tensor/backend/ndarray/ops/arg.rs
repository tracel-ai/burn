use crate::backend::ndarray::NdArrayBackend;
use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use crate::{Data, NdArrayElement};
use std::cmp::Ordering;

impl<E, const D: usize> TensorOpsArg<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn argmax(&self, dim: usize) -> NdArrayTensor<i64, D> {
        arg(self, dim, cmp_min)
    }

    fn argmin(&self, dim: usize) -> NdArrayTensor<i64, D> {
        arg(self, dim, cmp_max)
    }
}

fn arg<E: NdArrayElement, F, const D: usize>(
    tensor: &NdArrayTensor<E, D>,
    dim: usize,
    cmp: F,
) -> NdArrayTensor<i64, D>
where
    F: Fn(&f64, &f64) -> Ordering,
{
    let mut data = tensor.to_data();
    let batch_size = tensor.shape.dims[dim];
    let mut start = 0;
    let mut end = tensor.shape.dims[dim];
    let mut output = Vec::new();

    while end <= data.value.len() {
        let data_dim = &mut data.value[start..end];
        let mut sorted: Vec<f64> = data_dim.iter().map(|a| a.to_elem()).collect();
        sorted.sort_by(&cmp);

        let max = sorted[0];

        let data_dim = &mut data.value[start..end];
        let mut index: i64 = 0;
        for elem in data_dim {
            let as_float: f64 = elem.to_elem();
            if as_float == max {
                break;
            }
            index += 1;
        }
        output.push(index);
        start += batch_size;
        end += batch_size;
    }
    let mut shape = tensor.shape;
    shape.dims[dim] = 1;
    NdArrayTensor::from_data(Data::new(output, shape))
}

fn cmp_max(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    Ordering::Equal
}

fn cmp_min(a: &f64, b: &f64) -> Ordering {
    if a > b {
        return Ordering::Less;
    } else if a < b {
        return Ordering::Greater;
    }
    Ordering::Equal
}
