use crate::backend::ndarray::NdArrayBackend;
use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use crate::ElementValue;
use crate::NdArrayElement;
use std::cmp::Ordering;

impl<E, const D: usize> TensorOpsArg<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn argmax(&self, dim: usize) -> NdArrayTensor<i64, D> {
        arg(self, dim, cmp_max)
    }

    fn argmin(&self, dim: usize) -> NdArrayTensor<i64, D> {
        arg(self, dim, cmp_min)
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
    let mut start = 1;

    for i in 0..dim {
        start = start * tensor.shape.dims[i];
    }
    let end = start + tensor.shape.dims[dim];

    let data_dim = &mut data.value[start..end];
    let mut sorted: Vec<f64> = data_dim.iter().map(|a| a.to_elem()).collect();
    sorted.sort_by(cmp);

    let max = sorted[0];
    for elem in data_dim {
        *elem = <E as ElementValue>::zero();
    }

    let data_dim = &mut data.value[start..end];
    for elem in data_dim {
        let as_float: f64 = elem.to_elem();
        if as_float == max {
            *elem = <E as ElementValue>::one();
            break;
        }
    }

    NdArrayTensor::from_data(data.convert())
}

fn cmp_max(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

fn cmp_min(a: &f64, b: &f64) -> Ordering {
    if a > b {
        return Ordering::Less;
    } else if a < b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}
