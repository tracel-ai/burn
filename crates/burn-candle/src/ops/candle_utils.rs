use candle_core::{DType, Device, Shape, Tensor};

use crate::element::CandleElement;

pub(crate) fn fill<E: CandleElement, S: Into<Shape>>(
    value: E,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Tensor {
    let values = (Tensor::ones((1), dtype, device).unwrap() * value.elem::<f64>()).unwrap();
    values.expand(shape).unwrap()
}

pub(crate) fn fill_like<E: CandleElement>(value: E, reference_tensor: &Tensor) -> Tensor {
    fill(
        value,
        reference_tensor.shape(),
        reference_tensor.dtype(),
        reference_tensor.device(),
    )
}

/// Implementation based on PyTorch Cuda Kernel: https://fossies.org/linux/pytorch/aten/src/ATen/native/cuda/MultinomialKernel.cu
pub(crate) fn multinomial<E: CandleElement>(
    props: &[f64],
    num_samples: usize,
    device: &Device,
) -> Tensor {
    if props.len() == 0 {
        return Tensor::from_iter([0f64; 0].into_iter(), device)
            .unwrap()
            .to_dtype(E::DTYPE)
            .unwrap();
    }
    let p = Tensor::from_iter(props.into_iter().cloned(), &device).unwrap();
    let p_cum = p.cumsum(0).unwrap();
    let p_sum = p_cum.get(props.len() - 1).unwrap();
    let p = p.broadcast_div(&p_sum).unwrap(); // Binary search each probability in parallel, using index_select
    let randos = Tensor::rand(0.0, 1.0, num_samples, &device).unwrap();
    let mut starts = Tensor::zeros(randos.shape(), DType::I64, &device).unwrap();
    let mut ends = Tensor::full(p.dims1().unwrap() as i64 - 1, randos.shape(), &device).unwrap();
    let mut mids;
    let mut mid_vals; // Helper vals for scalar operations
    let twos = Tensor::full(2i64, randos.shape(), &device).unwrap();
    let ones = Tensor::full(1i64, randos.shape(), &device).unwrap();
    let mut ends_gt_starts = ends
        .gt(&starts)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    while ends_gt_starts > 0.0 {
        mids = (&starts + (&ends - &starts).unwrap().div(&twos).unwrap()).unwrap();
        mid_vals = p_cum.index_select(&mids, 0).unwrap();
        let mid_val_less_than_val = mid_vals.lt(&randos).unwrap();
        let new_starts = (&mids + &ones).unwrap();
        let new_ends = &mids;
        starts = mid_val_less_than_val
            .where_cond(&new_starts, &starts)
            .unwrap();
        ends = mid_val_less_than_val.where_cond(&ends, &new_ends).unwrap();
        ends_gt_starts = ends
            .gt(&starts)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
    }
    let size = Tensor::full(num_samples as i64, randos.shape(), &device).unwrap();
    let starts_are_size = starts.eq(&size).unwrap();
    let new_starts = (&size - &ones).unwrap();
    starts = starts_are_size.where_cond(&new_starts, &starts).unwrap();
    let mut starts_above_one = starts.gt(1i64).unwrap();
    let mut prob_is_zero = p.index_select(&starts, 0).unwrap().eq(0f64).unwrap();
    let mut both_true = (&starts_above_one + &prob_is_zero)
        .unwrap()
        .eq(2u8)
        .unwrap();
    let mut any_true = both_true
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    while any_true > 0.0 {
        starts = both_true
            .where_cond(&(&starts - &ones).unwrap(), &starts)
            .unwrap();
        starts_above_one = starts.gt(1i64).unwrap();
        prob_is_zero = p.index_select(&starts, 0).unwrap().gt(0f64).unwrap();
        both_true = (&starts_above_one + &prob_is_zero)
            .unwrap()
            .eq(2u8)
            .unwrap();
        any_true = both_true
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
    }
    let final_result = starts;
    return final_result.to_dtype(E::DTYPE).unwrap();
}
