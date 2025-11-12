use burn_common::rand::get_seeded_rng;
use burn_complex::base::{
    ComplexTensor, ComplexTensorBackend, ComplexTensorOps, InterleavedLayout, element::Complex32,
};
use ndarray::{ArrayD, IxDyn};

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor, QuantElement,
    SEED, SharedArray,
};
use burn_tensor::{Distribution, Shape, TensorData, TensorMetadata, backend::Backend};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ComplexTensorBackend
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type InnerBackend = NdArray<E, I, Q>;

    type ComplexTensorPrimitive = NdArrayTensor;
    type ComplexElem = Complex32;

    type Layout = InterleavedLayout;

    fn real(tensor: ComplexTensor<Self>) -> NdArrayTensor {
        todo!()
    }

    fn imag(tensor: ComplexTensor<Self>) -> NdArrayTensor {
        todo!()
    }

    fn to_complex(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement>
    ComplexTensorOps<NdArray<E, I, Q>> for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Layout = burn_complex::base::InterleavedLayout;
    fn complex_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor {
        NdArrayTensor::from_data(data)
    }

    fn complex_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let data = TensorData::random::<Complex32, _, _>(shape, distribution, &mut rng);
        *seed = Some(rng);
        NdArrayTensor::from_data(data)
    }

    fn complex_shape(tensor: &NdArrayTensor) -> Shape {
        tensor.shape()
    }

    fn complex_to_data(tensor: &NdArrayTensor) -> TensorData {
        // let shape = tensor.shape();
        // let vec: Vec<Complex32> = tensor.array.iter().cloned().collect();
        // TensorData::new(vec, shape)
        todo!()
    }

    fn complex_device(_tensor: &NdArrayTensor) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn complex_to_device(tensor: NdArrayTensor, _device: &NdArrayDevice) -> NdArrayTensor {
        tensor
    }

    fn complex_into_data(tensor: NdArrayTensor) -> TensorData {
        tensor.into_data()
    }

    fn complex_reshape(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        todo!()
    }

    fn complex_transpose(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_mul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        // let result = lhs
        //     .array
        //     .iter()
        //     .zip(rhs.array.iter())
        //     .map(|(a, b)| Complex32 {
        //         real: a.real * b.real - a.imag * b.imag,
        //         imag: a.real * b.imag + a.imag * b.real,
        //     })
        //     .collect::<Vec<_>>();

        // let shape = lhs.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), result).unwrap();
        // NdArrayTensor::new(array.into_shared())
        todo!()
    }

    fn complex_div(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
        // let result = lhs
        //     .array
        //     .iter()
        //     .zip(rhs.array.iter())
        //     .map(|(a, b)| {
        //         let denom = b.real * b.real + b.imag * b.imag;
        //         Complex32 {
        //             real: (a.real * b.real + a.imag * b.imag) / denom,
        //             imag: (a.imag * b.real - a.real * b.imag) / denom,
        //         }
        //     })
        //     .collect::<Vec<_>>();

        // let shape = lhs.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), result).unwrap();
        // NdArrayTensor::new(array.into_shared())
        todo!()
    }

    fn complex_neg(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| Complex32 {
        //     real: -c.real,
        //     imag: -c.imag,
        // });
        // todo!() //NdArrayTensor::new(array.into())
        todo!()
    }

    fn complex_conj(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| Complex32 {
        //     real: c.real,
        //     imag: -c.imag,
        // });
        // todo!() //NdArrayTensor::new(array.into())
        todo!()
    }

    fn complex_abs(tensor: NdArrayTensor) -> <Self as Backend>::FloatTensorPrimitive {
        // let abs_data: Vec<f32> = tensor
        //     .array
        //     .iter()
        //     .map(|c| (c.real * c.real + c.imag * c.imag).sqrt())
        //     .collect();
        // let shape = tensor.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), abs_data).unwrap();
        // let abs_tensor = NdArrayTensor::new(array.into_shared());
        //abs_tensor.into()
        todo!()
    }

    fn complex_arg(tensor: NdArrayTensor) -> <Self as Backend>::FloatTensorPrimitive {
        // let arg_data: Vec<f32> = tensor.array.iter().map(|c| c.imag.atan2(c.real)).collect();
        // let shape = tensor.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), arg_data).unwrap();
        // let arg_tensor = NdArrayTensor::new(array.into_shared());
        // arg_tensor.into()
        todo!()
    }

    fn complex_from_parts(
        real: <Self as Backend>::FloatTensorPrimitive,
        imag: <Self as Backend>::FloatTensorPrimitive,
    ) -> NdArrayTensor {
        // Extract real and imaginary parts as f32 tensors
        // let real_f32 = match real {
        //     NdArrayTensorFloat::F32(tensor) => tensor,
        //     NdArrayTensorFloat::F64(tensor) => {
        //         let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
        //         let shape = tensor.shape();
        //         let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
        //         NdArrayTensor::new(array.into_shared())
        //     }
        // };

        // let imag_f32 = match imag {
        //     NdArrayTensorFloat::F32(tensor) => tensor,
        //     NdArrayTensorFloat::F64(tensor) => {
        //         let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
        //         let shape = tensor.shape();
        //         let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
        //         NdArrayTensor::new(array.into_shared())
        //     }
        // };

        // let complex_data: Vec<Complex32> = real_f32
        //     .array
        //     .iter()
        //     .zip(imag_f32.array.iter())
        //     .map(|(&r, &i)| Complex32 { real: r, imag: i })
        //     .collect();

        // let shape = real_f32.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), complex_data).unwrap();
        // NdArrayTensor::new(array.into_shared())
        todo!()
    }

    fn complex_from_polar(
        magnitude: <Self as Backend>::FloatTensorPrimitive,
        phase: <Self as Backend>::FloatTensorPrimitive,
    ) -> NdArrayTensor {
        // Extract magnitude and phase as f32 tensors
        // let mag_f32 = match magnitude {
        //     NdArrayTensorFloat::F32(tensor) => tensor,
        //     NdArrayTensorFloat::F64(tensor) => {
        //         let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
        //         let shape = tensor.shape();
        //         let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
        //         NdArrayTensor::new(array.into_shared())
        //     }
        // };

        // let phase_f32 = match phase {
        //     NdArrayTensorFloat::F32(tensor) => tensor,
        //     NdArrayTensorFloat::F64(tensor) => {
        //         let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
        //         let shape = tensor.shape();
        //         let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
        //         NdArrayTensor::new(array.into_shared())
        //     }
        // };

        // let complex_data: Vec<Complex32> = mag_f32
        //     .array
        //     .iter()
        //     .zip(phase_f32.array.iter())
        //     .map(|(&m, &p)| Complex32 {
        //         real: m * p.cos(),
        //         imag: m * p.sin(),
        //     })
        //     .collect();

        // let shape = mag_f32.shape();
        // let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), complex_data).unwrap();
        // NdArrayTensor::new(array.into_shared())
        todo!()
    }

    fn complex_exp(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| {
        //     let exp_real = c.real.exp();
        //     Complex32 {
        //         real: exp_real * c.imag.cos(),
        //         imag: exp_real * c.imag.sin(),
        //     }
        // });
        todo!() //NdArrayTensor::new(array.into())
    }

    fn complex_log(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| {
        //     let magnitude = (c.real * c.real + c.imag * c.imag).sqrt();
        //     let phase = c.imag.atan2(c.real);
        //     Complex32 {
        //         real: magnitude.ln(),
        //         imag: phase,
        //     }
        // });
        todo!() //NdArrayTensor::new(array.into())
    }

    fn complex_powc(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        // a^b = exp(b * ln(a))
        let ln_lhs = Self::complex_log(lhs);
        let product = Self::complex_mul(rhs, ln_lhs);
        Self::complex_exp(product)
    }

    fn complex_sqrt(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| {
        //     let magnitude = (c.real * c.real + c.imag * c.imag).sqrt();
        //     let phase = c.imag.atan2(c.real);
        //     let sqrt_mag = magnitude.sqrt();
        //     let half_phase = phase / 2.0;
        //     Complex32 {
        //         real: sqrt_mag * half_phase.cos(),
        //         imag: sqrt_mag * half_phase.sin(),
        //     }
        // });
        todo!() //NdArrayTensor::new(array.into())
    }

    fn complex_sin(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| {
        //     // sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        //     Complex32 {
        //         real: c.real.sin() * c.imag.cosh(),
        //         imag: c.real.cos() * c.imag.sinh(),
        //     }
        // });
        todo!() //NdArrayTensor::new(array.into())
    }

    fn complex_cos(tensor: NdArrayTensor) -> NdArrayTensor {
        // let array = tensor.array.mapv(|c| {
        //     // cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        //     Complex32 {
        //         real: c.real.cos() * c.imag.cosh(),
        //         imag: -c.real.sin() * c.imag.sinh(),
        //     }
        // });
        todo!() //NdArrayTensor::new(array.into())
    }

    fn complex_tan(tensor: NdArrayTensor) -> NdArrayTensor {
        // tan(z) = sin(z) / cos(z)
        let sin_z = Self::complex_sin(tensor.clone());
        let cos_z = Self::complex_cos(tensor);
        Self::complex_div(sin_z, cos_z)
    }

    fn select(
        tensor: NdArrayTensor,
        dim: usize,
        indices: burn_tensor::Tensor<NdArray<E, I, Q>, 1, burn_tensor::Int>,
    ) -> NdArrayTensor {
        todo!()
    }

    fn select_assign(
        tensor: NdArrayTensor,
        dim: usize,
        indices: burn_tensor::Tensor<NdArray<E, I, Q>, 1, burn_tensor::Int>,
        values: NdArrayTensor,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_slice(tensor: NdArrayTensor, slices: &[burn_tensor::Slice]) -> NdArrayTensor {
        todo!()
    }

    fn complex_slice_assign(
        tensor: NdArrayTensor,
        ranges: &[burn_tensor::Slice],
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_swap_dims(tensor: NdArrayTensor, dim1: usize, dim2: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_repeat_dim(tensor: NdArrayTensor, dim: usize, times: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_not_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_cat(tensors: Vec<NdArrayTensor>, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_any(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_any_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_all(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_all_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_permute(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        todo!()
    }

    fn complex_expand(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        todo!()
    }

    fn complex_flip(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        todo!()
    }

    fn complex_unfold(
        tensor: NdArrayTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_select(
        tensor: NdArrayTensor,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<NdArray<E, I, Q>>,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_sum(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_sum_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_prod(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_prod_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_mean(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_mean_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_remainder(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_remainder_scalar(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_equal_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_not_equal_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_greater(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_greater_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_greater_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_greater_equal_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_lower(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_lower_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_lower_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_lower_equal_elem(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_mask_where(
        tensor: NdArrayTensor,
        mask: NdArrayTensor,
        source: NdArrayTensor,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_mask_fill(
        tensor: NdArrayTensor,
        mask: NdArrayTensor,
        value: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_gather(dim: usize, tensor: NdArrayTensor, indices: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_scatter(
        dim: usize,
        tensor: NdArrayTensor,
        indices: NdArrayTensor,
        values: NdArrayTensor,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_sign(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_argmax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_argmin(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_max(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_max_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_max_dim_with_indices(
        tensor: NdArrayTensor,
        dim: usize,
    ) -> (NdArrayTensor, NdArrayTensor) {
        todo!()
    }

    fn complex_max_abs(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_max_abs_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_min(tensor: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_min_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_min_dim_with_indices(
        tensor: NdArrayTensor,
        dim: usize,
    ) -> (NdArrayTensor, NdArrayTensor) {
        todo!()
    }

    fn complex_clamp(
        tensor: NdArrayTensor,
        min: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
        max: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_clamp_min(
        tensor: NdArrayTensor,
        min: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_clamp_max(
        tensor: NdArrayTensor,
        max: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_powi(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_powi_scalar(
        lhs: NdArrayTensor,
        rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        todo!()
    }

    fn complex_sort(tensor: NdArrayTensor, dim: usize, descending: bool) -> NdArrayTensor {
        todo!()
    }

    fn complex_sort_with_indices(
        tensor: NdArrayTensor,
        dim: usize,
        descending: bool,
    ) -> (NdArrayTensor, NdArrayTensor) {
        todo!()
    }

    fn complex_argsort(tensor: NdArrayTensor, dim: usize, descending: bool) -> NdArrayTensor {
        todo!()
    }

    fn complex_matmul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        todo!()
    }

    fn complex_cumsum(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_cumprod(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_cummin(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_cummax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        todo!()
    }

    fn complex_add(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_sub(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_real(tensor: ComplexTensor<NdArray<E, I, Q>>) -> NdArrayTensor {
        todo!()
    }

    fn complex_imag(tensor: ComplexTensor<NdArray<E, I, Q>>) -> NdArrayTensor {
        todo!()
    }
}
