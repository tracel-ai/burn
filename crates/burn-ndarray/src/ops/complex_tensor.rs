use crate::{
    IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SEED, backend::NdArrayDevice,
    element::FloatNdArrayElement,
};
use alloc::vec::Vec;
use burn_common::rand::get_seeded_rng;
use burn_tensor::{
    Complex32, Distribution, Shape, TensorData, TensorMetadata, backend::Backend,
    ops::ComplexTensorOps,
};
use ndarray::{ArrayD, IxDyn};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement>
    ComplexTensorOps<NdArray<E, I, Q>> for NdArray<E, I, Q>
{
    fn complex_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<Complex32> {
        NdArrayTensor::from_data(data)
    }

    fn complex_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<Complex32> {
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

    fn complex_shape(tensor: &NdArrayTensor<Complex32>) -> Shape {
        tensor.shape()
    }

    fn complex_to_data(tensor: &NdArrayTensor<Complex32>) -> TensorData {
        let shape = tensor.shape();
        let vec: Vec<Complex32> = tensor.array.iter().cloned().collect();
        TensorData::new(vec, shape)
    }

    fn complex_device(_tensor: &NdArrayTensor<Complex32>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn complex_to_device(
        tensor: NdArrayTensor<Complex32>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<Complex32> {
        tensor
    }

    fn complex_into_data(tensor: NdArrayTensor<Complex32>) -> TensorData {
        tensor.into_data()
    }

    fn complex_reshape(tensor: NdArrayTensor<Complex32>, shape: Shape) -> NdArrayTensor<Complex32> {
        let array = tensor
            .array
            .to_shape(IxDyn(shape.dims.as_slice()))
            .unwrap()
            .to_owned();
        NdArrayTensor::new(array.into())
    }

    fn complex_transpose(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.t().to_owned();
        NdArrayTensor::new(array.into_shared())
    }

    fn complex_add(
        lhs: NdArrayTensor<Complex32>,
        rhs: NdArrayTensor<Complex32>,
    ) -> NdArrayTensor<Complex32> {
        let array = &lhs.array + &rhs.array;
        NdArrayTensor::new(array.into())
    }

    fn complex_sub(
        lhs: NdArrayTensor<Complex32>,
        rhs: NdArrayTensor<Complex32>,
    ) -> NdArrayTensor<Complex32> {
        let array = &lhs.array - &rhs.array;
        NdArrayTensor::new(array.into())
    }

    fn complex_mul(
        lhs: NdArrayTensor<Complex32>,
        rhs: NdArrayTensor<Complex32>,
    ) -> NdArrayTensor<Complex32> {
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        let result = lhs
            .array
            .iter()
            .zip(rhs.array.iter())
            .map(|(a, b)| Complex32 {
                real: a.real * b.real - a.imag * b.imag,
                imag: a.real * b.imag + a.imag * b.real,
            })
            .collect::<Vec<_>>();

        let shape = lhs.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), result).unwrap();
        NdArrayTensor::new(array.into_shared())
    }

    fn complex_div(
        lhs: NdArrayTensor<Complex32>,
        rhs: NdArrayTensor<Complex32>,
    ) -> NdArrayTensor<Complex32> {
        // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
        let result = lhs
            .array
            .iter()
            .zip(rhs.array.iter())
            .map(|(a, b)| {
                let denom = b.real * b.real + b.imag * b.imag;
                Complex32 {
                    real: (a.real * b.real + a.imag * b.imag) / denom,
                    imag: (a.imag * b.real - a.real * b.imag) / denom,
                }
            })
            .collect::<Vec<_>>();

        let shape = lhs.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), result).unwrap();
        NdArrayTensor::new(array.into_shared())
    }

    fn complex_neg(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| Complex32 {
            real: -c.real,
            imag: -c.imag,
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_conj(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| Complex32 {
            real: c.real,
            imag: -c.imag,
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_real(tensor: NdArrayTensor<Complex32>) -> <Self as Backend>::FloatTensorPrimitive {
        let real_data: Vec<f32> = tensor.array.iter().map(|c| c.real).collect();
        let shape = tensor.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), real_data).unwrap();
        let real_tensor = NdArrayTensor::new(array.into_shared());
        real_tensor.into()
    }

    fn complex_imag(tensor: NdArrayTensor<Complex32>) -> <Self as Backend>::FloatTensorPrimitive {
        let imag_data: Vec<f32> = tensor.array.iter().map(|c| c.imag).collect();
        let shape = tensor.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), imag_data).unwrap();
        let imag_tensor = NdArrayTensor::new(array.into_shared());
        imag_tensor.into()
    }

    fn complex_abs(tensor: NdArrayTensor<Complex32>) -> <Self as Backend>::FloatTensorPrimitive {
        let abs_data: Vec<f32> = tensor
            .array
            .iter()
            .map(|c| (c.real * c.real + c.imag * c.imag).sqrt())
            .collect();
        let shape = tensor.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), abs_data).unwrap();
        let abs_tensor = NdArrayTensor::new(array.into_shared());
        abs_tensor.into()
    }

    fn complex_arg(tensor: NdArrayTensor<Complex32>) -> <Self as Backend>::FloatTensorPrimitive {
        let arg_data: Vec<f32> = tensor.array.iter().map(|c| c.imag.atan2(c.real)).collect();
        let shape = tensor.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), arg_data).unwrap();
        let arg_tensor = NdArrayTensor::new(array.into_shared());
        arg_tensor.into()
    }

    fn complex_from_parts(
        real: <Self as Backend>::FloatTensorPrimitive,
        imag: <Self as Backend>::FloatTensorPrimitive,
    ) -> NdArrayTensor<Complex32> {
        // Extract real and imaginary parts as f32 tensors
        let real_f32 = match real {
            crate::NdArrayTensorFloat::F32(tensor) => tensor,
            crate::NdArrayTensorFloat::F64(tensor) => {
                let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
                let shape = tensor.shape();
                let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
                NdArrayTensor::new(array.into_shared())
            }
        };

        let imag_f32 = match imag {
            crate::NdArrayTensorFloat::F32(tensor) => tensor,
            crate::NdArrayTensorFloat::F64(tensor) => {
                let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
                let shape = tensor.shape();
                let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
                NdArrayTensor::new(array.into_shared())
            }
        };

        let complex_data: Vec<Complex32> = real_f32
            .array
            .iter()
            .zip(imag_f32.array.iter())
            .map(|(&r, &i)| Complex32 { real: r, imag: i })
            .collect();

        let shape = real_f32.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), complex_data).unwrap();
        NdArrayTensor::new(array.into_shared())
    }

    fn complex_from_polar(
        magnitude: <Self as Backend>::FloatTensorPrimitive,
        phase: <Self as Backend>::FloatTensorPrimitive,
    ) -> NdArrayTensor<Complex32> {
        // Extract magnitude and phase as f32 tensors
        let mag_f32 = match magnitude {
            crate::NdArrayTensorFloat::F32(tensor) => tensor,
            crate::NdArrayTensorFloat::F64(tensor) => {
                let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
                let shape = tensor.shape();
                let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
                NdArrayTensor::new(array.into_shared())
            }
        };

        let phase_f32 = match phase {
            crate::NdArrayTensorFloat::F32(tensor) => tensor,
            crate::NdArrayTensorFloat::F64(tensor) => {
                let f32_data: Vec<f32> = tensor.array.iter().map(|&x| x as f32).collect();
                let shape = tensor.shape();
                let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), f32_data).unwrap();
                NdArrayTensor::new(array.into_shared())
            }
        };

        let complex_data: Vec<Complex32> = mag_f32
            .array
            .iter()
            .zip(phase_f32.array.iter())
            .map(|(&m, &p)| Complex32 {
                real: m * p.cos(),
                imag: m * p.sin(),
            })
            .collect();

        let shape = mag_f32.shape();
        let array = ArrayD::from_shape_vec(IxDyn(shape.dims.as_slice()), complex_data).unwrap();
        NdArrayTensor::new(array.into_shared())
    }

    fn complex_exp(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| {
            let exp_real = c.real.exp();
            Complex32 {
                real: exp_real * c.imag.cos(),
                imag: exp_real * c.imag.sin(),
            }
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_log(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| {
            let magnitude = (c.real * c.real + c.imag * c.imag).sqrt();
            let phase = c.imag.atan2(c.real);
            Complex32 {
                real: magnitude.ln(),
                imag: phase,
            }
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_powc(
        lhs: NdArrayTensor<Complex32>,
        rhs: NdArrayTensor<Complex32>,
    ) -> NdArrayTensor<Complex32> {
        // a^b = exp(b * ln(a))
        let ln_lhs = Self::complex_log(lhs);
        let product = Self::complex_mul(rhs, ln_lhs);
        Self::complex_exp(product)
    }

    fn complex_sqrt(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| {
            let magnitude = (c.real * c.real + c.imag * c.imag).sqrt();
            let phase = c.imag.atan2(c.real);
            let sqrt_mag = magnitude.sqrt();
            let half_phase = phase / 2.0;
            Complex32 {
                real: sqrt_mag * half_phase.cos(),
                imag: sqrt_mag * half_phase.sin(),
            }
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_sin(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| {
            // sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
            Complex32 {
                real: c.real.sin() * c.imag.cosh(),
                imag: c.real.cos() * c.imag.sinh(),
            }
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_cos(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        let array = tensor.array.mapv(|c| {
            // cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
            Complex32 {
                real: c.real.cos() * c.imag.cosh(),
                imag: -c.real.sin() * c.imag.sinh(),
            }
        });
        NdArrayTensor::new(array.into())
    }

    fn complex_tan(tensor: NdArrayTensor<Complex32>) -> NdArrayTensor<Complex32> {
        // tan(z) = sin(z) / cos(z)
        let sin_z = Self::complex_sin(tensor.clone());
        let cos_z = Self::complex_cos(tensor);
        Self::complex_div(sin_z, cos_z)
    }
}
