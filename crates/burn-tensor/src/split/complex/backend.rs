use crate::split::{
    base::{SplitBackend, SplitPrimitive},
    complex::SplitComplexLayout,
};
use burn_backend::{
    Backend, BackendTypes, ComplexDType, ComplexTensor, ComplexTensorBackend, ExecutionError,
    TensorMetadata, ops::ComplexTensorOps,
};
use burn_std::{
    ComplexScalar, Distribution, IndexingUpdateOp, Scalar, Shape, SplitTensorData, TensorData,
    cast::ToElement, complex_utils::complex_to_real_dtype,
};

impl<B> ComplexTensorBackend for SplitBackend<B>
where
    B: Backend,
    Self: BackendTypes<
            ComplexTensorPrimitive = SplitPrimitive<B::FloatTensorPrimitive, 2>,
            BoolTensorPrimitive = B::BoolTensorPrimitive,
            FloatTensorPrimitive = B::FloatTensorPrimitive,
            IntTensorPrimitive = B::IntTensorPrimitive,
            QuantizedTensorPrimitive = B::QuantizedTensorPrimitive,
            Device = B::Device,
        >,
{
    type InnerBackend = B;
    type Layout = SplitComplexLayout<B>;

    fn complex_from_real_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        let dims = data.shape.clone();
        let dtype = data.dtype.into();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_from_data(data, device),
            B::float_zeros(dims, device, dtype),
        ])
    }

    fn complex_from_imag_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        let dims = data.shape.clone();
        let dtype = data.dtype.into();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_zeros(dims, device, dtype),
            B::float_from_data(data, device),
        ])
    }
    // Should these be a result
    fn complex_from_interleaved_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        let data = burn_std::complex_utils::split_from_interleaved_data(data);
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_from_data(data.0, device),
            B::float_from_data(data.1, device),
        ])
    }

    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        let real = B::float_from_data(real_data, device);
        let imag = B::float_from_data(imag_data, device);
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([real, imag])
    }
}

impl<B> ComplexTensorOps<Self> for SplitBackend<B>
where
    B: Backend,
{
    // fn to_complex(tensor: B::FloatTensorPrimitive) -> ComplexTensor<Self> {
    //     let shape = tensor.shape().clone();
    //     let dtype = tensor.dtype().into();
    //     let device = &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor);
    //     ComplexTensor::<Self>{0: [tensor, B::float_zeros(shape, device, dtype)]}
    // }

    fn complex_real(tensor: ComplexTensor<Self>) -> B::FloatTensorPrimitive {
        let [real, _imag] = tensor.0;
        real
    }
    fn complex_imag(tensor: ComplexTensor<Self>) -> B::FloatTensorPrimitive {
        let [_real, imag] = tensor.0;
        imag
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [lhs_re, lhs_im] = lhs.0;
        let ComplexScalar::<f64> {
            real: rhs_re,
            imag: rhs_im,
        } = rhs.to_complex().elem::<ComplexScalar<f64>>();

        let real_cmp = B::float_not_equal_elem(lhs_re, Scalar::Float(rhs_re.to_f64()), out_dtype);
        let imag_cmp = B::float_not_equal_elem(lhs_im, Scalar::Float(rhs_im.to_f64()), out_dtype);
        B::bool_or(real_cmp, imag_cmp)
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [lhs_re, lhs_im] = lhs.0;
        let ComplexScalar::<f64> {
            real: rhs_re,
            imag: rhs_im,
        } = rhs.to_complex().elem::<ComplexScalar<f64>>();

        let real_cmp = B::float_equal_elem(lhs_re, Scalar::Float(rhs_re.to_f64()), out_dtype);
        let imag_cmp = B::float_equal_elem(lhs_im, Scalar::Float(rhs_im.to_f64()), out_dtype);
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        let real_cmp = B::float_equal(lhs_re, rhs_re, out_dtype);
        let imag_cmp = B::float_equal(lhs_im, rhs_im, out_dtype);
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_not_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        let real_cmp = B::float_not_equal(lhs_re, rhs_re, out_dtype);
        let imag_cmp = B::float_not_equal(lhs_im, rhs_im, out_dtype);
        B::bool_or(real_cmp, imag_cmp)
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<TensorData, ExecutionError> {
        let [real, _imag] = tensor.0;
        B::float_into_data(real).await
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<TensorData, ExecutionError> {
        let [_real, imag] = tensor.0;
        B::float_into_data(imag).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<TensorData, ExecutionError> {
        let [real, imag] = tensor.0;
        let real_data = B::float_into_data(real).await?;
        let imag_data = B::float_into_data(imag).await?;
        let element_size = real_data.dtype.size();
        let mut interleaved_bytes = Vec::with_capacity(real_data.bytes.len() * 2);
        for (real_chunk, imag_chunk) in real_data
            .bytes
            .chunks_exact(element_size)
            .zip(imag_data.bytes.chunks_exact(element_size))
        {
            interleaved_bytes.extend_from_slice(real_chunk);
            interleaved_bytes.extend_from_slice(imag_chunk);
        }
        Ok(TensorData::from_bytes_vec(
            interleaved_bytes,
            real_data.shape,
            burn_std::complex_utils::real_to_complex_dtype(real_data.dtype),
        ))
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<SplitTensorData, ExecutionError> {
        let [real, imag] = tensor.0;
        let real_data = B::float_into_data(real).await?;
        let imag_data = B::float_into_data(imag).await?;
        Ok((real_data, imag_data))
    }

    fn complex_device(tensor: &ComplexTensor<Self>) -> B::Device {
        let [real, _imag] = &tensor.0;
        B::float_device(real)
    }

    fn complex_add(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        ComplexTensor::<Self> {
            0: [B::float_add(lhs_re, rhs_re), B::float_add(lhs_im, rhs_im)],
        }
    }

    fn complex_sub(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        ComplexTensor::<Self> {
            0: [B::float_sub(lhs_re, rhs_re), B::float_sub(lhs_im, rhs_im)],
        }
    }

    fn complex_mul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        ComplexTensor::<Self> {
            0: [
                B::float_sub(
                    B::float_mul(lhs_re.clone(), rhs_re.clone()),
                    B::float_mul(lhs_im.clone(), rhs_im.clone()),
                ),
                B::float_add(B::float_mul(lhs_re, rhs_im), B::float_mul(rhs_re, lhs_im)),
            ],
        }
    }

    fn complex_div(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]

        let norm_sqr = SplitBackend::<B>::complex_squared_norm(rhs.clone());
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        ComplexTensor::<Self> {
            0: [
                B::float_div(
                    B::float_add(
                        B::float_mul(lhs_re.clone(), rhs_re.clone()),
                        B::float_mul(lhs_im.clone(), rhs_im.clone()),
                    ),
                    norm_sqr.clone(),
                ),
                B::float_div(
                    B::float_sub(B::float_mul(lhs_im, rhs_re), B::float_mul(lhs_re, rhs_im)),
                    norm_sqr,
                ),
            ],
        }
    }
    fn complex_abs(tensor: ComplexTensor<Self>) -> B::FloatTensorPrimitive {
        //todo! https://github.com/tracel-ai/burn/issues/4836
        B::float_sqrt(SplitBackend::<B>::complex_squared_norm(tensor))
    }

    fn complex_from_parts(
        real: TensorData,
        imag: TensorData,
        _device: &<B as BackendTypes>::Device,
    ) -> ComplexTensor<Self> {
        ComplexTensor::<Self> {
            0: [
                B::float_from_data(real, &Default::default()),
                B::float_from_data(imag, &Default::default()),
            ],
        }
    }

    fn complex_exp(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let [real, imag] = tensor.0;
        let exp_real = B::float_exp(real.clone());
        let cos_imag = B::float_cos(imag.clone());
        let sin_imag = B::float_sin(imag);

        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mul(exp_real.clone(), cos_imag),
            B::float_mul(exp_real, sin_imag),
        ])
    }

    fn complex_log(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let [real, imag] = tensor.0;
        let real_sq = B::float_mul(real.clone(), real.clone());
        let imag_sq = B::float_mul(imag.clone(), imag.clone());
        let norm_sq = B::float_add(real_sq, imag_sq);
        let norm = B::float_sqrt(norm_sq);

        // Compute arg: atan2(imag, real)
        let arg = B::float_atan2(imag, real);

        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_log(norm),
            arg,
        ])
    }

    fn complex_squared_norm(tensor: ComplexTensor<Self>) -> B::FloatTensorPrimitive {
        let [real, imag] = tensor.0;
        let real_sq = B::float_mul(real.clone(), real);
        let imag_sq = B::float_mul(imag.clone(), imag);
        B::float_add(real_sq, imag_sq)
    }

    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<Self> {
        ComplexTensor::<Self> {
            0: [
                B::float_mul(magnitude.clone(), B::float_cos(phase.clone())),
                B::float_mul(magnitude, B::float_sin(phase)),
            ],
        }
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        ComplexTensor::<Self> {
            0: [
                B::float_gather(dim, real, indices.clone()),
                B::float_gather(dim, imag, indices),
            ],
        }
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let [values_real, values_imag] = values.0;
        ComplexTensor::<Self> {
            0: [
                B::float_scatter_add(dim, real, indices.clone(), values_real),
                B::float_scatter_add(dim, imag, indices, values_imag),
            ],
        }
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: Distribution,
        device: &<B as BackendTypes>::Device,
        dtype: ComplexDType,
    ) -> ComplexTensor<Self> {
        let dtype = complex_to_real_dtype(dtype.into()).into();
        ComplexTensor::<Self> {
            0: [
                B::float_random(shape.clone(), distribution, device, dtype),
                B::float_random(shape, distribution, device, dtype),
            ],
        }
    }

    fn complex_to_device(
        tensor: ComplexTensor<Self>,
        device: &<B as BackendTypes>::Device,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_to_device(real, device),
            B::float_to_device(imag, device),
        ])
    }

    fn complex_reshape(tensor: ComplexTensor<Self>, shape: burn_std::Shape) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_reshape(real, shape.clone()),
            B::float_reshape(imag, shape),
        ])
    }

    fn complex_transpose(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_transpose(real),
            B::float_transpose(imag),
        ])
    }

    fn complex_neg(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_neg(real),
            B::float_neg(imag),
        ])
    }

    fn complex_conj(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // conj(a + bi) = a - bi
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            real,
            B::float_neg(imag),
        ])
    }

    fn complex_arg(tensor: ComplexTensor<Self>) -> B::FloatTensorPrimitive {
        // arg(a + bi) = atan2(b, a)
        let [real, imag] = tensor.0;
        B::float_atan2(imag, real)
    }

    fn complex_powc(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // z^w = exp(w * ln(z))
        let log_lhs = SplitBackend::<B>::complex_log(lhs);
        let product = SplitBackend::<B>::complex_mul(rhs, log_lhs);
        SplitBackend::<B>::complex_exp(product)
    }

    fn complex_sqrt(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // sqrt(z) = from_polar(sqrt(|z|), arg(z) / 2)
        let abs = SplitBackend::<B>::complex_abs(tensor.clone());
        let sqrt_abs = B::float_sqrt(abs);
        let [real, imag] = tensor.0;
        let arg = B::float_atan2(imag, real);
        let half_arg = B::float_div_scalar(arg, Scalar::Float(2.0));
        SplitBackend::<B>::complex_from_polar(sqrt_abs, half_arg)
    }

    fn complex_sin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mul(B::float_sin(real.clone()), B::float_cosh(imag.clone())),
            B::float_mul(B::float_cos(real), B::float_sinh(imag)),
        ])
    }

    fn complex_cos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mul(B::float_cos(real.clone()), B::float_cosh(imag.clone())),
            B::float_neg(B::float_mul(B::float_sin(real), B::float_sinh(imag))),
        ])
    }

    fn complex_tan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // tan(z) = sin(z) / cos(z)
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        // Compute sin(a), cos(a), sinh(b), cosh(b) once and share between numerator/denominator.
        let [real, imag] = tensor.0;
        let sin_a = B::float_sin(real.clone());
        let cos_a = B::float_cos(real);
        let sinh_b = B::float_sinh(imag.clone());
        let cosh_b = B::float_cosh(imag);
        let sin_z = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mul(sin_a.clone(), cosh_b.clone()),
            B::float_mul(cos_a.clone(), sinh_b.clone()),
        ]);
        let cos_z = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mul(cos_a, cosh_b),
            B::float_neg(B::float_mul(sin_a, sinh_b)),
        ]);
        SplitBackend::<B>::complex_div(sin_z, cos_z)
    }

    fn complex_acos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // acos(z) = -i * ln(z + i * sqrt(1 - z²))
        let real = &tensor.0[0];
        let device = B::float_device(real);
        let shape = real.shape().clone();
        let fdtype = real.dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        // 1 - z²
        let z_sq = SplitBackend::<B>::complex_mul(tensor.clone(), tensor.clone());
        let one_minus_z_sq = SplitBackend::<B>::complex_sub(ones, z_sq);
        // i * sqrt(1 - z²): multiply by i via (-imag, real)
        let [sqrt_real, sqrt_imag] = SplitBackend::<B>::complex_sqrt(one_minus_z_sq).0;
        let i_sqrt =
            SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
                B::float_neg(sqrt_imag),
                sqrt_real,
            ]);
        // z + i*sqrt(1 - z²)
        let inner = SplitBackend::<B>::complex_add(tensor, i_sqrt);
        // -i * ln(inner): multiply by -i via (imag, -real)
        let [log_real, log_imag] = SplitBackend::<B>::complex_log(inner).0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            log_imag,
            B::float_neg(log_real),
        ])
    }

    fn complex_acosh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // acosh(z) = ln(z + sqrt(z² - 1))
        let real = &tensor.0[0];
        let device = B::float_device(real);
        let shape = real.shape().clone();
        let fdtype = real.dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        // z² - 1
        let z_sq = SplitBackend::<B>::complex_mul(tensor.clone(), tensor.clone());
        let z_sq_minus_one = SplitBackend::<B>::complex_sub(z_sq, ones);
        // z + sqrt(z² - 1)
        let sqrt_term = SplitBackend::<B>::complex_sqrt(z_sq_minus_one);
        let inner = SplitBackend::<B>::complex_add(tensor, sqrt_term);
        SplitBackend::<B>::complex_log(inner)
    }

    fn complex_asin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // asin(z) = -i * ln(i*z + sqrt(1 - z²))
        let real = &tensor.0[0];
        let device = B::float_device(real);
        let shape = real.shape().clone();
        let fdtype = real.dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        // z² and i*z before consuming tensor
        let z_sq = SplitBackend::<B>::complex_mul(tensor.clone(), tensor.clone());
        // i*z = (-imag, real)
        let [real, imag] = tensor.0;
        let i_z = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_neg(imag),
            real,
        ]);
        // 1 - z²
        let one_minus_z_sq = SplitBackend::<B>::complex_sub(ones, z_sq);
        // i*z + sqrt(1 - z²)
        let sqrt_term = SplitBackend::<B>::complex_sqrt(one_minus_z_sq);
        let inner = SplitBackend::<B>::complex_add(i_z, sqrt_term);
        // -i * ln(inner): (imag, -real)
        let log_inner = SplitBackend::<B>::complex_log(inner);
        let [real, imag] = log_inner.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            imag,
            B::float_neg(real),
        ])
    }

    fn complex_asinh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // asinh(z) = ln(z + sqrt(z² + 1))
        let real = &tensor.0[0];
        let device = B::float_device(real);
        let shape = real.shape().clone();
        let fdtype = real.dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        // z² + 1
        let z_sq = SplitBackend::<B>::complex_mul(tensor.clone(), tensor.clone());
        let z_sq_plus_one = SplitBackend::<B>::complex_add(z_sq, ones);
        // z + sqrt(z² + 1)
        let sqrt_term = SplitBackend::<B>::complex_sqrt(z_sq_plus_one);
        let inner = SplitBackend::<B>::complex_add(tensor, sqrt_term);
        SplitBackend::<B>::complex_log(inner)
    }

    fn complex_atan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // atan(z) = (-i/2) * ln((1 + i*z) / (1 - i*z))
        let [real, imag] = tensor.0;
        let device = B::float_device(&real);
        let shape = real.shape().clone();
        let fdtype = real.dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        // i*z = (-imag, real)
        let i_z = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_neg(imag),
            real,
        ]);
        // 1 + i*z and 1 - i*z
        let one_plus_i_z = SplitBackend::<B>::complex_add(ones.clone(), i_z.clone());
        let one_minus_i_z = SplitBackend::<B>::complex_sub(ones, i_z);
        // ln((1 + i*z) / (1 - i*z))
        let log_ratio = SplitBackend::<B>::complex_log(SplitBackend::<B>::complex_div(
            one_plus_i_z,
            one_minus_i_z,
        ));
        let [log_ratio_real, log_ratio_imag] = log_ratio.0;
        // (-i/2) * log_ratio: -i*(a+bi) = (b, -a), then /2
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_div_scalar(log_ratio_imag, Scalar::Float(2.0)),
            B::float_neg(B::float_div_scalar(log_ratio_real, Scalar::Float(2.0))),
        ])
    }

    fn complex_atanh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))

        let device = B::float_device(&tensor.0[0]);
        let shape = tensor.0[0].shape().clone();
        let fdtype = tensor.0[0].dtype().into();
        let ones = SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        ]);
        let one_plus_z = SplitBackend::<B>::complex_add(ones.clone(), tensor.clone());
        let one_minus_z = SplitBackend::<B>::complex_sub(ones, tensor);
        let log_ratio =
            SplitBackend::<B>::complex_log(SplitBackend::<B>::complex_div(one_plus_z, one_minus_z));
        let [log_ratio_real, log_ratio_imag] = log_ratio.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_div_scalar(log_ratio_real, Scalar::Float(2.0)),
            B::float_div_scalar(log_ratio_imag, Scalar::Float(2.0)),
        ])
    }

    fn complex_slice(tensor: ComplexTensor<Self>, slices: &[crate::Slice]) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_slice(real, slices),
            B::float_slice(imag, slices),
        ])
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<Self>,
        ranges: &[crate::Slice],
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let [value_real, value_imag] = value.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_slice_assign(real, ranges, value_real),
            B::float_slice_assign(imag, ranges, value_imag),
        ])
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_swap_dims(real, dim1, dim2),
            B::float_swap_dims(imag, dim1, dim2),
        ])
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_repeat_dim(real, dim, times),
            B::float_repeat_dim(imag, dim, times),
        ])
    }

    fn complex_cat(tensors: Vec<ComplexTensor<Self>>, dim: usize) -> ComplexTensor<Self> {
        let (reals, imags): (Vec<_>, Vec<_>) = tensors
            .into_iter()
            .map(|t| {
                let [real, imag] = t.0;
                (real, imag)
            })
            .unzip();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_cat(reals, dim),
            B::float_cat(imags, dim),
        ])
    }

    fn complex_any(
        tensor: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [real, imag] = tensor.0;
        let real_any = B::float_any(real, out_dtype);
        let imag_any = B::float_any(imag, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [real, imag] = tensor.0;
        let real_any = B::float_any_dim(real, dim, out_dtype);
        let imag_any = B::float_any_dim(imag, dim, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_all(
        tensor: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        // A complex element is nonzero if either real or imag is nonzero.
        // all(z != 0) = all(real != 0 || imag != 0)
        let [real, imag] = tensor.0;
        let real_nonzero = B::float_not_equal_elem(real, Scalar::Float(0.0), out_dtype);
        let imag_nonzero = B::float_not_equal_elem(imag, Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all(elem_nonzero)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let [real, imag] = tensor.0;
        let real_nonzero = B::float_not_equal_elem(real, Scalar::Float(0.0), out_dtype);
        let imag_nonzero = B::float_not_equal_elem(imag, Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all_dim(elem_nonzero, dim)
    }

    fn complex_permute(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_permute(real, axes),
            B::float_permute(imag, axes),
        ])
    }

    fn complex_expand(tensor: ComplexTensor<Self>, shape: burn_std::Shape) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_expand(real, shape.clone()),
            B::float_expand(imag, shape),
        ])
    }

    fn complex_flip(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_flip(real, axes),
            B::float_flip(imag, axes),
        ])
    }

    fn complex_unfold(
        tensor: ComplexTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_unfold(real, dim, size, step),
            B::float_unfold(imag, dim, size, step),
        ])
    }

    fn complex_select(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_select(real, dim, indices.clone()),
            B::float_select(imag, dim, indices),
        ])
    }

    fn complex_sum(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_sum(real),
            B::float_sum(imag),
        ])
    }

    fn complex_sum_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_sum_dim(real, dim),
            B::float_sum_dim(imag, dim),
        ])
    }

    fn complex_prod(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // prod(z) = exp(sum(log(z)))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let sum_log = SplitBackend::<B>::complex_sum(log_tensor);
        SplitBackend::<B>::complex_exp(sum_log)
    }

    fn complex_prod_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        // prod_dim(z, dim) = exp(sum_dim(log(z), dim))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let sum_log = SplitBackend::<B>::complex_sum_dim(log_tensor, dim);
        SplitBackend::<B>::complex_exp(sum_log)
    }

    fn complex_mean(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mean(real),
            B::float_mean(imag),
        ])
    }

    fn complex_mean_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mean_dim(real, dim),
            B::float_mean_dim(imag, dim),
        ])
    }

    fn complex_remainder(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        // Componentwise remainder (matching Complex<E> Rem impl)
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_remainder(lhs_re, rhs_re),
            B::float_remainder(lhs_im, rhs_im),
        ])
    }

    fn complex_remainder_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        let [lhs_re, lhs_im] = lhs.0;
        let ComplexScalar::<f64> {
            real: rhs_re,
            imag: rhs_im,
        } = rhs.to_complex().elem::<ComplexScalar<f64>>();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_remainder_scalar(lhs_re, Scalar::Float(rhs_re.to_f64())),
            B::float_remainder_scalar(lhs_im, Scalar::Float(rhs_im.to_f64())),
        ])
    }

    fn complex_mask_where(
        tensor: ComplexTensor<Self>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let [source_real, source_imag] = source.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mask_where(real, mask.clone(), source_real),
            B::float_mask_where(imag, mask, source_imag),
        ])
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<Self>,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let ComplexScalar::<f64> {
            real: value_re,
            imag: value_im,
        } = value.to_complex().elem::<ComplexScalar<f64>>();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_mask_fill(real, mask.clone(), Scalar::Float(value_re.to_f64())),
            B::float_mask_fill(imag, mask, Scalar::Float(value_im.to_f64())),
        ])
    }

    fn complex_sign(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = SplitBackend::<B>::complex_abs(tensor.clone());
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_div(real, abs.clone()),
            B::float_div(imag, abs),
        ])
    }

    fn complex_matmul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        let [lhs_re, lhs_im] = lhs.0;
        let [rhs_re, rhs_im] = rhs.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_sub(
                B::float_matmul(lhs_re.clone(), rhs_re.clone()),
                B::float_matmul(lhs_im.clone(), rhs_im.clone()),
            ),
            B::float_add(
                B::float_matmul(lhs_re, rhs_im),
                B::float_matmul(lhs_im, rhs_re),
            ),
        ])
    }

    fn complex_cumsum(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        // cumsum is linear, so it works componentwise
        let [real, imag] = tensor.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_cumsum(real, dim),
            B::float_cumsum(imag, dim),
        ])
    }

    fn complex_cumprod(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        // cumprod(z, dim) = exp(cumsum(log(z), dim))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let cumsum_log = SplitBackend::<B>::complex_cumsum(log_tensor, dim);
        SplitBackend::<B>::complex_exp(cumsum_log)
    }

    fn complex_select_add(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let [values_real, values_imag] = values.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_select_add(real, dim, indices.clone(), values_real),
            B::float_select_add(imag, dim, indices, values_imag),
        ])
    }

    fn complex_powc_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        // z^c = exp(c * ln(z)) where c = a + bi is a scalar
        // (a + bi) * (u + vi) = (au - bv) + (av + bu)i
        let ComplexScalar::<f64> { real: a, imag: b } =
            rhs.to_complex().elem::<ComplexScalar<f64>>();
        let ln_z = SplitBackend::<B>::complex_log(lhs);
        let [ln_z_re, ln_z_im] = ln_z.0;
        let c_ln_z =
            SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
                B::float_sub(
                    B::float_mul_scalar(ln_z_re.clone(), Scalar::Float(a)),
                    B::float_mul_scalar(ln_z_im.clone(), Scalar::Float(b)),
                ),
                B::float_add(
                    B::float_mul_scalar(ln_z_re, Scalar::Float(b)),
                    B::float_mul_scalar(ln_z_im, Scalar::Float(a)),
                ),
            ]);
        SplitBackend::<B>::complex_exp(c_ln_z)
    }

    fn complex_powf(lhs: ComplexTensor<Self>, rhs: B::FloatTensorPrimitive) -> ComplexTensor<Self> {
        // z^w = exp(w * ln(z)) where w is a real tensor
        let log_z = SplitBackend::<B>::complex_log(lhs);
        let [log_z_re, log_z_im] = log_z.0;
        let w_log_z =
            SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
                B::float_mul(rhs.clone(), log_z_re),
                B::float_mul(rhs, log_z_im),
            ]);
        SplitBackend::<B>::complex_exp(w_log_z)
    }

    fn complex_powf_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        // z^w = exp(w * ln(z)) where w is a real scalar
        let [log_z_re, log_z_im] = SplitBackend::<B>::complex_log(lhs).0;
        let w_log_z =
            SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
                B::float_mul_scalar(log_z_re, rhs),
                B::float_mul_scalar(log_z_im, rhs),
            ]);
        SplitBackend::<B>::complex_exp(w_log_z)
    }

    fn complex_scatter_nd(
        tensor: ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<Self>,
        reduction: IndexingUpdateOp,
    ) -> ComplexTensor<Self> {
        let [real, imag] = tensor.0;
        let [values_real, values_imag] = values.0;
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_scatter_nd(real, indices.clone(), values_real, reduction),
            B::float_scatter_nd(imag, indices, values_imag, reduction),
        ])
    }

    fn complex_zeros(
        shape: Shape,
        device: &burn_backend::tensor::Device<SplitBackend<B>>,
        dtype: ComplexDType,
    ) -> ComplexTensor<SplitBackend<B>> {
        let dtype = burn_std::complex_utils::complex_to_real_dtype(dtype.into()).into();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_zeros(shape.clone(), device, dtype),
            B::float_zeros(shape, device, dtype),
        ])
    }

    fn complex_ones(
        shape: Shape,
        device: &burn_backend::tensor::Device<SplitBackend<B>>,
        dtype: ComplexDType,
    ) -> ComplexTensor<SplitBackend<B>> {
        let dtype = burn_std::complex_utils::complex_to_real_dtype(dtype.into()).into();
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_ones(shape.clone(), device, dtype),
            B::float_zeros(shape, device, dtype),
        ])
    }

    fn complex_full(
        shape: Shape,
        fill_value: Scalar,
        device: &burn_backend::tensor::Device<SplitBackend<B>>,
        dtype: ComplexDType,
    ) -> ComplexTensor<SplitBackend<B>> {
        let _fill_value = fill_value.to_complex();

        let dtype = burn_std::complex_utils::complex_to_real_dtype(dtype.into());
        SplitPrimitive::<<SplitBackend<B> as BackendTypes>::FloatTensorPrimitive, 2>([
            B::float_zeros(shape.clone(), device, dtype.into()),
            B::float_zeros(shape, device, dtype.into()),
        ])
    }

    async fn complex_into_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, ExecutionError> {
        let [real, imag] = tensor.0;
        Ok(burn_std::complex_utils::interleaved_data_from_parts_data(
            B::float_into_data(real).await?,
            B::float_into_data(imag).await?,
        ))
    }
}
