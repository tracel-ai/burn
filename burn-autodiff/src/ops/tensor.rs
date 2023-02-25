use crate::{
    grads::Gradients,
    graph::ops::{MetadataRef, Requirement},
    ops::{binary::BinaryOps, unary::UnaryOpsNoCapture},
    tensor::{clone_if_shared, ADTensor, BackwardTensor, BoolTensor, Elem, IntTensor},
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::TensorOps, Data, Shape};

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn from_data<const D: usize>(data: Data<Elem<B>, D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::from_data(data, device))
    }

    fn from_data_bool<const D: usize>(data: Data<bool, D>, device: &B::Device) -> BoolTensor<B, D> {
        B::from_data_bool(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: burn_tensor::Distribution<Elem<B>>,
        device: &B::Device,
    ) -> ADTensor<B, D> {
        ADTensor::new(B::random(shape, distribution, device))
    }

    fn shape<const D: usize>(tensor: &ADTensor<B, D>) -> Shape<D> {
        B::shape(&tensor.primitive)
    }

    fn to_data<const D: usize>(tensor: &ADTensor<B, D>) -> Data<Elem<B>, D> {
        B::to_data(&tensor.primitive)
    }

    fn into_data<const D: usize>(tensor: ADTensor<B, D>) -> Data<Elem<B>, D> {
        B::into_data(tensor.primitive)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_into_data(tensor)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D> {
        B::bool_into_int(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &B::Device,
    ) -> BoolTensor<B, D> {
        B::bool_to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<B, D1> {
        B::bool_index(tensor, indexes)
    }

    fn device<const D: usize>(tensor: &ADTensor<B, D>) -> B::Device {
        B::device(&tensor.primitive)
    }

    fn to_device<const D: usize>(tensor: ADTensor<B, D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::to_device(tensor.primitive, device))
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        ADTensor::new(B::empty(shape, device))
    }

    fn add<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Add;

        impl<B: Backend, const D: usize> BinaryOps<B, D> for Add {
            type BackwardState = ();

            fn forward(
                &self,
                lhs: B::TensorPrimitive<D>,
                rhs: B::TensorPrimitive<D>,
            ) -> B::TensorPrimitive<D> {
                B::add(lhs, rhs)
            }
            fn backward(
                self,
                lhs: Option<MetadataRef>,
                rhs: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _state: (),
            ) {
                let grad_output = grads.consume(&output);
                let (grad_output_lhs, grad_output_rhs) = clone_if_shared(&lhs, &rhs, grad_output);

                if let Some((lhs, grad)) = lhs.zip(grad_output_lhs) {
                    grads.update(lhs, grad)
                }
                if let Some((rhs, grad)) = rhs.zip(grad_output_rhs) {
                    grads.update(rhs, grad)
                }
            }
        }

        Add.execute(lhs, rhs, ())
    }

    fn add_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct AddScalar;

        impl<B: Backend, const D: usize> UnaryOpsNoCapture<B, D, D> for AddScalar {
            type StateForward = Elem<B>;
            type StateBackward = ();

            fn forward(&self, lhs: B::TensorPrimitive<D>, rhs: Elem<B>) -> B::TensorPrimitive<D> {
                B::add_scalar(lhs, rhs)
            }
            fn backward(
                self,
                tensor: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _rhs: (),
            ) {
                let grad_output = grads.consume(&output);

                if let Some(tensor) = tensor {
                    grads.update(tensor, grad_output)
                }
            }
        }

        AddScalar.execute(lhs, rhs, ())
    }

    fn sub<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Sub;

        impl<B: Backend, const D: usize> BinaryOps<B, D> for Sub {
            type BackwardState = ();

            fn forward(
                &self,
                lhs: B::TensorPrimitive<D>,
                rhs: B::TensorPrimitive<D>,
            ) -> B::TensorPrimitive<D> {
                B::sub(lhs, rhs)
            }
            fn backward(
                self,
                lhs: Option<MetadataRef>,
                rhs: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _state: (),
            ) {
                let grad_output = grads.consume(&output);
                let (grad_output_lhs, grad_output_rhs) = clone_if_shared(&lhs, &rhs, grad_output);

                if let Some((lhs, grad)) = lhs.zip(grad_output_lhs) {
                    grads.update(lhs, grad)
                }
                if let Some((rhs, grad)) = rhs.zip(grad_output_rhs) {
                    grads.update(rhs, B::neg(grad))
                }
            }
        }

        Sub.execute(lhs, rhs, ())
    }

    fn sub_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct SubScalar;

        impl<B: Backend, const D: usize> UnaryOpsNoCapture<B, D, D> for SubScalar {
            type StateForward = Elem<B>;
            type StateBackward = ();

            fn forward(&self, lhs: B::TensorPrimitive<D>, rhs: Elem<B>) -> B::TensorPrimitive<D> {
                B::sub_scalar(lhs, rhs)
            }
            fn backward(
                self,
                tensor: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                _rhs: (),
            ) {
                let grad_output = grads.consume(&output);

                if let Some(tensor) = tensor {
                    grads.update(tensor, grad_output)
                }
            }
        }

        SubScalar.execute(lhs, rhs, ())
    }

    fn mul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Mul;

        impl<B: Backend, const D: usize> BinaryOps<B, D> for Mul {
            type BackwardState = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn forward(
                &self,
                lhs: B::TensorPrimitive<D>,
                rhs: B::TensorPrimitive<D>,
            ) -> B::TensorPrimitive<D> {
                B::mul(lhs, rhs)
            }
            fn backward(
                self,
                lhs: Option<MetadataRef>,
                rhs: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (state_lhs, state_rhs): Self::BackwardState,
            ) {
                let grad_output = grads.consume(&output);
                let (grad_output_lhs, grad_output_rhs) = clone_if_shared(&lhs, &rhs, grad_output);

                if let Some((lhs, grad_output)) = lhs.zip(grad_output_lhs) {
                    let grad_lhs = B::mul(grad_output, state_rhs.unwrap());
                    grads.update(lhs, grad_lhs)
                }
                if let Some((rhs, grad_output)) = rhs.zip(grad_output_rhs) {
                    let grad_rhs = B::mul(grad_output, state_lhs.unwrap());
                    grads.update(rhs, grad_rhs)
                }
            }
        }

        let state_lhs = match rhs.metadata.requirement {
            Requirement::None => None,
            _ => Some(lhs.primitive.clone()),
        };
        let state_rhs = match lhs.metadata.requirement {
            Requirement::None => None,
            _ => Some(rhs.primitive.clone()),
        };
        Mul.execute(lhs, rhs, (state_lhs, state_rhs))
    }

    fn mul_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct MulScalar;

        impl<B: Backend, const D: usize> UnaryOpsNoCapture<B, D, D> for MulScalar {
            type StateForward = Elem<B>;
            type StateBackward = Elem<B>;

            fn forward(&self, lhs: B::TensorPrimitive<D>, rhs: Elem<B>) -> B::TensorPrimitive<D> {
                B::mul_scalar(lhs, rhs)
            }
            fn backward(
                self,
                tensor: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                rhs: Elem<B>,
            ) {
                let grad_output = grads.consume(&output);

                if let Some(tensor) = tensor {
                    let grad = B::mul_scalar(grad_output, rhs);
                    grads.update(tensor, grad)
                }
            }
        }

        MulScalar.execute(lhs, rhs, rhs)
    }

    fn div<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn div_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> ADTensor<B, D> {
        todo!()
    }

    fn matmul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Matmul;

        impl<B: Backend, const D: usize> BinaryOps<B, D> for Matmul {
            type BackwardState = (Option<B::TensorPrimitive<D>>, Option<B::TensorPrimitive<D>>);

            fn forward(
                &self,
                lhs: B::TensorPrimitive<D>,
                rhs: B::TensorPrimitive<D>,
            ) -> B::TensorPrimitive<D> {
                B::matmul(lhs, rhs)
            }
            fn backward(
                self,
                lhs: Option<MetadataRef>,
                rhs: Option<MetadataRef>,
                output: BackwardTensor<B, D>,
                grads: &mut Gradients<B>,
                (state_lhs, state_rhs): Self::BackwardState,
            ) {
                let grad_output = grads.consume(&output);
                let (grad_output_lhs, grad_output_rhs) = clone_if_shared(&lhs, &rhs, grad_output);

                if let Some(((lhs, grad_output), state_rhs)) =
                    lhs.zip(grad_output_lhs).zip(state_rhs)
                {
                    let rhs = B::transpose(state_rhs);
                    let grad_lhs = B::matmul(grad_output, rhs);
                    grads.update(lhs, grad_lhs)
                }
                if let Some(((rhs, grad_output), state_lhs)) =
                    rhs.zip(grad_output_rhs).zip(state_lhs)
                {
                    let lhs = B::transpose(state_lhs);
                    let grad_rhs = B::matmul(lhs, grad_output);
                    grads.update(rhs, grad_rhs)
                }
            }
        }

        let state_lhs = match rhs.metadata.requirement {
            Requirement::None => None,
            _ => Some(lhs.primitive.clone()),
        };
        let state_rhs = match lhs.metadata.requirement {
            Requirement::None => None,
            _ => Some(rhs.primitive.clone()),
        };

        Matmul.execute(lhs, rhs, (state_lhs, state_rhs))
    }

    fn neg<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        _tensor: ADTensor<B, D>,
        _dim1: usize,
        _dim2: usize,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _shape: Shape<D2>,
    ) -> ADTensor<B, D2> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _indexes: [std::ops::Range<usize>; D2],
    ) -> ADTensor<B, D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(
        _tensor: ADTensor<B, D1>,
        _indexes: [std::ops::Range<usize>; D2],
        _value: ADTensor<B, D1>,
    ) -> ADTensor<B, D1> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        _tensor: ADTensor<B, D>,
        _mask: BoolTensor<B, D>,
        _value: Elem<B>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn equal<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn equal_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        _lhs: ADTensor<B, D>,
        _rhs: ADTensor<B, D>,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal_scalar<const D: usize>(
        _lhs: ADTensor<B, D>,
        _rhs: Elem<B>,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal<const D: usize>(_lhs: ADTensor<B, D>, _rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal_scalar<const D: usize>(_lhs: ADTensor<B, D>, _rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn detach<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn mean<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn sum<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn mean_dim<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn sum_dim<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        _tensor: &ADTensor<B, D>,
    ) -> ADTensor<B::FullPrecisionBackend, D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        _tensor: ADTensor<B::FullPrecisionBackend, D>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn argmax<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn argmin<const D: usize>(_tensor: ADTensor<B, D>, _dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn exp<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log1p<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn powf<const D: usize>(_tensor: ADTensor<B, D>, _value: f32) -> ADTensor<B, D> {
        todo!()
    }

    fn sqrt<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cos<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn sin<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn tanh<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn erf<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cat<const D: usize>(_tensors: Vec<ADTensor<B, D>>, _dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn relu<const D: usize>(_tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> ADTensor<B, D> {
        Self::from_data(Data::ones(shape), device)
    }

    fn arange(range: std::ops::Range<usize>, device: &B::Device) -> IntTensor<B, 1> {
        B::arange(range, device)
    }
}
