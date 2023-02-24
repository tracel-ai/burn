use crate::{
    grads::Gradients,
    graph::ops::{Ops, OpsMetadataRef},
    tensor::{ADTensor, BoolTensor, Elem, IntTensor},
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::TensorOps};

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn from_data<const D: usize>(
        data: burn_tensor::Data<Elem<B>, D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        ADTensor::new(B::from_data(data, device))
    }

    fn from_data_bool<const D: usize>(
        data: burn_tensor::Data<bool, D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn random<const D: usize>(
        shape: burn_tensor::Shape<D>,
        distribution: burn_tensor::Distribution<Elem<B>>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn shape<const D: usize>(tensor: &ADTensor<B, D>) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn to_data<const D: usize>(tensor: &ADTensor<B, D>) -> burn_tensor::Data<Elem<B>, D> {
        B::to_data(&tensor.primitive)
    }

    fn into_data<const D: usize>(tensor: ADTensor<B, D>) -> burn_tensor::Data<Elem<B>, D> {
        B::into_data(tensor.primitive)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> burn_tensor::Data<bool, D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> burn_tensor::Data<bool, D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D> {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> BoolTensor<B, D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> BoolTensor<B, D2> {
        todo!()
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<B, D1> {
        todo!()
    }

    fn device<const D: usize>(
        tensor: &ADTensor<B, D>,
    ) -> <ADBackendDecorator<B> as Backend>::Device {
        todo!()
    }

    fn to_device<const D: usize>(
        tensor: ADTensor<B, D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn add<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(new, Debug)]
        struct Add<B: Backend, const D: usize> {
            lhs: ADTensor<B, D>,
            rhs: ADTensor<B, D>,
            output: ADTensor<B, D>,
        }

        impl<B: Backend, const D: usize> Ops<B> for Add<B, D> {
            fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
                let grad_output = grads.consume(&self.output);

                grads.update(self.lhs, grad_output.clone());
                grads.update(self.rhs, grad_output);
            }

            fn metadata(&self) -> OpsMetadataRef {
                self.output.metadata.clone()
            }
        }

        let output = B::add(lhs.primitive.clone(), rhs.primitive.clone());
        let output = ADTensor::from_binary_ops(lhs.clone(), rhs.clone(), output);
        let ops = Add::new(lhs, rhs, output.clone());

        output.register_ops(ops)
    }

    fn add_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        #[derive(new, Debug)]
        struct AddScalar<B: Backend, const D: usize> {
            lhs: ADTensor<B, D>,
            output: ADTensor<B, D>,
        }

        impl<B: Backend, const D: usize> Ops<B> for AddScalar<B, D> {
            fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
                let grad_output = grads.consume(&self.output);

                grads.update(self.lhs, grad_output.clone());
            }

            fn metadata(&self) -> OpsMetadataRef {
                self.output.metadata.clone()
            }
        }

        let output = B::add_scalar(lhs.primitive.clone(), rhs.clone());
        let output = ADTensor::from_unary_ops(lhs.clone(), output);
        let ops = AddScalar::new(lhs, output.clone());

        output.register_ops(ops)
    }

    fn sub<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn sub_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        todo!()
    }

    fn mul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        todo!()
    }

    fn div<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn div_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> ADTensor<B, D> {
        todo!()
    }

    fn matmul<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn neg<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        tensor: ADTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> ADTensor<B, D2> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> ADTensor<B, D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: ADTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: ADTensor<B, D1>,
    ) -> ADTensor<B, D1> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        tensor: ADTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: Elem<B>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn greater_equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal<const D: usize>(lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> BoolTensor<B, D> {
        todo!()
    }

    fn lower_equal_scalar<const D: usize>(lhs: ADTensor<B, D>, rhs: Elem<B>) -> BoolTensor<B, D> {
        todo!()
    }

    fn detach<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn mean<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn sum<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, 1> {
        todo!()
    }

    fn mean_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn sum_dim<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        tensor: &ADTensor<B, D>,
    ) -> ADTensor<B::FullPrecisionBackend, D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        tensor: ADTensor<B::FullPrecisionBackend, D>,
    ) -> ADTensor<B, D> {
        todo!()
    }

    fn argmax<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn argmin<const D: usize>(tensor: ADTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        todo!()
    }

    fn exp<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn log1p<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn powf<const D: usize>(tensor: ADTensor<B, D>, value: f32) -> ADTensor<B, D> {
        todo!()
    }

    fn sqrt<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cos<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn sin<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn tanh<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn erf<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn cat<const D: usize>(tensors: Vec<ADTensor<B, D>>, dim: usize) -> ADTensor<B, D> {
        todo!()
    }

    fn relu<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        todo!()
    }

    fn zeros<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        Self::from_data(burn_tensor::Data::zeros(shape), device)
    }

    fn ones<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> ADTensor<B, D> {
        Self::from_data(burn_tensor::Data::ones(shape), device)
    }

    fn arange(
        range: std::ops::Range<usize>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> IntTensor<B, 1> {
        todo!()
    }

    fn repeat<const D: usize>(tensor: ADTensor<B, D>, dim: usize, times: usize) -> ADTensor<B, D> {
        let mut shape = <ADBackendDecorator<B>>::shape(&tensor);
        if shape.dims[dim] != 1 {
            panic!("Can only repeat dimension with dim=1");
        }
        shape.dims[dim] = times;

        let mut i = 0;
        let indexes_select_all = [0; D].map(|_| {
            let start = 0;
            let end = shape.dims[i];
            i += 1;
            start..end
        });

        let mut tensor_output =
            <ADBackendDecorator<B>>::empty(shape, &<ADBackendDecorator<B>>::device(&tensor));
        for i in 0..times {
            let mut indexes = indexes_select_all.clone();
            indexes[dim] = i..i + 1;
            tensor_output =
                <ADBackendDecorator<B>>::index_assign(tensor_output, indexes, tensor.clone());
        }

        tensor_output
    }

    fn transpose<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        Self::swap_dims(tensor, D - 2, D - 1)
    }
}
