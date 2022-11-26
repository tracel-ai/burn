use crate::{backend::Backend, tensor::Shape, Data, ElementConversion};
use std::ops::Range;

pub trait ModuleOps<B: Backend> {
    fn embedding(
        weights: &B::TensorPrimitive<2>,
        indexes: &<B::IntegerBackend as Backend>::TensorPrimitive<2>,
    ) -> B::TensorPrimitive<3>;
    fn embedding_backward(
        weights: &B::TensorPrimitive<2>,
        output: &B::TensorPrimitive<3>,
        indexes: &<B::IntegerBackend as Backend>::TensorPrimitive<2>,
    ) -> B::TensorPrimitive<2>;
}

pub trait TensorOps<B: Backend> {
    fn shape<const D: usize>(tensor: &B::TensorPrimitive<D>) -> &Shape<D>;
    fn to_data<const D: usize>(tensor: &B::TensorPrimitive<D>) -> Data<B::Elem, D>;
    fn into_data<const D: usize>(tensor: B::TensorPrimitive<D>) -> Data<B::Elem, D>;
    fn bool_shape<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> &Shape<D>;
    fn bool_to_data<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> Data<bool, D>;
    fn bool_into_data<const D: usize>(tensor: B::BoolTensorPrimitive<D>) -> Data<bool, D>;
    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: &B::BoolTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::BoolTensorPrimitive<D2>;
    fn device<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::Device;
    fn to_device<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        device: B::Device,
    ) -> B::TensorPrimitive<D>;
    fn arange(
        range: Range<usize>,
        device: B::Device,
    ) -> <B::IntegerBackend as Backend>::TensorPrimitive<1> {
        let shape = Shape::new([range.end - range.start]);
        let value = range
            .into_iter()
            .map(|i| (i as i64).to_elem())
            .collect::<Vec<<B::IntegerBackend as Backend>::Elem>>();
        let data = Data::new(value, shape);
        <B::IntegerBackend as Backend>::from_data(data, device)
    }
    fn empty<const D: usize>(shape: Shape<D>, device: B::Device) -> B::TensorPrimitive<D>;
    fn repeat<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        dim: usize,
        times: usize,
    ) -> B::TensorPrimitive<D> {
        let mut shape = *B::shape(tensor);
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

        let mut tensor_output = B::empty(shape, B::device(tensor));
        for i in 0..times {
            let mut indexes = indexes_select_all.clone();
            indexes[dim] = i..i + 1;
            tensor_output = B::index_assign(&tensor_output, indexes, tensor);
        }

        tensor_output
    }
    fn add<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn add_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::TensorPrimitive<D>;
    fn sub<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn sub_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::TensorPrimitive<D>;
    fn mul<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn mul_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::TensorPrimitive<D>;
    fn div<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn div_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::TensorPrimitive<D>;
    fn matmul<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn neg<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
    fn transpose<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D> {
        Self::swap_dims(tensor, D - 2, D - 1)
    }
    fn swap_dims<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> B::TensorPrimitive<D>;
    fn reshape<const D1: usize, const D2: usize>(
        tensor: &B::TensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::TensorPrimitive<D2>;
    fn index<const D1: usize, const D2: usize>(
        tensor: &B::TensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> B::TensorPrimitive<D1>;
    fn index_assign<const D1: usize, const D2: usize>(
        tensor: &B::TensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
        value: &B::TensorPrimitive<D1>,
    ) -> B::TensorPrimitive<D1>;
    fn mask_fill<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        mask: &B::BoolTensorPrimitive<D>,
        value: B::Elem,
    ) -> B::TensorPrimitive<D>;
    fn equal<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn equal_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::BoolTensorPrimitive<D>;
    fn greater<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn greater_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::BoolTensorPrimitive<D>;
    fn greater_equal<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn greater_equal_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::BoolTensorPrimitive<D>;
    fn lower<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn lower_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::BoolTensorPrimitive<D>;
    fn lower_equal<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn lower_equal_scalar<const D: usize>(
        lhs: &B::TensorPrimitive<D>,
        rhs: &B::Elem,
    ) -> B::BoolTensorPrimitive<D>;
    fn detach<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
    fn mean<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<1>;
    fn sum<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<1>;
    fn mean_dim<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        dim: usize,
    ) -> B::TensorPrimitive<D>;
    fn sum_dim<const D: usize>(tensor: &B::TensorPrimitive<D>, dim: usize)
        -> B::TensorPrimitive<D>;
    fn to_full_precision<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
    ) -> <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>;
    fn from_full_precision<const D: usize>(
        tensor: &<B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn argmax<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        dim: usize,
    ) -> <B::IntegerBackend as Backend>::TensorPrimitive<D>;
    fn argmin<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
        dim: usize,
    ) -> <B::IntegerBackend as Backend>::TensorPrimitive<D>;
    fn exp<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
    fn log<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
    fn powf<const D: usize>(tensor: &B::TensorPrimitive<D>, value: f32) -> B::TensorPrimitive<D>;
    fn erf<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
    fn cat<const D: usize>(tensors: &[B::TensorPrimitive<D>], dim: usize) -> B::TensorPrimitive<D>;
    fn relu<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;
}

pub trait Zeros {
    fn zeros(&self) -> Self;
}

pub trait Ones {
    fn ones(&self) -> Self;
}
