use crate::{
    binary_int_ops,
    client::FusionClient,
    graph::{self, BinaryOpsDescription, NumericOpsDescription, Ops, ScalarOpsDescription},
    ops::binary::binary_ops_shape,
    scalar_int_ops, FusedBackend, Fusion,
};
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Data, Device, Reader, Shape,
};
use core::ops::Range;

impl<B: FusedBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        let client = B::client(&device.clone().into());
        let out = client.create_empty(shape.dims.into());
        out
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        tensor.int_into_data()
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let client = B::client(&device.clone().into());
        let out = client.create_int(data.value, data.shape.dims.into());
        out
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone().into()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        todo!()
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        todo!()
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        todo!()
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(AddOps, B::int_add);

        let out = lhs
            .client
            .create_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Add(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(AddOps::<D>),
                ),
            ));

        out
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let out = lhs.client.create_empty(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::AddScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(AddOps::<D>),
                ),
            ));

        out
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(SubOps, B::int_sub);

        let out = lhs
            .client
            .create_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Sub(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(SubOps::<D>),
                ),
            ));

        out
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let out = lhs.client.create_empty(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::SubScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(SubOps::<D>),
                ),
            ));

        out
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(MulOps, B::int_mul);

        let out = lhs
            .client
            .create_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Mul(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(MulOps::<D>),
                ),
            ));

        out
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let out = lhs.client.create_empty(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::MulScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(MulOps::<D>),
                ),
            ));

        out
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(DivOps, B::int_div);

        let out = lhs
            .client
            .create_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Div(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(DivOps::<D>),
                ),
            ));

        out
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let out = lhs.client.create_empty(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::DivScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(DivOps::<D>),
                ),
            ));

        out
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        todo!()
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_clamp_min<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_clamp_max<const D: usize>(
        tensor: IntTensor<Self, D>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        todo!()
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn int_swap_dims<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        todo!()
    }
}
