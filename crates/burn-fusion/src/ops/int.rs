use crate::{
    binary_int_cmp_ops, binary_int_ops,
    client::FusionClient,
    get_client, scalar_int_cmp_ops, scalar_int_ops,
    stream::{execution::Operation, StreamId},
    unary_int_ops, Fusion, FusionBackend,
};
use burn_ir::*;
use burn_tensor::{
    ops::{binary_ops_shape, BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Device, Distribution, Element, ElementConversion, Shape, TensorData, TensorMetadata,
};
use core::ops::Range;
use std::marker::PhantomData;

use super::NoOp;

impl<B: FusionBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        #[derive(new)]
        struct EmptyOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EmptyOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let output = B::int_empty(Shape::from(&self.desc.shape), &self.device);
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims.clone(), B::IntElem::dtype());

        let desc = out.to_ir_out();

        client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Empty(desc.clone())),
            EmptyOps::<B>::new(desc, device.clone()),
        );

        out
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        tensor.int_into_data::<B>().await
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let dtype = data.dtype;
        let tensor = B::int_from_data(data, device);
        let shape = tensor.shape();

        let handle = B::int_tensor_handle(tensor);
        let out = client.register_tensor(handle, shape.dims, stream, dtype);
        let desc = out.to_ir_out();

        client.register(
            vec![stream],
            OperationIr::Init(InitOperationIr { out: desc }),
            NoOp::<B>::new(),
        );

        out
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_int::<B>(tensor.into_ir(), client_target, id)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        #[derive(new)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_reshape(input, Shape::from(&self.desc.out.shape));
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(shape.dims, B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Reshape(desc.clone())),
            ReshapeDimsOps::<B>::new(desc),
        );

        out
    }

    fn int_slice(tensor: IntTensor<Self>, ranges: &[Range<usize>]) -> IntTensor<Self> {
        #[derive(new)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);

                let output = B::int_slice(tensor, self.desc.ranges.as_slice());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let ndims = burn_tensor::TensorMetadata::shape(&tensor).num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.into(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Slice(desc.clone())),
            SliceOps::<B>::new(desc),
        );

        out
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.into(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc.clone())),
            SliceAssignOps::<B>::new(desc),
        );

        out
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct MaskWhereOps<B: FusionBackend> {
            desc: MaskWhereOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let stream_3 = value.stream;
        let shape = binary_ops_shape(&tensor.shape, &mask.shape);
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = MaskWhereOpIr {
            tensor: tensor.into_ir(),
            value: value.into_ir(),
            mask: mask.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MaskWhere(desc.clone()),
            ),
            MaskWhereOps::<B>::new(desc),
        );

        out
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct MaskFillOps<B: FusionBackend> {
            desc: MaskFillOpIr<i32>,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::int_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = MaskFillOpIr {
            tensor: tensor.into_ir(),
            value: value.elem(),
            mask: mask.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MaskFill(desc.clone()),
            ),
            MaskFillOps::<B>::new(desc),
        );

        out
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::int_gather(self.desc.dim, tensor, indices);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = GatherOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Gather(desc.clone()),
            ),
            GatherOps::<B>::new(desc),
        );

        out
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct ScatterOps<B: FusionBackend> {
            desc: ScatterOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_scatter(self.desc.dim, tensor, indices, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ScatterOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Scatter(desc.clone()),
            ),
            ScatterOps::<B>::new(desc),
        );

        out
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct SelectOps<B: FusionBackend> {
            desc: SelectOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::int_select(tensor, self.desc.dim, indices);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SelectOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Select(desc.clone()),
            ),
            SelectOps::<B>::new(desc),
        );

        out
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct SelectAssignOps<B: FusionBackend> {
            desc: SelectAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SelectAssignOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::SelectAssign(desc.clone()),
            ),
            SelectAssignOps::<B>::new(desc),
        );

        out
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        #[derive(new)]
        struct CatOps<B: FusionBackend> {
            desc: CatOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_int_tensor::<B>(tensor))
                    .collect();

                let output = B::int_cat(tensors, self.desc.dim);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();

        // Calculate the output shape
        let streams = tensors.iter().map(|tensor| tensor.stream).collect();
        let mut shape: Vec<usize> = tensor_first.shape.clone();
        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }

        let out = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|t| t.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };
        client.register(
            streams,
            OperationIr::BaseInt(BaseOperationIr::Cat(desc.clone())),
            CatOps::<B>::new(desc),
        );

        out
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(EqualOps, B::int_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::BaseInt(BaseOperationIr::Equal(desc.clone())),
            EqualOps::<B>::new(desc),
        );

        out
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::EqualElem(desc.clone()),
            ),
            EqualElemOps::<B>::new(desc),
        );

        out
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(GreaterOps, B::int_greater);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Greater(desc.clone()),
            ),
            GreaterOps::<B>::new(desc),
        );

        out
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::GreaterElem(desc.clone()),
            ),
            GreaterElemOps::<B>::new(desc),
        );

        out
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(GreaterEqualOps, B::int_greater_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::GreaterEqual(desc.clone()),
            ),
            GreaterEqualOps::<B>::new(desc),
        );

        out
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::GreaterEqualElem(desc.clone()),
            ),
            GreaterEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(LowerOps, B::int_lower);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Lower(desc.clone()),
            ),
            LowerOps::<B>::new(desc),
        );

        out
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::LowerElem(desc.clone()),
            ),
            LowerElemOps::<B>::new(desc),
        );

        out
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(LowerEqualOps, B::int_lower_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::LowerEqual(desc.clone()),
            ),
            LowerEqualOps::<B>::new(desc),
        );

        out
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::LowerEqualElem(desc.clone()),
            ),
            LowerEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(AddOps, B::int_add);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Add(desc.clone()),
            ),
            AddOps::<B>::new(desc),
        );

        out
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::AddScalar(desc.clone()),
            ),
            AddOps::<B>::new(desc),
        );

        out
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(SubOps, B::int_sub);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Sub(desc.clone()),
            ),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::SubScalar(desc.clone()),
            ),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(MulOps, B::int_mul);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Mul(desc.clone()),
            ),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MulScalar(desc.clone()),
            ),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(DivOps, B::int_div);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Div(desc.clone()),
            ),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::DivScalar(desc.clone()),
            ),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(ModOps, B::int_remainder);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Rem(desc.clone()),
            ),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(ModOps, B::int_remainder_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::RemScalar(desc.clone()),
            ),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        #[derive(new)]
        struct ZerosOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_zeros(shape, &self.device);
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::IntElem::dtype());
        let desc = out.to_ir_out();
        client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Zeros(desc.clone()),
            ),
            ZerosOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        #[derive(new)]
        struct OnesOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_ones(shape, &self.device);
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::IntElem::dtype());

        let desc = out.to_ir_out();
        client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Ones(desc.clone()),
            ),
            OnesOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(SumOps, B::int_sum, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Sum(desc.clone()),
            ),
            SumOps::<B>::new(desc),
        );

        out
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(SumDimOps, B::int_sum_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::SumDim(desc.clone()),
            ),
            SumDimOps::<B>::new(desc),
        );

        out
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(ProdOps, B::int_prod, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Prod(desc.clone()),
            ),
            ProdOps::<B>::new(desc),
        );

        out
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(ProdDimOps, B::int_prod_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::ProdDim(desc.clone()),
            ),
            ProdDimOps::<B>::new(desc),
        );

        out
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MeanOps, B::int_mean, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Mean(desc.clone()),
            ),
            MeanOps::<B>::new(desc),
        );

        out
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(MeanDimOps, B::int_mean_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MeanDim(desc.clone()),
            ),
            MeanDimOps::<B>::new(desc),
        );

        out
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(ArgMaxOps, B::int_argmax, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::ArgMax(desc.clone()),
            ),
            ArgMaxOps::<B>::new(desc),
        );

        out
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(ArgMinOps, B::int_argmin, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::ArgMin(desc.clone()),
            ),
            ArgMinOps::<B>::new(desc),
        );

        out
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct ClampOps<B: FusionBackend> {
            desc: ClampOpIr<i32>,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.tensor);
                let output = B::int_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());
        let desc = ClampOpIr {
            tensor: tensor.into_ir(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Clamp(desc.clone()),
            ),
            ClampOps::<B>::new(desc),
        );

        out
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(AbsOps, B::int_abs);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Abs(desc.clone()),
            ),
            AbsOps::<B>::new(desc),
        );

        out
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        #[derive(new)]
        struct IntoFloatOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_into_float(input);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());
        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::IntoFloat(desc.clone())),
            IntoFloatOps::<B>::new(desc),
        );

        out
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        #[derive(new)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = SwapDimsOpIr {
            input: tensor.into_ir(),
            dim1,
            dim2,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::SwapDims(desc.clone())),
            SwapDimsOps::<B>::new(desc),
        );

        out
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MaxOps, B::int_max, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Max(desc.clone()),
            ),
            MaxOps::<B>::new(desc),
        );

        out
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(MaxDimOps, B::int_max_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MaxDim(desc.clone()),
            ),
            MaxDimOps::<B>::new(desc),
        );

        out
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::IntElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MaxDimWithIndices(desc.clone()),
            ),
            MaxDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MinOps, B::int_min, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::Min(desc.clone()),
            ),
            MinOps::<B>::new(desc),
        );

        out
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_int_ops!(MinDimOps, B::int_min_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MinDim(desc.clone()),
            ),
            MinDimOps::<B>::new(desc),
        );

        out
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::IntElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::MinDimWithIndices(desc.clone()),
            ),
            MinDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        #[derive(new)]
        struct IntRandomOps<B: FusionBackend> {
            desc: RandomOpIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntRandomOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output = B::int_random(shape, self.desc.distribution, &self.device);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::IntElem::dtype());

        let desc = RandomOpIr {
            out: out.to_ir_out(),
            distribution,
        };
        client.register(
            vec![stream],
            OperationIr::NumericInt(
                IntElem::<Self>::dtype(),
                NumericOperationIr::IntRandom(desc.clone()),
            ),
            IntRandomOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        #[derive(new)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_permute(input, self.desc.axes.as_slice());
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            axes: axes.to_vec(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Permute(desc.clone())),
            PermuteDimsOps::<B>::new(desc),
        );

        out
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        #[derive(new)]
        struct ExpandOps<B: FusionBackend> {
            desc: ExpandOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_expand(input, self.desc.shape.into());
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.clone(), B::IntElem::dtype());

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            shape: shape.dims,
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Expand(desc.clone())),
            ExpandOps::<B>::new(desc),
        );

        out
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        #[derive(new)]
        struct FlipDimsOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let axes = &self.desc.axes;
                let output = B::int_flip(input, axes);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            axes: axes.to_vec(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Flip(desc.clone())),
            FlipDimsOps::<B>::new(desc),
        );

        out
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        #[derive(new)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);

                let output = B::int_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = RepeatDimOpIr {
            tensor: tensor.into_ir(),
            dim,
            times,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::RepeatDim(desc.clone())),
            RepeatDimOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseAndOps, B::bitwise_and);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Int(IntOperationIr::BitwiseAnd(desc.clone())),
            BitwiseAndOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseAndOps, B::bitwise_and_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseAndScalar(desc.clone())),
            BitwiseAndOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseOrOps, B::bitwise_or);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Int(IntOperationIr::BitwiseOr(desc.clone())),
            BitwiseOrOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseOrOps, B::bitwise_or_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseOrScalar(desc.clone())),
            BitwiseOrOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseXorOps, B::bitwise_xor);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Int(IntOperationIr::BitwiseXor(desc.clone())),
            BitwiseXorOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseXorOps, B::bitwise_xor_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseXorScalar(desc.clone())),
            BitwiseXorOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(BitwiseNotOps, B::bitwise_not);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseNot(desc.clone())),
            BitwiseNotOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseLeftShiftOps, B::bitwise_left_shift);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Int(IntOperationIr::BitwiseLeftShift(desc.clone())),
            BitwiseLeftShiftOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseLeftShiftOps, B::bitwise_left_shift_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseLeftShiftScalar(desc.clone())),
            BitwiseLeftShiftOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseRightShiftOps, B::bitwise_right_shift);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Int(IntOperationIr::BitwiseRightShift(desc.clone())),
            BitwiseRightShiftOps::<B>::new(desc),
        );

        out
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseRightShiftOps, B::bitwise_right_shift_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Int(IntOperationIr::BitwiseRightShiftScalar(desc.clone())),
            BitwiseRightShiftOps::<B>::new(desc),
        );

        out
    }
}
