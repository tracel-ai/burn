use crate::{
    binary_byte_cmp_ops, binary_byte_ops,
    client::FusionClient,
    get_client, scalar_byte2int_ops, scalar_byte_cmp_ops, scalar_byte_ops,
    stream::{execution::Operation, StreamId},
    unary_byte_ops, Fusion, FusionBackend,
};
use burn_tensor::{
    ops::{
        binary_ops_shape, BoolTensor, ByteElem, ByteTensor, ByteTensorOps, FloatTensor, IntElem,
        IntTensor,
    },
    repr::{self, *},
    DType, Device, Distribution, Element, ElementConversion, Shape, TensorData,
};
use core::ops::Range;
use std::marker::PhantomData;

impl<B: FusionBackend> ByteTensorOps<Self> for Fusion<B> {
    fn byte_empty(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        let client = get_client::<B>(&device.clone());
        let tensor = B::byte_empty(shape.clone(), device);
        let stream = StreamId::current();

        client.register_tensor(
            B::byte_tensor_handle(tensor),
            shape.dims,
            stream,
            B::ByteElem::dtype(),
        )
    }

    async fn byte_into_data(tensor: ByteTensor<Self>) -> TensorData {
        tensor.byte_into_data::<B>().await
    }

    fn byte_from_data(data: TensorData, device: &Device<Self>) -> ByteTensor<Self> {
        let client = get_client::<B>(&device.clone());
        let tensor = B::byte_from_data(data, device);
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        let stream = StreamId::current();

        client.register_tensor(
            B::byte_tensor_handle(tensor),
            shape.dims,
            stream,
            B::ByteElem::dtype(),
        )
    }

    fn byte_device(tensor: &ByteTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn byte_to_device(tensor: ByteTensor<Self>, device: &Device<Self>) -> ByteTensor<Self> {
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
            .change_client_int::<B>(tensor.into_description(), client_target, id)
    }

    fn byte_reshape(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        #[derive(new)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: ReshapeDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_reshape(input, Shape::from(&self.desc.out.shape));
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(shape.dims, B::ByteElem::dtype());

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<B>::new(desc),
        );

        out
    }

    fn byte_slice(tensor: ByteTensor<Self>, ranges: &[Range<usize>]) -> ByteTensor<Self> {
        #[derive(new)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);

                let output = B::byte_slice(tensor, self.desc.ranges.as_slice());

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
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
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<B>::new(desc),
        );

        out
    }

    fn byte_slice_assign(
        tensor: ByteTensor<Self>,
        ranges: &[Range<usize>],
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let value = handles.get_byte_tensor::<B>(&self.desc.value);

                let output = B::byte_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseByte(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<B>::new(desc),
        );

        out
    }

    fn byte_mask_where(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct MaskWhereOps<B: FusionBackend> {
            desc: MaskWhereOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let value = handles.get_byte_tensor::<B>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::byte_mask_where(tensor, mask, value);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let stream_3 = value.stream;
        let shape = binary_ops_shape(&tensor.shape, &mask.shape);
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = MaskWhereOperationDescription {
            tensor: tensor.into_description(),
            value: value.into_description(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MaskWhere(desc.clone()),
            ),
            MaskWhereOps::<B>::new(desc),
        );

        out
    }

    fn byte_mask_fill(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct MaskFillOps<B: FusionBackend> {
            desc: MaskFillOperationDescription<u32>,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::byte_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = MaskFillOperationDescription {
            tensor: tensor.into_description(),
            value: value.elem(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MaskFill(desc.clone()),
            ),
            MaskFillOps::<B>::new(desc),
        );

        out
    }

    fn byte_gather(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::byte_gather(self.desc.dim, tensor, indices);
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = GatherOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Gather(desc.clone()),
            ),
            GatherOps::<B>::new(desc),
        );

        out
    }

    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct ScatterOps<B: FusionBackend> {
            desc: ScatterOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_byte_tensor::<B>(&self.desc.value);

                let output = B::byte_scatter(self.desc.dim, tensor, indices, value);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = ScatterOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Scatter(desc.clone()),
            ),
            ScatterOps::<B>::new(desc),
        );

        out
    }

    fn byte_select(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct SelectOps<B: FusionBackend> {
            desc: SelectOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::byte_select(tensor, self.desc.dim, indices);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = SelectOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Select(desc.clone()),
            ),
            SelectOps::<B>::new(desc),
        );

        out
    }

    fn byte_select_assign(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct SelectAssignOps<B: FusionBackend> {
            desc: SelectAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_byte_tensor::<B>(&self.desc.value);

                let output = B::byte_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());
        let desc = SelectAssignOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::SelectAssign(desc.clone()),
            ),
            SelectAssignOps::<B>::new(desc),
        );

        out
    }

    fn byte_cat(tensors: Vec<ByteTensor<Self>>, dim: usize) -> ByteTensor<Self> {
        #[derive(new)]
        struct CatOps<B: FusionBackend> {
            desc: CatOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_byte_tensor::<B>(tensor))
                    .collect();

                let output = B::byte_cat(tensors, self.desc.dim);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
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

        let out = client.tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            streams,
            OperationDescription::BaseByte(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<B>::new(desc),
        );

        out
    }

    fn byte_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        binary_byte_cmp_ops!(EqualOps, B::byte_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseByte(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<B>::new(desc),
        );

        out
    }

    fn byte_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        scalar_byte_cmp_ops!(EqualElemOps, B::byte_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::EqualElem(desc.clone()),
            ),
            EqualElemOps::<B>::new(desc),
        );

        out
    }

    fn byte_greater(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        binary_byte_cmp_ops!(GreaterOps, B::byte_greater);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Greater(desc.clone()),
            ),
            GreaterOps::<B>::new(desc),
        );

        out
    }

    fn byte_greater_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        scalar_byte_cmp_ops!(GreaterElemOps, B::byte_greater_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::GreaterElem(desc.clone()),
            ),
            GreaterElemOps::<B>::new(desc),
        );

        out
    }

    fn byte_greater_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        binary_byte_cmp_ops!(GreaterEqualOps, B::byte_greater_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::GreaterEqual(desc.clone()),
            ),
            GreaterEqualOps::<B>::new(desc),
        );

        out
    }

    fn byte_greater_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        scalar_byte_cmp_ops!(GreaterEqualElemOps, B::byte_greater_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::GreaterEqualElem(desc.clone()),
            ),
            GreaterEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn byte_lower(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        binary_byte_cmp_ops!(LowerOps, B::byte_lower);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Lower(desc.clone()),
            ),
            LowerOps::<B>::new(desc),
        );

        out
    }

    fn byte_lower_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        scalar_byte_cmp_ops!(LowerElemOps, B::byte_lower_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::LowerElem(desc.clone()),
            ),
            LowerElemOps::<B>::new(desc),
        );

        out
    }

    fn byte_lower_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        binary_byte_cmp_ops!(LowerEqualOps, B::byte_lower_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::LowerEqual(desc.clone()),
            ),
            LowerEqualOps::<B>::new(desc),
        );

        out
    }

    fn byte_lower_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        scalar_byte_cmp_ops!(LowerEqualElemOps, B::byte_lower_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::LowerEqualElem(desc.clone()),
            ),
            LowerEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn byte_add(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        binary_byte_ops!(AddOps, B::byte_add);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::ByteElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Add(desc.clone()),
            ),
            AddOps::<B>::new(desc),
        );

        out
    }

    fn byte_add_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        scalar_byte_ops!(AddOps, B::byte_add_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::AddScalar(desc.clone()),
            ),
            AddOps::<B>::new(desc),
        );

        out
    }

    fn byte_sub(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        binary_byte_ops!(SubOps, B::byte_sub);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::ByteElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Sub(desc.clone()),
            ),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn byte_sub_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        scalar_byte_ops!(SubOps, B::byte_sub_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::SubScalar(desc.clone()),
            ),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn byte_mul(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        binary_byte_ops!(MulOps, B::byte_mul);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::ByteElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Mul(desc.clone()),
            ),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn byte_mul_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        scalar_byte_ops!(MulOps, B::byte_mul_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MulScalar(desc.clone()),
            ),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn byte_div(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        binary_byte_ops!(DivOps, B::byte_div);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::ByteElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Div(desc.clone()),
            ),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn byte_div_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        scalar_byte_ops!(DivOps, B::byte_div_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::DivScalar(desc.clone()),
            ),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn byte_remainder(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        binary_byte_ops!(ModOps, B::byte_remainder);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::ByteElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Rem(desc.clone()),
            ),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn byte_remainder_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        scalar_byte_ops!(ModOps, B::byte_remainder_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::RemScalar(desc.clone()),
            ),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn byte_zeros(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        #[derive(new)]
        struct ZerosOps<B: FusionBackend> {
            desc: TensorDescription,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::byte_zeros(shape, &self.device);
                handles.register_byte_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::ByteElem::dtype());
        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Zeros(desc.clone()),
            ),
            ZerosOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn byte_ones(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        #[derive(new)]
        struct OnesOps<B: FusionBackend> {
            desc: TensorDescription,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::byte_ones(shape, &self.device);
                handles.register_byte_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::ByteElem::dtype());

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Ones(desc.clone()),
            ),
            OnesOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn byte_sum(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(SumOps, B::byte_sum, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Sum(desc.clone()),
            ),
            SumOps::<B>::new(desc),
        );

        out
    }

    fn byte_sum_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte_ops!(SumDimOps, B::byte_sum_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::SumDim(desc.clone()),
            ),
            SumDimOps::<B>::new(desc),
        );

        out
    }

    fn byte_prod(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(ProdOps, B::byte_prod, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Prod(desc.clone()),
            ),
            ProdOps::<B>::new(desc),
        );

        out
    }

    fn byte_prod_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte_ops!(ProdDimOps, B::byte_prod_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::ProdDim(desc.clone()),
            ),
            ProdDimOps::<B>::new(desc),
        );

        out
    }

    fn byte_mean(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(MeanOps, B::byte_mean, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Mean(desc.clone()),
            ),
            MeanOps::<B>::new(desc),
        );

        out
    }

    fn byte_mean_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte_ops!(MeanDimOps, B::byte_mean_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MeanDim(desc.clone()),
            ),
            MeanDimOps::<B>::new(desc),
        );

        out
    }

    fn byte_argmax(tensor: ByteTensor<Self>, dim: usize) -> IntTensor<Self> {
        scalar_byte2int_ops!(ArgMaxOps, B::byte_argmax, usize);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                IntElem::<Self>::dtype(),
                NumericOperationDescription::ArgMax(desc.clone()),
            ),
            ArgMaxOps::<B>::new(desc),
        );

        out
    }

    fn byte_argmin(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte2int_ops!(ArgMinOps, B::byte_argmin, usize);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::ArgMin(desc.clone()),
            ),
            ArgMinOps::<B>::new(desc),
        );

        out
    }

    fn byte_clamp(
        tensor: ByteTensor<Self>,
        min: ByteElem<Self>,
        max: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct ClampOps<B: FusionBackend> {
            desc: ClampOperationDescription<u32>,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let output = B::byte_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::ByteElem::dtype());
        let desc = ClampOperationDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Clamp(desc.clone()),
            ),
            ClampOps::<B>::new(desc),
        );

        out
    }

    fn byte_abs(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(AbsOps, B::byte_abs);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Abs(desc.clone()),
            ),
            AbsOps::<B>::new(desc),
        );

        out
    }

    fn byte_into_float(tensor: ByteTensor<Self>) -> FloatTensor<Self> {
        #[derive(new)]
        struct IntoFloatOps<B: FusionBackend> {
            desc: UnaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_into_float(input);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());
        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Byte(repr::ByteOperationDescription::IntoFloat(desc.clone())),
            IntoFloatOps::<B>::new(desc),
        );

        out
    }

    fn byte_into_int(tensor: ByteTensor<Self>) -> IntTensor<Self> {
        #[derive(new)]
        struct IntoIntOps<B: FusionBackend> {
            desc: UnaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_into_int(input);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());
        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Byte(repr::ByteOperationDescription::IntoInt(desc.clone())),
            IntoIntOps::<B>::new(desc),
        );

        out
    }

    fn byte_swap_dims(tensor: ByteTensor<Self>, dim1: usize, dim2: usize) -> ByteTensor<Self> {
        #[derive(new)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            dim1,
            dim2,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<B>::new(desc),
        );

        out
    }

    fn byte_max(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(MaxOps, B::byte_max, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Max(desc.clone()),
            ),
            MaxOps::<B>::new(desc),
        );

        out
    }

    fn byte_max_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte_ops!(MaxDimOps, B::byte_max_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MaxDim(desc.clone()),
            ),
            MaxDimOps::<B>::new(desc),
        );

        out
    }

    fn byte_max_dim_with_indices(
        tensor: ByteTensor<Self>,
        dim: usize,
    ) -> (ByteTensor<Self>, IntTensor<Self>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::byte_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::ByteElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MaxDimWithIndices(desc.clone()),
            ),
            MaxDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn byte_min(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_byte_ops!(MinOps, B::byte_min, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::ByteElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::Min(desc.clone()),
            ),
            MinOps::<B>::new(desc),
        );

        out
    }

    fn byte_min_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        scalar_byte_ops!(MinDimOps, B::byte_min_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MinDim(desc.clone()),
            ),
            MinDimOps::<B>::new(desc),
        );

        out
    }

    fn byte_min_dim_with_indices(
        tensor: ByteTensor<Self>,
        dim: usize,
    ) -> (ByteTensor<Self>, IntTensor<Self>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::byte_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::ByteElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::MinDimWithIndices(desc.clone()),
            ),
            MinDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> ByteTensor<Self> {
        #[derive(new)]
        struct IntRandomOps<B: FusionBackend> {
            desc: RandomOperationDescription,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntRandomOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output = B::byte_random(shape, self.desc.distribution, &self.device);
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims, B::ByteElem::dtype());

        let desc = RandomOperationDescription {
            out: out.to_description_out(),
            distribution,
        };
        client.register(
            vec![stream],
            OperationDescription::NumericByte(
                ByteElem::<Self>::dtype(),
                NumericOperationDescription::IntRandom(desc.clone()),
            ),
            IntRandomOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        #[derive(new)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_permute(input, self.desc.axes.as_slice());
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = PermuteOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::Permute(desc.clone())),
            PermuteDimsOps::<B>::new(desc),
        );

        out
    }

    fn byte_expand(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        #[derive(new)]
        struct ExpandOps<B: FusionBackend> {
            desc: ExpandOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let output = B::byte_expand(input, self.desc.shape.into());
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.clone(), B::ByteElem::dtype());

        let desc = ExpandOperationDescription {
            input: tensor.into_description(),
            shape: shape.dims,
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<B>::new(desc),
        );

        out
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        #[derive(new)]
        struct FlipDimsOps<B: FusionBackend> {
            desc: FlipOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_byte_tensor::<B>(&self.desc.input);
                let axes = &self.desc.axes;
                let output = B::byte_flip(input, axes);
                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::ByteElem::dtype());

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::Flip(desc.clone())),
            FlipDimsOps::<B>::new(desc),
        );

        out
    }

    fn byte_repeat_dim(tensor: ByteTensor<Self>, dim: usize, times: usize) -> ByteTensor<Self> {
        #[derive(new)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_byte_tensor::<B>(&self.desc.tensor);

                let output = B::byte_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_byte_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::ByteElem::dtype());

        let desc = RepeatDimOperationDescription {
            tensor: tensor.into_description(),
            dim,
            times,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseByte(BaseOperationDescription::RepeatDim(desc.clone())),
            RepeatDimOps::<B>::new(desc),
        );

        out
    }
}
