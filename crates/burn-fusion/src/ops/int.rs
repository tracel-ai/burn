use super::NoOp;
use crate::{
    Fusion, FusionBackend, binary_int_cmp_ops, binary_int_ops,
    client::{FusionClient, OperationOutput},
    get_client, reduce_int_ops, scalar_int_cmp_ops, scalar_int_ops,
    stream::{OperationStreams, StreamId, execution::Operation},
    unary_int_ops,
};
use burn_ir::*;
use burn_tensor::{
    Device, Distribution, Element, IntDType, Shape, Slice, TensorData,
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
};
use std::marker::PhantomData;

impl<B: FusionBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct EmptyOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EmptyOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output = B::int_empty(
                    self.desc.shape.clone(),
                    &self.device,
                    self.desc.dtype.into(),
                );
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseInt(BaseOperationIr::Empty(desc.clone())),
                EmptyOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        tensor.int_into_data::<B>().await
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let stream = StreamId::current();
        let client = get_client::<B>(device);
        let dtype = data.dtype;
        let tensor = B::int_from_data(data, device);
        let shape = burn_tensor::TensorMetadata::shape(&tensor);

        let handle = B::int_tensor_handle(tensor);
        let out = client.register_tensor(handle, shape, stream, dtype);
        let desc = out.to_ir_out();

        client.register(
            OperationStreams::default(),
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

        if device_original == device {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(device);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_int::<B>(tensor.into_ir(), client_target, id)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        if tensor.shape == shape {
            return tensor;
        }

        #[derive(new, Debug)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_reshape(input, self.desc.out.shape.clone());
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Reshape(desc.clone())),
                ReshapeDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);

                let output = B::int_slice(tensor, self.desc.ranges.as_slice());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Slice(desc.clone())),
                SliceOps::<B>::new(desc),
            )
            .output()
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &value]);

        let client = tensor.client.clone();
        let desc =
            SliceAssignOpIr::create(tensor.into_ir(), slices.into(), value.into_ir(), || {
                client.create_empty_handle()
            });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc.clone())),
                SliceAssignOps::<B>::new(desc),
            )
            .output()
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(MatmulOps, B::int_matmul);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = MatmulOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Matmul(desc.clone())),
                MatmulOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct MaskWhereOps<B: FusionBackend> {
            desc: MaskWhereOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &mask, &value]);

        let client = tensor.client.clone();
        let desc = MaskWhereOpIr::create(tensor.into_ir(), mask.into_ir(), value.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::MaskWhere(desc.clone()),
                ),
                MaskWhereOps::<B>::new(desc),
            )
            .output()
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct MaskFillOps<B: FusionBackend> {
            desc: MaskFillOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::int_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &mask]);

        let client = tensor.client.clone();
        let desc = MaskFillOpIr::create(tensor.into_ir(), mask.into_ir(), value, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::MaskFill(desc.clone())),
                MaskFillOps::<B>::new(desc),
            )
            .output()
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::int_gather(self.desc.dim, tensor, indices);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices]);

        let client = tensor.client.clone();
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Gather(desc.clone())),
                GatherOps::<B>::new(desc),
            )
            .output()
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct ScatterOps<B: FusionBackend> {
            desc: ScatterOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_scatter(self.desc.dim, tensor, indices, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);

        let client = tensor.client.clone();
        let desc = ScatterOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Scatter(desc.clone())),
                ScatterOps::<B>::new(desc),
            )
            .output()
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct SelectOps<B: FusionBackend> {
            desc: SelectOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::int_select(tensor, self.desc.dim, indices);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices]);

        let client = tensor.client.clone();
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Select(desc.clone())),
                SelectOps::<B>::new(desc),
            )
            .output()
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct SelectAssignOps<B: FusionBackend> {
            desc: SelectAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_int_tensor::<B>(&self.desc.value);

                let output = B::int_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);

        let client = tensor.client.clone();
        let desc = SelectAssignOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::SelectAssign(desc.clone()),
                ),
                SelectAssignOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CatOps<B: FusionBackend> {
            desc: CatOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
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

        let streams = OperationStreams::with_inputs(&tensors);

        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Cat(desc.clone())),
                CatOps::<B>::new(desc),
            )
            .output()
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(EqualOps, B::int_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Equal(desc.clone())),
                EqualOps::<B>::new(desc),
            )
            .output()
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::EqualElem(desc.clone()),
                ),
                EqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(GreaterOps, B::int_greater);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.lhs.dtype, NumericOperationIr::Greater(desc.clone())),
                GreaterOps::<B>::new(desc),
            )
            .output()
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterElem(desc.clone()),
                ),
                GreaterElemOps::<B>::new(desc),
            )
            .output()
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(GreaterEqualOps, B::int_greater_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterEqual(desc.clone()),
                ),
                GreaterEqualOps::<B>::new(desc),
            )
            .output()
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterEqualElem(desc.clone()),
                ),
                GreaterEqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(LowerOps, B::int_lower);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.lhs.dtype, NumericOperationIr::Lower(desc.clone())),
                LowerOps::<B>::new(desc),
            )
            .output()
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerElem(desc.clone()),
                ),
                LowerElemOps::<B>::new(desc),
            )
            .output()
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int_cmp_ops!(LowerEqualOps, B::int_lower_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerEqual(desc.clone()),
                ),
                LowerEqualOps::<B>::new(desc),
            )
            .output()
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerEqualElem(desc.clone()),
                ),
                LowerEqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(AddOps, B::int_add);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Add(desc.clone())),
                AddOps::<B>::new(desc),
            )
            .output()
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::AddScalar(desc.clone()),
                ),
                AddOps::<B>::new(desc),
            )
            .output()
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(SubOps, B::int_sub);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Sub(desc.clone())),
                SubOps::<B>::new(desc),
            )
            .output()
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::SubScalar(desc.clone()),
                ),
                SubOps::<B>::new(desc),
            )
            .output()
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(MulOps, B::int_mul);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Mul(desc.clone())),
                MulOps::<B>::new(desc),
            )
            .output()
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::MulScalar(desc.clone()),
                ),
                MulOps::<B>::new(desc),
            )
            .output()
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(DivOps, B::int_div);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Div(desc.clone())),
                DivOps::<B>::new(desc),
            )
            .output()
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::DivScalar(desc.clone()),
                ),
                DivOps::<B>::new(desc),
            )
            .output()
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(ModOps, B::int_remainder);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Rem(desc.clone())),
                ModOps::<B>::new(desc),
            )
            .output()
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(ModOps, B::int_remainder_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::RemScalar(desc.clone()),
                ),
                ModOps::<B>::new(desc),
            )
            .output()
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct ZerosOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.desc.shape.clone();
                let output = B::int_zeros(shape, &self.device, self.desc.dtype.into());
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseInt(BaseOperationIr::Zeros(desc.clone())),
                ZerosOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct OnesOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.desc.shape.clone();
                let output = B::int_ones(shape, &self.device, self.desc.dtype.into());
                handles.register_int_tensor::<B>(&self.desc.id, output);
            }
        }
        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseInt(BaseOperationIr::Ones(desc.clone())),
                OnesOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn int_full(
        shape: Shape,
        fill_value: IntElem<Self>,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct FullOps<B: FusionBackend> {
            out: TensorIr,
            elem: ScalarIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FullOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.out.shape.clone();
                let output =
                    B::int_full(shape, self.elem.elem(), &self.device, self.out.dtype.into());
                handles.register_int_tensor::<B>(&self.out.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = FullOpIr::create(shape, dtype.into(), fill_value, || {
            client.create_empty_handle()
        });

        client
            .register(
                OperationStreams::default(),
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Full(desc.clone())),
                FullOps::<B>::new(desc.out, desc.value, device.clone()),
            )
            .output()
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(SumOps, B::int_sum, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Sum(desc.clone())),
                SumOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_sum_dim(tensor: IntTensor<Self>, axis: usize) -> IntTensor<Self> {
        reduce_int_ops!(SumDimOps, B::int_sum_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), axis, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::SumDim(desc.clone())),
                SumDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(ProdOps, B::int_prod, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Prod(desc.clone())),
                ProdOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(ProdDimOps, B::int_prod_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::ProdDim(desc.clone())),
                ProdDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MeanOps, B::int_mean, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Mean(desc.clone())),
                MeanOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(MeanDimOps, B::int_mean_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::MeanDim(desc.clone())),
                MeanDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CumsumOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumsumOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_cumsum(input, self.desc.axis);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::CumSum(desc.clone())),
                CumsumOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CumprodOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumprodOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_cumprod(input, self.desc.axis);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::CumProd(desc.clone())),
                CumprodOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CumminOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumminOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_cummin(input, self.desc.axis);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::CumMin(desc.clone())),
                CumminOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CummaxOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CummaxOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_cummax(input, self.desc.axis);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::CumMax(desc.clone())),
                CummaxOps::<B>::new(desc),
            )
            .output()
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(ArgMaxOps, B::int_argmax);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::ArgMax(desc.clone())),
                ArgMaxOps::<B>::new(desc),
            )
            .output()
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(ArgMinOps, B::int_argmin);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::ArgMin(desc.clone())),
                ArgMinOps::<B>::new(desc),
            )
            .output()
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct ClampOps<B: FusionBackend> {
            desc: ClampOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.tensor);
                let output = B::int_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ClampOpIr::create(tensor.into_ir(), min, max, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Clamp(desc.clone())),
                ClampOps::<B>::new(desc),
            )
            .output()
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(AbsOps, B::int_abs);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Abs(desc.clone())),
                AbsOps::<B>::new(desc),
            )
            .output()
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct IntoFloatOps<B: FusionBackend> {
            desc: CastOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_into_float(input);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), B::FloatElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::IntoFloat(desc.clone())),
                IntoFloatOps::<B>::new(desc),
            )
            .output()
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::SwapDims(desc.clone())),
                SwapDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MaxOps, B::int_max, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Max(desc.clone())),
                MaxOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(MaxDimOps, B::int_max_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::MaxDim(desc.clone())),
                MaxDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        #[derive(new, Debug)]
        struct MaxDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let desc = ReduceDimWithIndicesOpIr::create(tensor.into_ir(), dim, dtype, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(dtype, NumericOperationIr::MaxDimWithIndices(desc.clone())),
                MaxDimWithIndicesOps::<B>::new(desc),
            )
            .outputs()
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MinOps, B::int_min, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::Min(desc.clone())),
                MinOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(MaxAbsOps, B::int_max_abs, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::MaxAbs(desc.clone())),
                MaxAbsOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(MaxAbsDimOps, B::int_max_abs_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(
                    desc.out.dtype,
                    NumericOperationIr::MaxAbsDim(desc.clone()),
                ),
                MaxAbsDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_int_ops!(MinDimOps, B::int_min_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericInt(desc.out.dtype, NumericOperationIr::MinDim(desc.clone())),
                MinDimOps::<B>::new(desc),
            )
            .output()
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        #[derive(new, Debug)]
        struct MinDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let desc = ReduceDimWithIndicesOpIr::create(tensor.into_ir(), dim, dtype, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericInt(dtype, NumericOperationIr::MinDimWithIndices(desc.clone())),
                MinDimWithIndicesOps::<B>::new(desc),
            )
            .outputs()
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct IntRandomOps<B: FusionBackend> {
            desc: RandomOpIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntRandomOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.desc.out.shape.clone();
                let output = B::int_random(shape, self.desc.distribution, &self.device);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let dtype = IntElem::<Self>::dtype();
        let client = get_client::<B>(device);
        let desc = RandomOpIr::create(shape, dtype, distribution, || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::NumericInt(dtype, NumericOperationIr::IntRandom(desc.clone())),
                IntRandomOps::<B>::new(desc, device.clone()),
            )
            .output()
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_permute(input, self.desc.axes.as_slice());
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Permute(desc.clone())),
                PermuteDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_expand(input, self.desc.out.shape.clone());
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Expand(desc.clone())),
                ExpandOps::<B>::new(desc),
            )
            .output()
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct FlipDimsOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let axes = &self.desc.axes;
                let output = B::int_flip(input, axes);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Flip(desc.clone())),
                FlipDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B>(&self.desc.tensor);

                let output = B::int_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = RepeatDimOpIr::create(tensor.into_ir(), dim, times, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::RepeatDim(desc.clone())),
                RepeatDimOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseAndOps, B::bitwise_and);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseAnd(desc.clone())),
                BitwiseAndOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseAndOps, B::bitwise_and_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseAndScalar(desc.clone())),
                BitwiseAndOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseOrOps, B::bitwise_or);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseOr(desc.clone())),
                BitwiseOrOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseOrOps, B::bitwise_or_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseOrScalar(desc.clone())),
                BitwiseOrOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseXorOps, B::bitwise_xor);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseXor(desc.clone())),
                BitwiseXorOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseXorOps, B::bitwise_xor_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseXorScalar(desc.clone())),
                BitwiseXorOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int_ops!(BitwiseNotOps, B::bitwise_not);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseNot(desc.clone())),
                BitwiseNotOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseLeftShiftOps, B::bitwise_left_shift);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseLeftShift(desc.clone())),
                BitwiseLeftShiftOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseLeftShiftOps, B::bitwise_left_shift_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseLeftShiftScalar(desc.clone())),
                BitwiseLeftShiftOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_ops!(BitwiseRightShiftOps, B::bitwise_right_shift);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseRightShift(desc.clone())),
                BitwiseRightShiftOps::<B>::new(desc),
            )
            .output()
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        scalar_int_ops!(BitwiseRightShiftOps, B::bitwise_right_shift_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Int(IntOperationIr::BitwiseRightShiftScalar(desc.clone())),
                BitwiseRightShiftOps::<B>::new(desc),
            )
            .output()
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: burn_tensor::IntDType) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct CastOps<B: FusionBackend> {
            desc: CastOpIr,
            dtype: burn_tensor::IntDType,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CastOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output: B::IntTensorPrimitive = B::int_cast(input, self.dtype);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), dtype.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Cast(desc.clone())),
                CastOps::<B>::new(desc, dtype),
            )
            .output()
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct UnfoldOps<B: FusionBackend> {
            desc: UnfoldOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for UnfoldOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::int_unfold(input, self.desc.dim, self.desc.size, self.desc.step);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnfoldOpIr::create(tensor.into_ir(), dim, size, step, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Unfold(desc.clone())),
                UnfoldOps::<B>::new(desc),
            )
            .output()
    }
}
