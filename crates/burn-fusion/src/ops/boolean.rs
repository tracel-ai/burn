use crate::{
    Fusion, FusionBackend, get_client,
    stream::{OperationStreams, execution::Operation},
};
use burn_backend::{
    Element, ExecutionError, Shape, Slice, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolElem, BoolTensor, Device, FloatTensor, IndexingUpdateOp, IntTensor},
};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, BoolOperationIr, CastOpIr, CatOpIr, CreationOpIr, FlipOpIr,
    GatherOpIr, HandleContainer, InitOperationIr, MaskFillOpIr, MaskWhereOpIr, OperationIr,
    OperationOutput, PermuteOpIr, RepeatDimOpIr, ScalarIr, ScalarOpIr, ScatterOpIr, ShapeOpIr,
    SliceAssignOpIr, SliceOpIr, SwapDimsOpIr, TensorIr, UnaryOpIr, UnfoldOpIr,
};
use std::marker::PhantomData;

use super::NoOp;

impl<B: FusionBackend> BoolTensorOps<Self> for Fusion<B> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct EmptyOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EmptyOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output = B::bool_empty(self.desc.shape.clone(), &self.device);
                handles.register_bool_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc =
            CreationOpIr::create(shape, B::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseBool(BaseOperationIr::Empty(desc.clone())),
                EmptyOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct ZerosOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output = B::bool_zeros(self.desc.shape.clone(), &self.device);
                handles.register_bool_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc =
            CreationOpIr::create(shape, B::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseBool(BaseOperationIr::Zeros(desc.clone())),
                ZerosOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn bool_ones(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct OnesOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output = B::bool_ones(self.desc.shape.clone(), &self.device);
                handles.register_bool_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc =
            CreationOpIr::create(shape, B::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseBool(BaseOperationIr::Ones(desc.clone())),
                OnesOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.bool_into_data::<B>().await
    }

    fn bool_from_data(data: burn_backend::TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<B>(device);
        let tensor = B::bool_from_data(data, device);
        let shape = burn_backend::TensorMetadata::shape(&tensor);

        let handle = B::bool_tensor_handle(tensor);
        let desc = InitOperationIr::create(shape, B::BoolElem::dtype(), || {
            client.register_tensor_handle(handle)
        });

        client
            .register(
                OperationStreams::default(),
                OperationIr::Init(desc),
                NoOp::<B>::new(),
            )
            .output()
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct IntoIntOps<B: FusionBackend> {
            desc: CastOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_into_int(input);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), B::IntElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Bool(BoolOperationIr::IntoInt(desc.clone())),
                IntoIntOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct IntoFloatOps<B: FusionBackend> {
            desc: CastOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_into_float(input);
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
                OperationIr::Bool(BoolOperationIr::IntoFloat(desc.clone())),
                IntoFloatOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        let device_original: &B::Device = tensor.client.device();

        if device_original == device {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(device);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_bool::<B>(tensor.into_ir(), client_target, id)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
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
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_reshape(input, self.desc.out.shape.clone());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::Reshape(desc.clone())),
                ReshapeDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);

                let output = B::bool_slice(tensor, self.desc.ranges.as_slice());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::Slice(desc.clone())),
                SliceOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let value = handles.get_bool_tensor::<B>(&self.desc.value);

                let output = B::bool_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc.clone())),
                SliceAssignOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
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
                    .map(|tensor| handles.get_bool_tensor::<B>(tensor))
                    .collect();

                let output = B::bool_cat(tensors, self.desc.dim);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs(&tensors);

        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::Cat(desc.clone())),
                CatOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct EqualOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EqualOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_equal(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::Equal(desc.clone())),
                EqualOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct NotOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for NotOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_not(input);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Bool(BoolOperationIr::Not(desc.clone())),
                NotOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct AndOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for AndOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_and(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Bool(BoolOperationIr::And(desc.clone())),
                AndOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct OrOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OrOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_or(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });
        client
            .register(
                streams,
                OperationIr::Bool(BoolOperationIr::Or(desc.clone())),
                OrOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::SwapDims(desc.clone())),
                SwapDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_permute(input, self.desc.axes.as_slice());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_expand(input, self.desc.out.shape.clone());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::Expand(desc.clone())),
                ExpandOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct FlipOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_flip(input, self.desc.axes.as_slice());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::Flip(desc.clone())),
                FlipOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);

                let output = B::bool_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::RepeatDim(desc.clone())),
                RepeatDimOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct UnfoldOps<B: FusionBackend> {
            desc: UnfoldOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for UnfoldOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_unfold(input, self.desc.dim, self.desc.size, self.desc.step);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::Unfold(desc.clone())),
                UnfoldOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct MaskWhereOps<B: FusionBackend> {
            desc: MaskWhereOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let value = handles.get_bool_tensor::<B>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::bool_mask_where(tensor, mask, value);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::MaskWhere(desc.clone())),
                MaskWhereOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolElem<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct MaskFillOps<B: FusionBackend> {
            desc: MaskFillOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::bool_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &mask]);

        let client = tensor.client.clone();
        let value = ScalarIr::with_dtype(value, &tensor.dtype);
        let desc = MaskFillOpIr::create(tensor.into_ir(), mask.into_ir(), value, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::MaskFill(desc.clone())),
                MaskFillOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::bool_gather(self.desc.dim, tensor, indices);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseBool(BaseOperationIr::Gather(desc.clone())),
                GatherOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct ScatterOps<B: FusionBackend> {
            desc: ScatterOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_bool_tensor::<B>(&self.desc.value);

                let output = B::bool_scatter_or(self.desc.dim, tensor, indices, value);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);

        let client = tensor.client.clone();
        let desc = ScatterOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            IndexingUpdateOp::Add,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::Scatter(desc.clone())),
                ScatterOps::<B>::new(desc),
            )
            .output()
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: BoolElem<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct EqualElemOps<B: FusionBackend> {
            desc: ScalarOpIr,
            _b: PhantomData<B>,
        }
        impl<B: FusionBackend> Operation<B::FusionRuntime> for EqualElemOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let output = B::bool_equal_elem(lhs, self.desc.rhs.elem());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseBool(BaseOperationIr::EqualElem(desc.clone())),
                EqualElemOps::<B>::new(desc),
            )
            .output()
    }
}
