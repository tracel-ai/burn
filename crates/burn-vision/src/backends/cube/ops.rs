use crate::{
    BoolVisionOps, ConnectedStatsOptions, Connectivity, FloatVisionOps, IntVisionOps,
    VisionBackend, backends::cpu, dispatch_int_dtype,
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use burn_core::tensor::{
    Element, IntDType,
    backend::{
        BoolTensor, IntTensor,
        ops::{BoolTensorOps, IntTensorOps},
    },
};

use super::connected_components::hardware_accelerated;

impl<R, F, I, BT> BoolVisionOps for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn connected_components(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        dispatch_int_dtype!(out_dtype, |I| hardware_accelerated::<R, F, I, BT>(
            img.clone(),
            ConnectedStatsOptions::none(),
            connectivity,
        )
        .map(|it| it.0)
        .unwrap_or_else(|_| {
            let device = Self::bool_device(&img);
            Self::int_from_data(
                cpu::connected_components::<Self>(img, connectivity, out_dtype),
                &device,
            )
        }))
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
        out_dtype: IntDType,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
    ) {
        let device = Self::bool_device(&img);
        dispatch_int_dtype!(out_dtype, |I| hardware_accelerated::<R, F, I, BT>(
            img.clone(),
            opts,
            connectivity
        )
        .unwrap_or_else(|_| {
            let (labels, (area, top, left, right, bottom, max_label)) =
                cpu::connected_components_with_stats::<Self>(img, connectivity, opts, out_dtype);
            (
                Self::int_from_data(labels, &device),
                area,
                top,
                left,
                right,
                bottom,
                max_label,
            )
        }))
    }
}

impl<R, F, I, BT> IntVisionOps for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
impl<R, F, I, BT> FloatVisionOps for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
impl<R, F, I, BT> VisionBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}

#[cfg(feature = "fusion")]
mod fusion {
    use super::*;
    use burn_core::tensor::Shape;
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

    impl<B: FusionBackend + BoolVisionOps> BoolVisionOps for Fusion<B> {
        fn connected_components(
            img: BoolTensor<Self>,
            conn: Connectivity,
            out_dtype: IntDType,
        ) -> IntTensor<Self> {
            let height = img.shape[0];
            let width = img.shape[1];
            let client = img.client.clone();

            #[derive(derive_new::new, Clone, Debug)]
            struct ConnComp<B> {
                desc: CustomOpIr,
                conn: Connectivity,
                dtype: IntDType,
                _b: core::marker::PhantomData<B>,
            }

            impl<B1: FusionBackend + BoolVisionOps> Operation<B1::FusionRuntime> for ConnComp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([img], [labels]) = self.desc.as_fixed();
                    let input = handles.get_bool_tensor::<B1>(img);
                    let output = B1::connected_components(input, self.conn, self.dtype);

                    handles.register_int_tensor::<B1>(&labels.id, output);
                }
            }

            let streams = OperationStreams::with_inputs([&img]);
            let out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([height, width]),
                B::IntElem::dtype(),
            );

            let desc = CustomOpIr::new("connected_components", &[img.into_ir()], &[out]);
            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    ConnComp::<B>::new(desc, conn, out_dtype),
                )
                .output()
        }

        fn connected_components_with_stats(
            img: BoolTensor<Self>,
            conn: Connectivity,
            opts: ConnectedStatsOptions,
            out_dtype: IntDType,
        ) -> (
            IntTensor<Self>,
            IntTensor<Self>,
            IntTensor<Self>,
            IntTensor<Self>,
            IntTensor<Self>,
            IntTensor<Self>,
            IntTensor<Self>,
        ) {
            let height = img.shape[0];
            let width = img.shape[1];
            let client = img.client.clone();

            #[derive(derive_new::new, Clone, Debug)]
            struct ConnCompStats<B> {
                desc: CustomOpIr,
                conn: Connectivity,
                opts: ConnectedStatsOptions,
                dtype: IntDType,
                _b: core::marker::PhantomData<B>,
            }

            impl<B1: FusionBackend + BoolVisionOps> Operation<B1::FusionRuntime> for ConnCompStats<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [img],
                        [
                            labels_ir,
                            area_ir,
                            left_ir,
                            top_ir,
                            right_ir,
                            bottom_ir,
                            max_label_ir,
                        ],
                    ) = self.desc.as_fixed();
                    let input = handles.get_bool_tensor::<B1>(img);
                    let (output, area, left, top, right, bottom, max_label) =
                        B1::connected_components_with_stats(
                            input, self.conn, self.opts, self.dtype,
                        );

                    handles.register_int_tensor::<B1>(&labels_ir.id, output);
                    handles.register_int_tensor::<B1>(&area_ir.id, area);
                    handles.register_int_tensor::<B1>(&left_ir.id, left);
                    handles.register_int_tensor::<B1>(&top_ir.id, top);
                    handles.register_int_tensor::<B1>(&right_ir.id, right);
                    handles.register_int_tensor::<B1>(&bottom_ir.id, bottom);
                    handles.register_int_tensor::<B1>(&max_label_ir.id, max_label);
                }
            }

            let dtype = B::IntElem::dtype();
            let shape = Shape::new([height, width]);
            let shape_flat = shape.clone().flatten();
            let streams = OperationStreams::with_inputs([&img]);
            let out = TensorIr::uninit(client.create_empty_handle(), shape.clone(), dtype);
            let area = TensorIr::uninit(client.create_empty_handle(), shape_flat.clone(), dtype);
            let left = TensorIr::uninit(client.create_empty_handle(), shape_flat.clone(), dtype);
            let top = TensorIr::uninit(client.create_empty_handle(), shape_flat.clone(), dtype);
            let right = TensorIr::uninit(client.create_empty_handle(), shape_flat.clone(), dtype);
            let bottom = TensorIr::uninit(client.create_empty_handle(), shape_flat, dtype);
            let max_label = TensorIr::uninit(client.create_empty_handle(), [1].into(), dtype);

            let desc = CustomOpIr::new(
                "connected_components",
                &[img.into_ir()],
                &[out, area, left, top, right, bottom, max_label],
            );
            let [out, area, left, top, right, bottom, max_label] = client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    ConnCompStats::<B>::new(desc, conn, opts, out_dtype),
                )
                .try_into()
                .unwrap();

            (out, area, left, top, right, bottom, max_label)
        }
    }
    impl<B: FusionBackend + IntVisionOps> IntVisionOps for Fusion<B> {}
    impl<B: FusionBackend + FloatVisionOps> FloatVisionOps for Fusion<B> {}
    impl<B: FusionBackend + VisionBackend> VisionBackend for Fusion<B> {}
}
