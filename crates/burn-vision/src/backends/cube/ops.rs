use crate::{
    BoolVisionOps, ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity, FloatVisionOps,
    IntVisionOps, VisionBackend, backends::cpu,
};
use burn_cubecl::{CubeBackend, CubeRuntime};

use burn_core::backend::{
    TensorMetadata, ops::IntTensorOps, tensor::{BoolTensor, IntTensor}
};
use burn_core::tensor::IntDType;

use super::connected_components::hardware_accelerated;

impl<R: CubeRuntime> BoolVisionOps for CubeBackend<R> {
    fn connected_components(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        hardware_accelerated(
            img.clone(),
            ConnectedStatsOptions::none(),
            connectivity,
            out_dtype.into(),
        )
        .map(|it| it.0)
        .unwrap_or_else(|_| {
            let device = &img.device();
            Self::int_from_data(
                cpu::connected_components::<Self>(img, connectivity, out_dtype),
                device,
            )
        })
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
        out_dtype: IntDType,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        let device = &img.device();
        hardware_accelerated::<R>(img.clone(), opts, connectivity, out_dtype.into()).unwrap_or_else(
            |_| {
                let (labels, stats) = cpu::connected_components_with_stats::<Self>(
                    img,
                    connectivity,
                    opts,
                    out_dtype,
                );
                (Self::int_from_data(labels, device), stats)
            },
        )
    }
}

impl<R: CubeRuntime> IntVisionOps for CubeBackend<R> {}
impl<R: CubeRuntime> FloatVisionOps for CubeBackend<R> {}
impl<R: CubeRuntime> VisionBackend for CubeBackend<R> {}

#[cfg(feature = "fusion")]
mod fusion {
    use super::*;
    use burn_core::tensor::Shape;
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, StreamId},
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

            let streams = StreamId::current();
            let out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([height, width]),
                out_dtype.into(),
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
        ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
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
                    let (output, stats) = B1::connected_components_with_stats(
                        input, self.conn, self.opts, self.dtype,
                    );

                    handles.register_int_tensor::<B1>(&labels_ir.id, output);
                    handles.register_int_tensor::<B1>(&area_ir.id, stats.area);
                    handles.register_int_tensor::<B1>(&left_ir.id, stats.left);
                    handles.register_int_tensor::<B1>(&top_ir.id, stats.top);
                    handles.register_int_tensor::<B1>(&right_ir.id, stats.right);
                    handles.register_int_tensor::<B1>(&bottom_ir.id, stats.bottom);
                    handles.register_int_tensor::<B1>(&max_label_ir.id, stats.max_label);
                }
            }

            let dtype = out_dtype.into();
            let shape = Shape::new([height, width]);
            let shape_flat = shape.clone().flatten();
            let streams = StreamId::current();
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

            let stats = ConnectedStatsPrimitive {
                area,
                left,
                top,
                right,
                bottom,
                max_label,
            };
            (out, stats)
        }
    }
    impl<B: FusionBackend + IntVisionOps> IntVisionOps for Fusion<B> {}
    impl<B: FusionBackend + FloatVisionOps> FloatVisionOps for Fusion<B> {}
    impl<B: FusionBackend + VisionBackend> VisionBackend for Fusion<B> {}
}
