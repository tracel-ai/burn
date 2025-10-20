use crate::{
    BoolVisionOps, ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity, FloatVisionOps,
    IntVisionOps, QVisionOps, VisionBackend, backends::cpu,
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use burn_tensor::{
    Element,
    ops::{BoolTensor, IntTensor},
};

use super::connected_components::hardware_accelerated;

impl<R, F, I, BT> BoolVisionOps for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn connected_components(img: BoolTensor<Self>, connectivity: Connectivity) -> IntTensor<Self> {
        hardware_accelerated::<R, F, I, BT>(
            img.clone(),
            ConnectedStatsOptions::none(),
            connectivity,
        )
        .map(|it| it.0)
        .unwrap_or_else(|_| cpu::connected_components::<Self>(img, connectivity))
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        hardware_accelerated::<R, F, I, BT>(img.clone(), opts, connectivity).unwrap_or_else(|_| {
            cpu::connected_components_with_stats::<Self>(img, connectivity, opts)
        })
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
impl<R, F, I, BT> QVisionOps for CubeBackend<R, F, I, BT>
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
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        client::OperationOutput,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};
    use burn_tensor::Shape;

    impl<B: FusionBackend + BoolVisionOps> BoolVisionOps for Fusion<B> {
        fn connected_components(img: BoolTensor<Self>, conn: Connectivity) -> IntTensor<Self> {
            let height = img.shape[0];
            let width = img.shape[1];
            let client = img.client.clone();

            #[derive(derive_new::new, Clone, Debug)]
            struct ConnComp<B> {
                desc: CustomOpIr,
                conn: Connectivity,
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
                    let output = B1::connected_components(input, self.conn);

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
                    ConnComp::<B>::new(desc, conn),
                )
                .output()
        }

        fn connected_components_with_stats(
            img: BoolTensor<Self>,
            conn: Connectivity,
            opts: ConnectedStatsOptions,
        ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
            let height = img.shape[0];
            let width = img.shape[1];
            let client = img.client.clone();

            #[derive(derive_new::new, Clone, Debug)]
            struct ConnCompStats<B> {
                desc: CustomOpIr,
                conn: Connectivity,
                opts: ConnectedStatsOptions,
                _b: core::marker::PhantomData<B>,
            }

            impl<B1: FusionBackend + BoolVisionOps> Operation<B1::FusionRuntime> for ConnCompStats<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([img], [labels, area, left, top, right, bottom, max_label]) =
                        self.desc.as_fixed();
                    let input = handles.get_bool_tensor::<B1>(img);
                    let (output, stats) =
                        B1::connected_components_with_stats(input, self.conn, self.opts);

                    handles.register_int_tensor::<B1>(&labels.id, output);
                    handles.register_int_tensor::<B1>(&area.id, stats.area);
                    handles.register_int_tensor::<B1>(&left.id, stats.left);
                    handles.register_int_tensor::<B1>(&top.id, stats.top);
                    handles.register_int_tensor::<B1>(&right.id, stats.right);
                    handles.register_int_tensor::<B1>(&bottom.id, stats.bottom);
                    handles.register_int_tensor::<B1>(&max_label.id, stats.max_label);
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
                    ConnCompStats::<B>::new(desc, conn, opts),
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
    impl<B: FusionBackend + QVisionOps> QVisionOps for Fusion<B> {}
    impl<B: FusionBackend + VisionBackend> VisionBackend for Fusion<B> {}
}
