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
        client::FusionClient,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr};
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

            let mut streams = OperationStreams::default();
            streams.tensor(&img);
            let out = client.tensor_uninitialized(Shape::new([height, width]), B::IntElem::dtype());

            let desc =
                CustomOpIr::new("connected_components", &[img.into_ir()], &[out.to_ir_out()]);
            client.register(
                streams,
                OperationIr::Custom(desc.clone()),
                ConnComp::<B>::new(desc, conn),
            );

            out
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

            let mut streams = OperationStreams::default();
            streams.tensor(&img);
            let out = client.tensor_uninitialized(Shape::new([height, width]), B::IntElem::dtype());
            let area =
                client.tensor_uninitialized(Shape::new([height * width]), B::IntElem::dtype());
            let left =
                client.tensor_uninitialized(Shape::new([height * width]), B::IntElem::dtype());
            let top =
                client.tensor_uninitialized(Shape::new([height * width]), B::IntElem::dtype());
            let right =
                client.tensor_uninitialized(Shape::new([height * width]), B::IntElem::dtype());
            let bottom =
                client.tensor_uninitialized(Shape::new([height * width]), B::IntElem::dtype());
            let max_label = client.tensor_uninitialized(Shape::new([1]), B::IntElem::dtype());

            let desc = CustomOpIr::new(
                "connected_components",
                &[img.into_ir()],
                &[
                    out.to_ir_out(),
                    area.to_ir_out(),
                    left.to_ir_out(),
                    top.to_ir_out(),
                    right.to_ir_out(),
                    bottom.to_ir_out(),
                    max_label.to_ir_out(),
                ],
            );
            client.register(
                streams,
                OperationIr::Custom(desc.clone()),
                ConnCompStats::<B>::new(desc, conn, opts),
            );

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
