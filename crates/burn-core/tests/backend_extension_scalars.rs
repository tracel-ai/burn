//! Scalar inference, aliases, integer limits, and bindings across repeated calls.
#![cfg(feature = "extension-tests")]
extern crate burn_core as burn;

use burn::backend::fusion::{
    custom::{OperationIr, ScalarIr},
    inspect::FusionInspector,
};

type Count = usize;

#[burn::backend::backend_extension(Cube, Fusion)]
trait ScalarProbe: burn::backend::Backend {
    #[fusion(dtype = x, shape = x)]
    fn scalar_probe(
        x: burn::backend::tensor::FloatTensor<Self>,
        offset: isize,
        #[fusion(scalar)] count: Count,
        enabled: bool,
        epsilon: f32,
    ) -> burn::backend::tensor::FloatTensor<Self>;
}

impl ScalarProbe for burn_cubecl::CubeBackend {
    fn scalar_probe(
        x: burn::backend::tensor::FloatTensor<Self>,
        offset: isize,
        count: Count,
        enabled: bool,
        epsilon: f32,
    ) -> burn::backend::tensor::FloatTensor<Self> {
        let expected = if enabled {
            (isize::MIN, usize::MAX, true, 0.25)
        } else {
            (isize::MAX, 0, false, -0.5)
        };
        assert_eq!((offset, count, enabled, epsilon), expected);
        x
    }
}

#[test]
fn scalars_preserve_types_order_and_values_across_calls() {
    use burn::backend::Dispatch;
    use burn::tensor::{Device, StreamId, Tensor, TensorData};

    let inspector = FusionInspector::install(StreamId::current());
    let device = Device::default();
    for (offset, count, enabled, epsilon) in [
        (isize::MIN, usize::MAX, true, 0.25),
        (isize::MAX, 0, false, -0.5),
    ] {
        let probe = Dispatch::scalar_probe(
            Tensor::<1>::from_floats([1.], &device).into_dispatch(),
            offset,
            count,
            enabled,
            epsilon,
        );
        Tensor::<1>::from_dispatch(probe)
            .into_data()
            .assert_eq(&TensorData::from([1.]), false);
    }

    device.sync().unwrap();
    let scalars: Vec<_> = inspector
        .drain()
        .into_iter()
        .flat_map(|report| report.blocks)
        .flat_map(|block| block.operations)
        .filter_map(|operation| match operation {
            OperationIr::Custom(op) => {
                assert_eq!(op.id, "scalar_probe");
                Some(op.scalars)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        scalars,
        [
            vec![
                ScalarIr::Int(isize::MIN as i64),
                ScalarIr::UInt(usize::MAX as u64),
                ScalarIr::Bool(true),
                ScalarIr::Float(0.25)
            ],
            vec![
                ScalarIr::Int(isize::MAX as i64),
                ScalarIr::UInt(0),
                ScalarIr::Bool(false),
                ScalarIr::Float(-0.5)
            ],
        ]
    );
}
