use super::*;
use burn_backend::ops::{FloatTensorOps, ModuleOps};
use burn_dispatch::Dispatch;
use burn_fusion::{custom::OperationIr, inspect::FusionInspector};
use burn_tensor::TensorData;

#[test]
fn transposed_convolution_bias_backward_forwards_to_backend() {
    for spatial_dims in 1..=3 {
        let stream = test_stream();
        stream.executes(|| {
            let device = Default::default();
            let grad = TestTensor::<3>::from_floats(
                [
                    [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]],
                    [[7.0, 8.0], [-9.0, 10.0], [11.0, -12.0]],
                ],
                &device,
            );
            let dtype = grad.dtype();
            let mut shape = vec![2, 3];
            shape.extend(vec![1; spatial_dims - 1]);
            shape.push(2);
            let grad = Dispatch::float_reshape(grad.into_dispatch(), shape.clone().into());
            shape[1] = 1;
            let input = Dispatch::float_zeros(shape.into(), device.as_dispatch(), dtype.into());
            let bias = TestTensor::<1>::zeros([3], &device).into_dispatch();
            device.sync().unwrap();

            let inspector = FusionInspector::install(stream);
            let backward = match spatial_dims {
                1 => Dispatch::conv_transpose1d_bias_backward,
                2 => Dispatch::conv_transpose2d_bias_backward,
                3 => Dispatch::conv_transpose3d_bias_backward,
                _ => unreachable!(),
            };
            let result = TestTensor::<1>::from_dispatch(backward(input, bias, grad));
            assert_eq!(result.dtype(), dtype);
            assert_eq!(result.device(), device);
            result
                .into_data()
                .assert_eq(&TensorData::from([14.0, 8.0, 0.0]), false);
            device.sync().unwrap();

            let reports = inspector.drain();
            assert!(
                reports
                    .iter()
                    .flat_map(|report| &report.blocks)
                    .flat_map(|block| &block.operations)
                    .any(|operation| matches!(operation, OperationIr::Module(_))),
                "expected a forwarded module operation for {spatial_dims}D bias backward; got {reports:#?}"
            );
        });
    }
}
