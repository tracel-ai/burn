use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::TensorData;

#[test]
fn smaller_gather_indices_with_strided_operands_fuse() {
    test_stream().executes(|| {
        let device = Default::default();
        let tensor = TestTensor::<3>::from_data(
            TensorData::new((0..60).map(|i| i as f32).collect(), [5, 4, 3]),
            &device,
        )
        .swap_dims(0, 2);
        let indices = TestTensorInt::<3>::from_data(
            [
                [[2, 0], [3, 1]],
                [[1, 3], [0, 2]],
                [[3, 2], [1, 0]],
                [[0, 1], [2, 3]],
            ],
            &device,
        )
        .swap_dims(0, 2);
        device.sync().unwrap();

        let inspector = FusionInspector::install(burn_tensor::StreamId::current());
        let out = tensor.gather(1, indices).add_scalar(10).into_data();

        out.assert_eq(
            &TensorData::from([
                [[16., 25., 43., 46.], [19., 22., 37., 52.]],
                [[11., 32., 41., 50.], [14., 29., 35., 56.]],
            ]),
            false,
        );
        assert_gather_and_add_fused(&inspector);
    });
}

#[test]
fn smaller_gather_indices_on_last_axis_fuse() {
    test_stream().executes(|| {
        let device = Default::default();
        let tensor = TestTensor::<3>::from_data(
            TensorData::new((0..60).map(|i| i as f32).collect(), [3, 4, 5]),
            &device,
        );
        let indices = TestTensorInt::<3>::from_data(
            [[[4, 1, 3, 0], [2, 4, 0, 1]], [[1, 3, 2, 4], [4, 0, 1, 3]]],
            &device,
        );
        device.sync().unwrap();

        let inspector = FusionInspector::install(burn_tensor::StreamId::current());
        let out = tensor.gather(2, indices).add_scalar(10).into_data();

        out.assert_eq(
            &TensorData::from([
                [[14., 11., 13., 10.], [17., 19., 15., 16.]],
                [[31., 33., 32., 34.], [39., 35., 36., 38.]],
            ]),
            false,
        );
        assert_gather_and_add_fused(&inspector);
    });
}

#[test]
fn smaller_gather_indices_broadcast_in_fused_epilogue() {
    test_stream().executes(|| {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [
                [10., 11., 12., 13.],
                [20., 21., 22., 23.],
                [30., 31., 32., 33.],
            ],
            &device,
        );
        let indices = TestTensorInt::<2>::from_data([[2], [0]], &device);
        let bias = TestTensor::<2>::from_data([[1., 2., 3., 4.], [5., 6., 7., 8.]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(burn_tensor::StreamId::current());
        let out = (tensor.gather(0, indices) + bias).into_data();

        out.assert_eq(
            &TensorData::from([[31., 32., 33., 34.], [15., 16., 17., 18.]]),
            false,
        );
        assert_gather_and_add_fused(&inspector);
    });
}

fn assert_gather_and_add_fused(inspector: &FusionInspector) {
    let reports = inspector.drain();
    assert!(
        reports
            .iter()
            .flat_map(|report| report.fused_blocks())
            .any(|block| block.operations.len() == 2),
        "gather and add should share a fused block: {reports:#?}",
    );
}
