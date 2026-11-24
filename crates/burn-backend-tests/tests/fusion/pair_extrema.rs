use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::{TensorData, Tolerance};

fn assert_pair_fuses(pair: fn(TestTensor<1>, TestTensor<1>) -> TestTensor<1>, expected: [f32; 6]) {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let lhs = TestTensor::from_data([1., 5., f32::NAN, 4., f32::INFINITY, -3.], &device);
        let rhs = TestTensor::from_data([2., 3., 4., f32::NAN, 1., -3.], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        pair(lhs, rhs)
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from(expected), Tolerance::absolute(0.));

        let reports = inspector.drain();
        assert_eq!(reports.len(), 1, "{reports:#?}");
        reports[0].assert_single_fused_block();
    });
}

#[test]
fn max_pair_fuses() {
    assert_pair_fuses(
        TestTensor::max_pair,
        [2., 5., f32::NAN, f32::NAN, f32::INFINITY, -3.],
    );
}

#[test]
fn min_pair_fuses() {
    assert_pair_fuses(TestTensor::min_pair, [1., 3., f32::NAN, f32::NAN, 1., -3.]);
}
