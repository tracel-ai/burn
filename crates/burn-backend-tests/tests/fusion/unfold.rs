use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::TensorData;

#[test]
fn elementwise_is_unchanged_by_unfold() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let values = (1..=34).map(|value| value as f32).collect::<Vec<_>>();
        let input = TestTensor::<3>::from_data(TensorData::new(values, [2, 17, 1]), &device);
        let expected = ((input.clone() + 1.0) * 2.0)
            .unfold::<4, _>(1, 8, 8)
            .into_data();

        let inspector = FusionInspector::install(stream);
        let unfolded = input.unfold::<4, _>(1, 8, 8);
        ((unfolded + 1.0) * 2.0)
            .into_data()
            .assert_eq(&expected, true);

        assert!(
            inspector
                .drain()
                .iter()
                .flat_map(|report| report.fused_blocks())
                .any(|block| block.fuser_name() == Some("ElementWise")),
            "expected elementwise fusion",
        );
    });
}
