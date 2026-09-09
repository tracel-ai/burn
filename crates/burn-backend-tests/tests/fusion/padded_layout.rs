use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector};
use burn_tensor::{TensorData, s};

/// Exercise the layout vote and strided reads with storage whose padding does
/// not depend on the backend allocator. The output planner tests separately
/// verify the chosen strides and that padded reads cannot use `SameAsRef`.
#[test]
fn padded_channels_last_input_fuses_and_computes_correctly() {
    test_stream().executes(|| {
        let device = Default::default();
        // Cover both an odd channel count and the vectorizable 48-channel case.
        for channels in [3, 48] {
            let [batch, height, width] = [2, 3, 5];
            // Removing 128 elements preserves 256-byte alignment for both the
            // f32 and f16 suites, allowing a metadata-only slice.
            let pitch = channels + 128;
            let mut storage = vec![-10000.0f32; batch * height * width * pitch];
            let mut expected = vec![0.0f32; batch * channels * height * width];
            for n in 0..batch {
                for h in 0..height {
                    for w in 0..width {
                        for c in 0..channels {
                            let value = ((n * height + h) * width + w) as f32 + c as f32;
                            storage[((n * height + h) * width + w) * pitch + c] = value;
                            expected[((n * channels + c) * height + h) * width + w] =
                                (value + c as f32) * 2.0;
                        }
                    }
                }
            }
            let input = TestTensor::<4>::from_data(
                TensorData::new(storage, [batch, height, width, pitch]),
                &device,
            )
            .slice(s![.., .., .., 0..channels])
            .permute([0, 3, 1, 2]);
            // Materialize the view before the elementwise chain is traced.
            let _ = input.clone().into_data();

            let bias = TestTensor::<4>::from_data(
                TensorData::new(
                    (0..channels).map(|c| c as f32).collect::<Vec<_>>(),
                    [1, channels, 1, 1],
                ),
                &device,
            );
            device.sync().unwrap();
            let inspector = FusionInspector::install(burn_tensor::StreamId::current());
            let output = (input.clone() + bias).mul_scalar(2.0);
            output.into_data().assert_eq(
                &TensorData::new(expected, [batch, channels, height, width]),
                false,
            );
            let reports = inspector.drain();
            assert!(
                !reports.is_empty()
                    && reports
                        .iter()
                        .flat_map(|report| &report.blocks)
                        .all(|block| { matches!(block.kind, BlockKind::Fused { .. }) }),
                "the elementwise chain must execute through fusion: {reports:?}",
            );
        }
    });
}
