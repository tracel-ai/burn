use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector};

/// An upload is on the device as soon as it is created, so it must not cut the fused block the
/// computation around it is accumulating. More uploads than the fusion search has blocks
/// (`max_blocks`, 5 by default), since a queued upload would take a block of its own.
#[test]
fn uploads_mid_computation_do_not_split_the_fused_block() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let x = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let mut y = x.mul_scalar(2.0).exp();
        for _ in 0..8 {
            y = y + TestTensor::<1>::from_data([1.0, 1.0, 1.0, 1.0], &device);
        }
        y.into_data();
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");
        let blocks = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 1, "the uploads split the block\n\n{tables}");
        assert!(
            matches!(blocks[0].kind, BlockKind::Fused { .. }),
            "expected a fused block\n\n{tables}"
        );
    });
}
