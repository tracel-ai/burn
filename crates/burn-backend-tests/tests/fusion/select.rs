//! Tests that a fused `select` reads its input at the coordinates of its own output.

use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector};
use burn_ir::{BaseOperationIr, NumericOperationIr, OperationIr};
use burn_tensor::TensorData;

/// A `select` followed by a wider element-wise op in the same block: the wider op must not
/// widen the block's reference past the `select`, whose layout it computes its coordinates from.
#[test]
fn select_then_wider_elementwise_reads_the_selected_index() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let tensor = TestTensorInt::<2>::from_data([[10, 20, 30, 40]], &device).add_scalar(1);
        let indices = TestTensorInt::<1>::from_data([2], &device);
        device.sync().unwrap();

        let selected = tensor.clone().select(1, indices);
        let doubled = tensor.mul_scalar(2);
        let out = TestTensorInt::cat(vec![selected, doubled], 1).into_data();

        assert_eq!(
            out,
            TensorData::from([[31, 22, 42, 62, 82]]).convert_dtype(out.dtype())
        );
    });
}

#[test]
fn singleton_select_is_independent_of_unrelated_wide_operations() {
    let stream = test_stream();
    stream.executes(|| {
        let device = burn_tensor::Device::default().autodiff();
        let inspector = FusionInspector::install(stream);

        for large_first in [false, true] {
            let input = TestTensor::<1>::from_data(
                TensorData::new((0..32).map(|index| index as f32).collect(), [32]),
                &device,
            )
            .require_grad();
            let mut index_storage = vec![19; 65_536];
            index_storage[0] = 7;
            let indices =
                TestTensorInt::<1>::from_data(TensorData::new(index_storage, [65_536]), &device)
                    .slice([0..1]);
            let scalar = TestTensor::<1>::from_data([100.0], &device);
            let large =
                TestTensor::<1>::from_data(TensorData::new(vec![-1.0; 65_536], [65_536]), &device)
                    .require_grad();
            let expression = || large.clone().mul_scalar(2.0).add_scalar(1.0).square();

            let pending = if large_first {
                Some(expression())
            } else {
                None
            };
            let selected = scalar.sub(input.select(0, indices));
            let unrelated = pending.unwrap_or_else(expression);
            let _ = unrelated.into_data();

            assert_eq!(selected.into_scalar::<f32>(), 93.0);
            device.sync().unwrap();

            let reports = inspector.drain();
            let report = reports
                .iter()
                .find(|report| {
                    let has_select = report.blocks.iter().any(|block| {
                        block.operations.iter().any(|operation| {
                            matches!(
                                operation,
                                OperationIr::BaseFloat(BaseOperationIr::Select(_))
                            )
                        })
                    });
                    let has_wide_ops = report.blocks.iter().any(|block| {
                        block.operations.iter().any(|operation| {
                            matches!(
                                operation,
                                OperationIr::NumericFloat(_, NumericOperationIr::MulScalar(_))
                            )
                        })
                    });

                    has_select && has_wide_ops
                })
                .expect("select and unrelated wide operations missing from same fusion trace");
            let select_blocks = report
                .blocks
                .iter()
                .filter(|block| {
                    block.operations.iter().any(|operation| {
                        matches!(
                            operation,
                            OperationIr::BaseFloat(BaseOperationIr::Select(_))
                        )
                    })
                })
                .collect::<Vec<_>>();
            let wide_blocks = report
                .blocks
                .iter()
                .filter(|block| {
                    block.operations.iter().any(|operation| {
                        matches!(
                            operation,
                            OperationIr::NumericFloat(_, NumericOperationIr::MulScalar(_))
                        )
                    })
                })
                .collect::<Vec<_>>();

            assert!(!select_blocks.is_empty());
            assert!(
                select_blocks
                    .iter()
                    .all(|block| matches!(block.kind, BlockKind::Fused { .. }))
            );
            assert!(!wide_blocks.is_empty());
            assert!(
                wide_blocks
                    .iter()
                    .all(|block| matches!(block.kind, BlockKind::Fused { .. }))
            );
            assert!(select_blocks.iter().all(|block| {
                !block.operations.iter().any(|operation| {
                    matches!(
                        operation,
                        OperationIr::NumericFloat(_, NumericOperationIr::MulScalar(_))
                    )
                })
            }));
        }
    });
}
