use burn_ir::{BaseOperationIr, GraphIr, OperationIr, TensorId};

use crate::{ExportError, GraphStructureValidator};

/// Annotation for one runtime input axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSpec {
    /// The dimension is fixed to the captured value.
    Static,
    /// The dimension is runtime-variable and carries a validation value.
    Dynamic {
        /// ONNX symbolic dimension name.
        symbol: String,
        /// Dimension used for the second validation capture.
        validation_value: usize,
    },
}

/// Shape annotations for one graph input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    /// Tensor being annotated.
    pub tensor: TensorId,
    /// Axis annotations in tensor order.
    pub axes: Vec<AxisSpec>,
}

/// An explicit ONNX-compatible dimension expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    /// Constant dimension.
    Static(usize),
    /// Dimension of a declared runtime input.
    InputDim { input: TensorId, axis: usize },
    /// Dimension of an intermediate/source tensor.
    TensorDim { tensor: TensorId, axis: usize },
    /// Element-count-preserving inferred dimension (`-1` in ONNX reshape).
    Infer,
}

/// Resolved shape operand for an operation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedShape {
    /// Operation index in [`GraphIr::operations`].
    pub operation: usize,
    /// Output tensor receiving the shape.
    pub tensor: TensorId,
    /// Dimension expressions in axis order.
    pub dimensions: Vec<ShapeExpr>,
}

/// Graph plus every explicit shape expression needed during ONNX lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedExportGraph {
    /// Validated captured graph.
    pub graph: GraphIr,
    /// Resolved shape-sensitive operands.
    pub shapes: Vec<ResolvedShape>,
}

/// Replaceable shape-resolution stage used before ONNX lowering.
pub trait ShapeResolver {
    /// Resolve shape-sensitive operands in a captured graph.
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError>;
}

/// Resolves every captured reshape dimension as a constant.
pub struct StaticShapeResolver<'a> {
    /// Captured graph.
    pub graph: &'a GraphIr,
}

impl ShapeResolver for StaticShapeResolver<'_> {
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError> {
        Ok(ResolvedExportGraph {
            graph: self.graph.clone(),
            shapes: reshapes(self.graph)
                .map(|(operation, _, output)| ResolvedShape {
                    operation,
                    tensor: output.id,
                    dimensions: output
                        .shape
                        .iter()
                        .copied()
                        .map(ShapeExpr::Static)
                        .collect(),
                })
                .collect(),
        })
    }
}

/// Conservative shape resolver using two structurally validated captures.
pub struct PairedTraceShapeResolver<'a> {
    /// Primary capture.
    pub sample: &'a GraphIr,
    /// Capture produced with validation input dimensions.
    pub validation: &'a GraphIr,
    /// Explicit dynamic input annotations.
    pub inputs: &'a [InputSpec],
}

impl ShapeResolver for PairedTraceShapeResolver<'_> {
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError> {
        GraphStructureValidator::validate(self.sample, self.validation)?;
        let validation_reshapes: Vec<_> = reshapes(self.validation).collect();
        let mut shapes = Vec::new();
        for ((operation, source, output), (_, validation_source, validation_output)) in
            reshapes(self.sample).zip(validation_reshapes)
        {
            let mut dimensions = Vec::new();
            let mut unresolved = Vec::new();
            for (axis, (&sample_dim, &validation_dim)) in output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                if sample_dim == validation_dim {
                    dimensions.push(ShapeExpr::Static(sample_dim));
                    continue;
                }
                let mut candidates = Vec::new();
                for spec in self.inputs {
                    let validation_id = self
                        .sample
                        .inputs
                        .iter()
                        .position(|id| *id == spec.tensor)
                        .and_then(|position| self.validation.inputs.get(position))
                        .copied();
                    if let (Some(sample_input), Some(validation_input)) = (
                        tensor(self.sample, spec.tensor),
                        validation_id.and_then(|id| tensor(self.validation, id)),
                    ) {
                        for (input_axis, axis_spec) in spec.axes.iter().enumerate() {
                            if matches!(axis_spec, AxisSpec::Dynamic { .. })
                                && sample_input.shape.get(input_axis) == Some(&sample_dim)
                                && validation_input.shape.get(input_axis) == Some(&validation_dim)
                            {
                                candidates.push(ShapeExpr::InputDim {
                                    input: spec.tensor,
                                    axis: input_axis,
                                });
                            }
                        }
                    }
                }
                if candidates.is_empty() {
                    for (source_axis, (&a, &b)) in source
                        .shape
                        .iter()
                        .zip(validation_source.shape.iter())
                        .enumerate()
                    {
                        if a == sample_dim && b == validation_dim {
                            candidates.push(ShapeExpr::TensorDim {
                                tensor: source.id,
                                axis: source_axis,
                            });
                        }
                    }
                }
                candidates.dedup();
                if candidates.len() == 1 {
                    dimensions.push(candidates.pop().unwrap());
                } else {
                    unresolved.push((axis, candidates.len()));
                    dimensions.push(ShapeExpr::Infer);
                }
            }
            if unresolved.len() > 1 {
                let (axis, count) = unresolved[0];
                return Err(ExportError::DynamicShapeLost {
                    tensor: output.id,
                    axis,
                    reason: format!(
                        "multiple derived dimensions remain; first axis has {count} candidates"
                    ),
                });
            }
            shapes.push(ResolvedShape {
                operation,
                tensor: output.id,
                dimensions,
            });
        }
        Ok(ResolvedExportGraph {
            graph: self.sample.clone(),
            shapes,
        })
    }
}

fn reshapes(
    graph: &GraphIr,
) -> impl Iterator<Item = (usize, &burn_ir::TensorIr, &burn_ir::TensorIr)> {
    graph
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| match operation {
            OperationIr::BaseFloat(BaseOperationIr::Reshape(op))
            | OperationIr::BaseInt(BaseOperationIr::Reshape(op))
            | OperationIr::BaseBool(BaseOperationIr::Reshape(op)) => {
                Some((index, &op.input, &op.out))
            }
            _ => None,
        })
}

fn tensor(graph: &GraphIr, id: TensorId) -> Option<&burn_ir::TensorIr> {
    graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .find(|tensor| tensor.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape};
    use burn_ir::{ShapeOpIr, TensorIr};

    fn tensor(id: u64, shape: &[usize]) -> TensorIr {
        TensorIr::uninit(
            TensorId::new(id),
            shape.iter().copied().collect::<Shape>(),
            DType::F32,
        )
    }

    fn reshape(input_id: u64, output_id: u64, input: &[usize], output: &[usize]) -> GraphIr {
        GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::Reshape(
            ShapeOpIr {
                input: tensor(input_id, input),
                out: tensor(output_id, output),
            },
        ))])
    }

    #[test]
    fn static_resolver_emits_constants() {
        let graph = reshape(1, 2, &[2, 3, 4], &[2, 12]);
        let resolved = StaticShapeResolver { graph: &graph }.resolve().unwrap();
        assert_eq!(
            resolved.shapes[0].dimensions,
            vec![ShapeExpr::Static(2), ShapeExpr::Static(12)]
        );
    }

    #[test]
    fn paired_resolver_uses_input_dim_and_infer() {
        let sample = reshape(1, 2, &[2, 3, 4, 5], &[2, 3, 20]);
        // Deliberately use different tensor IDs to exercise structural normalization.
        let validation = reshape(11, 12, &[7, 3, 6, 7], &[7, 3, 42]);
        let specs = [InputSpec {
            tensor: TensorId::new(1),
            axes: vec![
                AxisSpec::Dynamic {
                    symbol: "N".into(),
                    validation_value: 7,
                },
                AxisSpec::Static,
                AxisSpec::Dynamic {
                    symbol: "H".into(),
                    validation_value: 6,
                },
                AxisSpec::Dynamic {
                    symbol: "W".into(),
                    validation_value: 7,
                },
            ],
        }];
        let resolved = PairedTraceShapeResolver {
            sample: &sample,
            validation: &validation,
            inputs: &specs,
        }
        .resolve()
        .unwrap();
        assert_eq!(
            resolved.shapes[0].dimensions,
            vec![
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 0
                },
                ShapeExpr::Static(3),
                ShapeExpr::Infer,
            ]
        );
    }

    #[test]
    fn validator_rejects_static_attribute_changes() {
        use burn_ir::SwapDimsOpIr;
        let sample = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: tensor(1, &[2, 3]),
                out: tensor(2, &[3, 2]),
                dim1: 0,
                dim2: 1,
            },
        ))]);
        let validation = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: tensor(4, &[7, 3]),
                out: tensor(5, &[3, 7]),
                dim1: 1,
                dim2: 0,
            },
        ))]);
        assert!(matches!(
            GraphStructureValidator::validate(&sample, &validation),
            Err(ExportError::DynamicGraphMismatch { operation: 0, .. })
        ));
    }
}
