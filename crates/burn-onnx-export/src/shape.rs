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
    /// Axis annotations in tensor order.
    pub axes: Vec<AxisSpec>,
}

impl InputSpec {
    /// Create annotations for one runtime input, in input-axis order.
    pub fn new(axes: impl Into<Vec<AxisSpec>>) -> Self {
        Self { axes: axes.into() }
    }
}

impl AxisSpec {
    /// Create a dynamic axis annotation.
    pub fn dynamic(symbol: impl Into<String>, validation_value: usize) -> Self {
        Self::Dynamic {
            symbol: symbol.into(),
            validation_value,
        }
    }
}

/// Symbolic axis attached to a captured runtime input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicAxis {
    /// Runtime input tensor.
    pub tensor: TensorId,
    /// Axis within the input tensor.
    pub axis: usize,
    /// ONNX symbolic dimension name.
    pub symbol: String,
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
    /// Symbolic dimensions declared on runtime graph inputs.
    pub dynamic_axes: Vec<DynamicAxis>,
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
            dynamic_axes: Vec::new(),
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
        if self.inputs.len() != self.sample.inputs.len() {
            return Err(ExportError::InvalidBoundary(format!(
                "received {} input specs for {} graph inputs",
                self.inputs.len(),
                self.sample.inputs.len()
            )));
        }
        let mut dynamic_axes = Vec::new();
        for (position, spec) in self.inputs.iter().enumerate() {
            let sample_id = self.sample.inputs[position];
            let validation_id = self.validation.inputs[position];
            let sample_input =
                tensor(self.sample, sample_id).ok_or(ExportError::MissingValue(sample_id))?;
            let validation_input = tensor(self.validation, validation_id)
                .ok_or(ExportError::MissingValue(validation_id))?;
            if spec.axes.len() != sample_input.shape.len() {
                return Err(ExportError::InvalidBoundary(format!(
                    "input {position} has rank {} but its spec has {} axes",
                    sample_input.shape.len(),
                    spec.axes.len()
                )));
            }
            for (axis, axis_spec) in spec.axes.iter().enumerate() {
                match axis_spec {
                    AxisSpec::Static => {
                        if sample_input.shape[axis] != validation_input.shape[axis] {
                            return Err(ExportError::DynamicGraphMismatch {
                                operation: 0,
                                reason: format!(
                                    "static input {position} axis {axis} differs ({} != {})",
                                    sample_input.shape[axis], validation_input.shape[axis]
                                ),
                            });
                        }
                    }
                    AxisSpec::Dynamic {
                        symbol,
                        validation_value,
                    } => {
                        if validation_input.shape[axis] != *validation_value {
                            return Err(ExportError::InvalidBoundary(format!(
                                "validation input {position} axis {axis} is {}, expected {validation_value}",
                                validation_input.shape[axis]
                            )));
                        }
                        if sample_input.shape[axis] == *validation_value {
                            return Err(ExportError::InvalidBoundary(format!(
                                "dynamic input {position} axis {axis} must use a validation value different from the sample value"
                            )));
                        }
                        dynamic_axes.push(DynamicAxis {
                            tensor: sample_id,
                            axis,
                            symbol: symbol.clone(),
                        });
                    }
                }
            }
        }
        for (position, (&sample_id, &validation_id)) in self
            .sample
            .outputs
            .iter()
            .zip(&self.validation.outputs)
            .enumerate()
        {
            let sample_output =
                tensor(self.sample, sample_id).ok_or(ExportError::MissingValue(sample_id))?;
            let validation_output = tensor(self.validation, validation_id)
                .ok_or(ExportError::MissingValue(validation_id))?;
            for (axis, (&sample_dim, &validation_dim)) in sample_output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                if sample_dim == validation_dim {
                    continue;
                }
                let mut symbols = Vec::new();
                for (input_position, spec) in self.inputs.iter().enumerate() {
                    let sample_input = tensor(self.sample, self.sample.inputs[input_position])
                        .ok_or(ExportError::MissingValue(
                            self.sample.inputs[input_position],
                        ))?;
                    let validation_input =
                        tensor(self.validation, self.validation.inputs[input_position]).ok_or(
                            ExportError::MissingValue(self.validation.inputs[input_position]),
                        )?;
                    for (input_axis, axis_spec) in spec.axes.iter().enumerate() {
                        if let AxisSpec::Dynamic { symbol, .. } = axis_spec
                            && sample_input.shape[input_axis] == sample_dim
                            && validation_input.shape[input_axis] == validation_dim
                        {
                            symbols.push(symbol.clone());
                        }
                    }
                }
                symbols.sort();
                symbols.dedup();
                dynamic_axes.push(DynamicAxis {
                    tensor: sample_id,
                    axis,
                    symbol: if symbols.len() == 1 {
                        symbols.pop().unwrap()
                    } else {
                        format!("output_{position}_dim_{axis}")
                    },
                });
            }
        }
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
                for (position, spec) in self.inputs.iter().enumerate() {
                    let sample_id = self.sample.inputs[position];
                    if let (Some(sample_input), Some(validation_input)) = (
                        tensor(self.sample, sample_id),
                        tensor(self.validation, self.validation.inputs[position]),
                    ) {
                        for (input_axis, axis_spec) in spec.axes.iter().enumerate() {
                            if matches!(axis_spec, AxisSpec::Dynamic { .. })
                                && sample_input.shape.get(input_axis) == Some(&sample_dim)
                                && validation_input.shape.get(input_axis) == Some(&validation_dim)
                            {
                                candidates.push(ShapeExpr::InputDim {
                                    input: sample_id,
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
                match candidates.len() {
                    1 => dimensions.push(candidates.pop().unwrap()),
                    0 => {
                        unresolved.push(axis);
                        dimensions.push(ShapeExpr::Infer);
                    }
                    count => {
                        return Err(ExportError::DynamicShapeLost {
                            tensor: output.id,
                            axis,
                            reason: format!(
                                "dimension matches {count} dynamic source axes and is ambiguous"
                            ),
                        });
                    }
                }
            }
            if unresolved.len() > 1 {
                let axis = unresolved[0];
                return Err(ExportError::DynamicShapeLost {
                    tensor: output.id,
                    axis,
                    reason: "multiple element-count-derived dimensions remain".into(),
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
            dynamic_axes,
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
        let specs = [InputSpec::new(vec![
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
        ])];
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
    fn paired_resolver_preserves_inserted_static_axis() {
        let sample = reshape(1, 2, &[2, 5, 7], &[2, 1, 5, 7]);
        let validation = reshape(11, 12, &[3, 6, 8], &[3, 1, 6, 8]);
        let specs = [InputSpec::new([
            AxisSpec::dynamic("N", 3),
            AxisSpec::dynamic("H", 6),
            AxisSpec::dynamic("W", 8),
        ])];
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
                    axis: 0,
                },
                ShapeExpr::Static(1),
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 1,
                },
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 2,
                },
            ]
        );
    }

    #[test]
    fn paired_resolver_rejects_coincident_dynamic_axes() {
        let sample = reshape(1, 2, &[2, 2], &[2, 2]);
        let validation = reshape(11, 12, &[3, 3], &[3, 3]);
        let specs = [InputSpec::new([
            AxisSpec::dynamic("first", 3),
            AxisSpec::dynamic("second", 3),
        ])];
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &specs,
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost { .. })
        ));
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
