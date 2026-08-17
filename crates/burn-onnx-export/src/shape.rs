use burn_ir::{
    BaseOperationIr, GraphIr, ModuleOperationIr, NumericOperationIr, OperationIr, TensorId,
};

use crate::{
    DynamicAxis, ExportError, GraphStructureValidator, ResolvedExportGraph, ResolvedShape,
    ShapeExpr,
};

/// Annotation for one runtime input axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSpec {
    /// The dimension is fixed to the captured value.
    Static,
    /// The dimension is runtime-variable and identified by an ONNX symbol.
    Dynamic {
        /// ONNX symbolic dimension name.
        symbol: String,
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
    pub fn dynamic(symbol: impl Into<String>) -> Self {
        Self::Dynamic {
            symbol: symbol.into(),
        }
    }
}

/// Replaceable shape-resolution stage used before ONNX lowering.
pub(crate) trait ShapeResolver {
    /// Resolve shape-sensitive operands in a captured graph.
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError>;
}

/// Resolves every captured reshape dimension as a constant.
pub(crate) struct StaticShapeResolver<'a> {
    /// Captured graph.
    pub(crate) graph: &'a GraphIr,
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
pub(crate) struct PairedTraceShapeResolver<'a> {
    /// Primary capture.
    pub(crate) sample: &'a GraphIr,
    /// Capture produced with validation input dimensions.
    pub(crate) validation: &'a GraphIr,
    /// Explicit dynamic input annotations.
    pub(crate) inputs: &'a [InputSpec],
}

impl ShapeResolver for PairedTraceShapeResolver<'_> {
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError> {
        GraphStructureValidator::validate(self.sample, self.validation)?;
        validate_shape_sensitive_operations(self.sample, self.validation)?;
        let sample_shapes = boundary_shapes(self.sample, &self.sample.inputs)?;
        let validation_shapes = boundary_shapes(self.validation, &self.validation.inputs)?;
        validate_input_specs(self.inputs, &sample_shapes, &validation_shapes)?;
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
                    AxisSpec::Dynamic { symbol } => {
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

/// Reject varying shape operands that the paired resolver cannot represent.
///
/// Structural validation deliberately ignores these values. Every ignored
/// operand must be handled here so lowering can never silently embed a value
/// taken only from the sample capture.
fn validate_shape_sensitive_operations(
    sample: &GraphIr,
    validation: &GraphIr,
) -> Result<(), ExportError> {
    for (index, (sample, validation)) in sample
        .operations
        .iter()
        .zip(&validation.operations)
        .enumerate()
    {
        match (sample, validation) {
            (
                OperationIr::NumericFloat(_, NumericOperationIr::Full(sample))
                | OperationIr::NumericInt(_, NumericOperationIr::Full(sample)),
                OperationIr::NumericFloat(_, NumericOperationIr::Full(validation))
                | OperationIr::NumericInt(_, NumericOperationIr::Full(validation)),
            ) if sample.out.shape != validation.out.shape => {
                let axis = first_different_dimension(&sample.out.shape, &validation.out.shape);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!(
                        "operation {index} creates a tensor with a varying shape, but dynamic Full shapes are not resolved"
                    ),
                });
            }
            (
                OperationIr::Module(ModuleOperationIr::Interpolate(sample)),
                OperationIr::Module(ModuleOperationIr::Interpolate(validation)),
            ) if sample.output_size != validation.output_size => {
                let offset = sample
                    .output_size
                    .iter()
                    .zip(validation.output_size)
                    .position(|(sample, validation)| *sample != validation)
                    .unwrap_or(0);
                let axis = sample.out.shape.num_dims().saturating_sub(2) + offset;
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!("operation {index} has a varying interpolation output size"),
                });
            }
            (
                OperationIr::BaseFloat(BaseOperationIr::Slice(sample))
                | OperationIr::BaseInt(BaseOperationIr::Slice(sample))
                | OperationIr::BaseBool(BaseOperationIr::Slice(sample)),
                OperationIr::BaseFloat(BaseOperationIr::Slice(validation))
                | OperationIr::BaseInt(BaseOperationIr::Slice(validation))
                | OperationIr::BaseBool(BaseOperationIr::Slice(validation)),
            ) if sample.ranges != validation.ranges => {
                let axis = sample
                    .ranges
                    .iter()
                    .zip(&validation.ranges)
                    .position(|(sample, validation)| sample != validation)
                    .unwrap_or(0);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!("operation {index} has varying slice bounds"),
                });
            }
            (
                OperationIr::BaseFloat(BaseOperationIr::SliceAssign(sample))
                | OperationIr::BaseInt(BaseOperationIr::SliceAssign(sample)),
                OperationIr::BaseFloat(BaseOperationIr::SliceAssign(validation))
                | OperationIr::BaseInt(BaseOperationIr::SliceAssign(validation)),
            ) if sample.ranges != validation.ranges => {
                let axis = sample
                    .ranges
                    .iter()
                    .zip(&validation.ranges)
                    .position(|(sample, validation)| sample != validation)
                    .unwrap_or(0);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!("operation {index} has varying slice-assignment bounds"),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn first_different_dimension(
    sample: &burn_backend::Shape,
    validation: &burn_backend::Shape,
) -> usize {
    sample
        .iter()
        .zip(validation.iter())
        .position(|(sample, validation)| sample != validation)
        .unwrap_or_else(|| sample.num_dims().min(validation.num_dims()))
}

pub(crate) fn validate_input_specs(
    specs: &[InputSpec],
    sample_shapes: &[Vec<usize>],
    validation_shapes: &[Vec<usize>],
) -> Result<(), ExportError> {
    if sample_shapes.len() != validation_shapes.len() {
        return Err(ExportError::InvalidBoundary(format!(
            "sample inputs contain {} tensors but validation inputs contain {}",
            sample_shapes.len(),
            validation_shapes.len()
        )));
    }
    if specs.len() != sample_shapes.len() {
        return Err(ExportError::InvalidBoundary(format!(
            "received {} input specs for {} input tensors",
            specs.len(),
            sample_shapes.len()
        )));
    }

    let mut symbols = hashbrown::HashMap::<&str, (usize, usize)>::new();
    for (input, ((spec, sample), validation)) in specs
        .iter()
        .zip(sample_shapes)
        .zip(validation_shapes)
        .enumerate()
    {
        if sample.len() != validation.len() {
            return Err(ExportError::InvalidBoundary(format!(
                "sample input {input} has rank {} but validation input has rank {}",
                sample.len(),
                validation.len()
            )));
        }
        if spec.axes.len() != sample.len() {
            return Err(ExportError::InvalidBoundary(format!(
                "input {input} has rank {} but its spec has {} axes",
                sample.len(),
                spec.axes.len()
            )));
        }
        for (axis, ((axis_spec, &sample_dim), &validation_dim)) in
            spec.axes.iter().zip(sample).zip(validation).enumerate()
        {
            match axis_spec {
                AxisSpec::Static if sample_dim != validation_dim => {
                    return Err(ExportError::InvalidBoundary(format!(
                        "static input {input} axis {axis} differs ({sample_dim} != {validation_dim})"
                    )));
                }
                AxisSpec::Dynamic { symbol } => {
                    if symbol.is_empty() {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic input {input} axis {axis} has an empty symbol"
                        )));
                    }
                    if sample_dim == validation_dim {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic input {input} axis {axis} must differ between sample and validation inputs"
                        )));
                    }
                    if let Some((previous_sample, previous_validation)) =
                        symbols.insert(symbol, (sample_dim, validation_dim))
                        && (previous_sample, previous_validation) != (sample_dim, validation_dim)
                    {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic symbol `{symbol}` refers to inconsistent dimensions"
                        )));
                    }
                }
                AxisSpec::Static => {}
            }
        }
    }
    Ok(())
}

fn boundary_shapes(graph: &GraphIr, ids: &[TensorId]) -> Result<Vec<Vec<usize>>, ExportError> {
    ids.iter()
        .map(|&id| {
            tensor(graph, id)
                .map(|tensor| tensor.shape.to_vec())
                .ok_or(ExportError::MissingValue(id))
        })
        .collect()
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
    use burn_ir::{
        FullOpIr, InterpolateModeIr, InterpolateOpIr, InterpolateOptionsIr, ScalarIr, ShapeOpIr,
        TensorIr,
    };

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
            AxisSpec::Dynamic { symbol: "N".into() },
            AxisSpec::Static,
            AxisSpec::Dynamic { symbol: "H".into() },
            AxisSpec::Dynamic { symbol: "W".into() },
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
            AxisSpec::dynamic("N"),
            AxisSpec::dynamic("H"),
            AxisSpec::dynamic("W"),
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
            AxisSpec::dynamic("first"),
            AxisSpec::dynamic("second"),
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
    fn paired_resolver_rejects_unresolved_dynamic_full_shape() {
        let sample = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Full(FullOpIr {
                out: tensor(1, &[2, 3]),
                value: ScalarIr::Float(0.0),
            }),
        )]);
        let validation = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Full(FullOpIr {
                out: tensor(11, &[5, 3]),
                value: ScalarIr::Float(0.0),
            }),
        )]);

        GraphStructureValidator::validate(&sample, &validation).unwrap();
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &[],
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost {
                tensor,
                axis: 0,
                ..
            }) if tensor == TensorId::new(1)
        ));
    }

    #[test]
    fn paired_resolver_rejects_varying_interpolation_size() {
        let sample = GraphIr::new(vec![OperationIr::Module(ModuleOperationIr::Interpolate(
            InterpolateOpIr {
                x: tensor(1, &[1, 1, 4, 4]),
                output_size: [8, 8],
                options: InterpolateOptionsIr {
                    mode: InterpolateModeIr::Nearest,
                    align_corners: false,
                },
                out: tensor(2, &[1, 1, 8, 8]),
            },
        ))]);
        let validation = GraphIr::new(vec![OperationIr::Module(ModuleOperationIr::Interpolate(
            InterpolateOpIr {
                x: tensor(11, &[1, 1, 5, 5]),
                output_size: [10, 10],
                options: InterpolateOptionsIr {
                    mode: InterpolateModeIr::Nearest,
                    align_corners: false,
                },
                out: tensor(12, &[1, 1, 10, 10]),
            },
        ))]);

        GraphStructureValidator::validate(&sample, &validation).unwrap();
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &[],
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost { tensor, axis: 2, .. })
                if tensor == TensorId::new(2)
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
