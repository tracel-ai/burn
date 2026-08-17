//! Lowering for Burn base tensor operations.
//!
//! This family contains operations requiring ONNX-specific operands or
//! attributes, including reshape shape expressions and concatenation axes.

use burn_ir::{BaseOperationIr, OperationIr};

use crate::{ExportError, ShapeExpr};

use super::{context::LoweringContext, patterns};

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    if let Some(pad) = patterns::constant_pad(&context.graph.graph.operations, index) {
        let pads_name = format!("node_{index}_pads");
        context.i64_initializer(pads_name.clone(), &pad.pads);
        let value_name = format!("node_{index}_value");
        context.scalar_initializer(
            value_name.clone(),
            pad.full.out.dtype,
            pad.full.value,
            pad.full.out.id,
        )?;
        let input = context.tensor_name(pad.slice_assign.value.id);
        let output = context.tensor_name(pad.slice_assign.out.id);
        context.node(
            format!("node_{index}"),
            "Pad",
            vec![input, pads_name, value_name],
            vec![output],
        );
        context.string_attribute("mode", "constant");
        return Ok(true);
    }

    let base = match operation {
        OperationIr::BaseFloat(operation)
        | OperationIr::BaseInt(operation)
        | OperationIr::BaseBool(operation) => operation,
        _ => return Ok(false),
    };
    match base {
        BaseOperationIr::Reshape(reshape) => {
            let resolved = context
                .graph
                .shapes
                .iter()
                .find(|shape| shape.operation == index)
                .ok_or_else(|| ExportError::DynamicShapeLost {
                    tensor: reshape.out.id,
                    axis: 0,
                    reason: "reshape has no resolved shape operand".into(),
                })?;
            let shape_name = format!("node_{index}_shape");
            if resolved
                .dimensions
                .iter()
                .all(|dimension| matches!(dimension, ShapeExpr::Static(_) | ShapeExpr::Infer))
            {
                let dimensions = resolved
                    .dimensions
                    .iter()
                    .map(|dimension| match dimension {
                        ShapeExpr::Static(value) => *value as i64,
                        ShapeExpr::Infer => -1,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                context.i64_initializer(shape_name.clone(), &dimensions);
            } else {
                let mut parts = Vec::with_capacity(resolved.dimensions.len());
                for (dimension_index, dimension) in resolved.dimensions.iter().enumerate() {
                    let part = format!("node_{index}_shape_part_{dimension_index}");
                    match dimension {
                        ShapeExpr::Static(value) => {
                            context.i64_initializer(part.clone(), &[*value as i64]);
                        }
                        ShapeExpr::Infer => {
                            context.i64_initializer(part.clone(), &[-1]);
                        }
                        ShapeExpr::InputDim { input, axis }
                        | ShapeExpr::TensorDim {
                            tensor: input,
                            axis,
                        } => {
                            let source_shape =
                                format!("node_{index}_source_shape_{dimension_index}");
                            let input = context.tensor_name(*input);
                            context.node(
                                source_shape.clone(),
                                "Shape",
                                vec![input],
                                vec![source_shape.clone()],
                            );
                            let indices = format!("node_{index}_shape_index_{dimension_index}");
                            context.i64_initializer(indices.clone(), &[*axis as i64]);
                            context.node(
                                part.clone(),
                                "Gather",
                                vec![source_shape, indices],
                                vec![part.clone()],
                            );
                            context.int_attribute("axis", 0);
                        }
                    }
                    parts.push(part);
                }
                context.node(
                    shape_name.clone(),
                    "Concat",
                    parts,
                    vec![shape_name.clone()],
                );
                context.int_attribute("axis", 0);
            }
            let input = context.tensor_name(reshape.input.id);
            let output = context.tensor_name(reshape.out.id);
            context.node(
                format!("node_{index}"),
                "Reshape",
                vec![input, shape_name],
                vec![output],
            );
            Ok(true)
        }
        BaseOperationIr::Cat(cat) => {
            let inputs = cat
                .tensors
                .iter()
                .map(|tensor| context.tensor_name(tensor.id))
                .collect();
            let output = context.tensor_name(cat.out.id);
            context.node(format!("node_{index}"), "Concat", inputs, vec![output]);
            context.int_attribute("axis", cat.dim as i64);
            Ok(true)
        }
        _ => Ok(false),
    }
}
