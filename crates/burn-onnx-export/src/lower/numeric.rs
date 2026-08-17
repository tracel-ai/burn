//! Lowering for numeric tensor and tensor-scalar operations.
//!
//! Scalar operands and creation shapes are materialized as ONNX initializers
//! before their corresponding nodes are emitted.

use burn_ir::{NumericOperationIr, OperationIr, ScalarOpIr};

use crate::ExportError;

use super::{context::LoweringContext, scalar_tensor};

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    let numeric = match operation {
        OperationIr::NumericFloat(_, operation) | OperationIr::NumericInt(_, operation) => {
            operation
        }
        _ => return Ok(false),
    };
    if let NumericOperationIr::Full(full) = numeric {
        let shape_name = format!("node_{index}_shape");
        context.i64_initializer(
            shape_name.clone(),
            &full
                .out
                .shape
                .iter()
                .map(|dimension| *dimension as i64)
                .collect::<Vec<_>>(),
        );
        let output = context.tensor_name(full.out.id);
        context.node(
            format!("node_{index}"),
            "ConstantOfShape",
            vec![shape_name],
            vec![output],
        );
        context.tensor_attribute(
            "value",
            scalar_tensor(full.out.dtype, full.value, full.out.id)?,
        );
        return Ok(true);
    }
    if let Some((op_type, scalar)) = scalar_operation(numeric) {
        lower_scalar(context, index, op_type, scalar)?;
        return Ok(true);
    }
    let op_type = match numeric {
        NumericOperationIr::Add(_) => "Add",
        NumericOperationIr::Sub(_) => "Sub",
        NumericOperationIr::Mul(_) => "Mul",
        NumericOperationIr::Div(_) => "Div",
        NumericOperationIr::Abs(_) => "Abs",
        NumericOperationIr::Neg(_) => "Neg",
        _ => return Ok(false),
    };
    let inputs = operation
        .inputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    let outputs = operation
        .outputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    context.node(format!("node_{index}"), op_type, inputs, outputs);
    Ok(true)
}

fn scalar_operation(operation: &NumericOperationIr) -> Option<(&'static str, &ScalarOpIr)> {
    match operation {
        NumericOperationIr::AddScalar(operation) => Some(("Add", operation)),
        NumericOperationIr::SubScalar(operation) => Some(("Sub", operation)),
        NumericOperationIr::MulScalar(operation) => Some(("Mul", operation)),
        NumericOperationIr::DivScalar(operation) => Some(("Div", operation)),
        _ => None,
    }
}

fn lower_scalar(
    context: &mut LoweringContext<'_>,
    index: usize,
    op_type: &'static str,
    scalar: &ScalarOpIr,
) -> Result<(), ExportError> {
    let scalar_name = format!("node_{index}_scalar");
    context.scalar_initializer(
        scalar_name.clone(),
        scalar.lhs.dtype,
        scalar.rhs,
        scalar.lhs.id,
    )?;
    let input = context.tensor_name(scalar.lhs.id);
    let output = context.tensor_name(scalar.out.id);
    context.node(
        format!("node_{index}"),
        op_type,
        vec![input, scalar_name],
        vec![output],
    );
    Ok(())
}
