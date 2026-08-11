use burn_backend::DType;
use burn_ir::{
    ActivationOperationIr, FloatOperationIr, IntOperationIr, NumericOperationIr, OperationIr,
    TensorId, TensorIr,
};
use onnx_ir::{GraphProto, ModelProto, TypeProto, ValueInfoProto};
use protobuf::{Message, MessageField};

use crate::{ExportError, ResolvedExportGraph};

/// ONNX IR version emitted by this exporter.
pub const ONNX_IR_VERSION: i64 = 8;
/// Default ONNX operator set emitted by this exporter.
pub const ONNX_OPSET_VERSION: i64 = 18;

/// Lower an already captured and shape-resolved graph to an embedded ONNX model.
///
/// This low-level API intentionally accepts no Burn module. Parameters can be
/// added as initializers once capture supplies their tensor data and bindings.
pub fn export_graph(graph: &ResolvedExportGraph) -> Result<Vec<u8>, ExportError> {
    let mut proto = GraphProto::new();
    proto.name = "burn_graph".into();

    for &id in &graph.graph.inputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto.input.push(value_info(tensor)?);
    }
    for &id in &graph.graph.outputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto.output.push(value_info(tensor)?);
    }

    for (index, operation) in graph.graph.operations.iter().enumerate() {
        let Some(op_type) = onnx_op_type(operation) else {
            return Err(ExportError::UnsupportedOperation {
                operation: index,
                kind: operation_kind(operation),
            });
        };
        proto.node.push(Default::default());
        let node = proto.node.last_mut().expect("node was just inserted");
        node.name = format!("node_{index}");
        node.op_type = op_type.into();
        node.input = operation.inputs().map(|tensor| name(tensor.id)).collect();
        node.output = operation.outputs().map(|tensor| name(tensor.id)).collect();
    }

    let mut model = ModelProto::new();
    model.ir_version = ONNX_IR_VERSION;
    model.producer_name = "burn-onnx-export".into();
    model.graph = MessageField::some(proto);
    model.opset_import.push(Default::default());
    model.opset_import[0].version = ONNX_OPSET_VERSION;
    model
        .write_to_bytes()
        .map_err(|error| ExportError::Serialization(error.to_string()))
}

fn value_info(tensor: &TensorIr) -> Result<ValueInfoProto, ExportError> {
    let mut info = ValueInfoProto::new();
    info.name = name(tensor.id);
    let mut ty = TypeProto::new();
    let tensor_type = ty.mut_tensor_type();
    tensor_type.elem_type = onnx_dtype(tensor)?;
    let shape = tensor_type.shape.mut_or_insert_default();
    for &dim in tensor.shape.iter() {
        shape.dim.push(Default::default());
        shape.dim.last_mut().unwrap().set_dim_value(dim as i64);
    }
    info.type_ = MessageField::some(ty);
    Ok(info)
}

fn onnx_dtype(tensor: &TensorIr) -> Result<i32, ExportError> {
    // TensorProto.DataType numeric values from the ONNX specification.
    match tensor.dtype {
        DType::F32 => Ok(1),
        DType::U8 => Ok(2),
        DType::I8 => Ok(3),
        DType::I16 => Ok(5),
        DType::I32 => Ok(6),
        DType::I64 => Ok(7),
        DType::Bool(_) => Ok(9),
        DType::F16 => Ok(10),
        DType::F64 => Ok(11),
        DType::U32 => Ok(12),
        DType::U64 => Ok(13),
        DType::BF16 => Ok(16),
        dtype => Err(ExportError::UnsupportedDType {
            tensor: tensor.id,
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn onnx_op_type(operation: &OperationIr) -> Option<&'static str> {
    match operation {
        OperationIr::NumericFloat(_, op) | OperationIr::NumericInt(_, op) => match op {
            NumericOperationIr::Add(_) => Some("Add"),
            NumericOperationIr::Sub(_) => Some("Sub"),
            NumericOperationIr::Mul(_) => Some("Mul"),
            NumericOperationIr::Div(_) => Some("Div"),
            NumericOperationIr::Abs(_) => Some("Abs"),
            _ => None,
        },
        OperationIr::Float(_, op) => match op {
            FloatOperationIr::Exp(_) => Some("Exp"),
            FloatOperationIr::Log(_) => Some("Log"),
            FloatOperationIr::Sqrt(_) => Some("Sqrt"),
            FloatOperationIr::Tanh(_) => Some("Tanh"),
            FloatOperationIr::Matmul(_) => Some("MatMul"),
            _ => None,
        },
        OperationIr::Int(IntOperationIr::Matmul(_)) => Some("MatMul"),
        OperationIr::Activation(op) => match op {
            ActivationOperationIr::Relu(_) => Some("Relu"),
            ActivationOperationIr::Sigmoid(_) => Some("Sigmoid"),
            _ => None,
        },
        _ => None,
    }
}

fn operation_kind(operation: &OperationIr) -> String {
    format!("{operation:?}")
        .split(['(', '{'])
        .next()
        .unwrap_or("unknown")
        .into()
}

fn find_tensor(graph: &ResolvedExportGraph, id: TensorId) -> Option<&TensorIr> {
    graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .find(|tensor| tensor.id == id)
}

fn name(id: TensorId) -> String {
    format!("tensor_{}", id.value())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::Shape;
    use burn_ir::{BinaryOpIr, GraphIr};

    fn tensor(id: u64) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), Shape::new([2, 3]), DType::F32)
    }

    #[test]
    fn writes_parseable_opset_18_model() {
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor(1),
                rhs: tensor(2),
                out: tensor(3),
            }),
        )]);
        let bytes = export_graph(&ResolvedExportGraph {
            graph,
            shapes: vec![],
        })
        .unwrap();
        let model = ModelProto::parse_from_bytes(&bytes).unwrap();
        assert_eq!(model.ir_version, ONNX_IR_VERSION);
        assert_eq!(model.opset_import[0].version, ONNX_OPSET_VERSION);
        assert_eq!(model.graph.node[0].op_type, "Add");
        assert_eq!(model.graph.input.len(), 2);
        assert_eq!(model.graph.output.len(), 1);
    }
}
