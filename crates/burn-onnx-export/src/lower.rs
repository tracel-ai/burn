use burn_backend::{DType, TensorData};
use burn_ir::{
    ActivationOperationIr, BaseOperationIr, FloatOperationIr, IntOperationIr, InterpolateModeIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, ScalarIr, ScalarOpIr, TensorId, TensorIr,
};
use hashbrown::{HashMap, HashSet};
use onnx_ir::{GraphProto, ModelProto, TensorProto, TypeProto, ValueInfoProto};
use protobuf::{EnumOrUnknown, Message, MessageField};

use crate::{ExportError, ResolvedExportGraph, ShapeExpr};

macro_rules! push_int_attribute {
    ($node:expr, $name:expr, $value:expr) => {{
        $node.attribute.push(Default::default());
        let attribute = $node.attribute.last_mut().unwrap();
        attribute.name = $name.into();
        attribute.type_ = EnumOrUnknown::from_i32(2);
        attribute.i = $value as i64;
    }};
}

macro_rules! push_ints_attribute {
    ($node:expr, $name:expr, $values:expr) => {{
        $node.attribute.push(Default::default());
        let attribute = $node.attribute.last_mut().unwrap();
        attribute.name = $name.into();
        attribute.type_ = EnumOrUnknown::from_i32(7);
        attribute.ints = $values.into_iter().map(|value| value as i64).collect();
    }};
}

macro_rules! push_float_attribute {
    ($node:expr, $name:expr, $value:expr) => {{
        $node.attribute.push(Default::default());
        let attribute = $node.attribute.last_mut().unwrap();
        attribute.name = $name.into();
        attribute.type_ = EnumOrUnknown::from_i32(1);
        attribute.f = $value as f32;
    }};
}

macro_rules! push_string_attribute {
    ($node:expr, $name:expr, $value:expr) => {{
        $node.attribute.push(Default::default());
        let attribute = $node.attribute.last_mut().unwrap();
        attribute.name = $name.into();
        attribute.type_ = EnumOrUnknown::from_i32(3);
        attribute.s = bytes::Bytes::from_static($value.as_bytes());
    }};
}

/// ONNX IR version emitted by this exporter.
pub const ONNX_IR_VERSION: i64 = 8;
/// Default ONNX operator set emitted by this exporter.
pub const ONNX_OPSET_VERSION: i64 = 18;
/// Maximum protobuf payload supported by embedded ONNX tensor data.
pub const MAX_EMBEDDED_PROTOBUF_BYTES: u64 = i32::MAX as u64;

/// Lower an already captured and shape-resolved graph to an embedded ONNX model.
///
/// This low-level API intentionally accepts no Burn module. Parameters can be
/// added as initializers once capture supplies their tensor data and bindings.
pub fn export_graph(graph: &ResolvedExportGraph) -> Result<Vec<u8>, ExportError> {
    export_graph_with_bindings(graph, &HashMap::new(), &graph.graph.inputs, &HashMap::new())
}

/// Lower a resolved graph with concrete initialized values.
///
/// Values belonging to `runtime_inputs` describe sample inputs and are not
/// embedded. Every other value is emitted as an ONNX initializer.
pub fn export_graph_with_values(
    graph: &ResolvedExportGraph,
    values: &HashMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
) -> Result<Vec<u8>, ExportError> {
    export_graph_with_bindings(graph, values, runtime_inputs, &HashMap::new())
}

/// Lower a resolved graph with concrete values and stable initializer names.
///
/// `initializer_names` typically maps captured module parameter tensor IDs to
/// their module paths. Unnamed initialized values retain deterministic tensor-ID names.
pub fn export_graph_with_bindings(
    graph: &ResolvedExportGraph,
    values: &HashMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<Vec<u8>, ExportError> {
    validate_bindings(graph, values, runtime_inputs, initializer_names)?;
    let runtime_set: HashSet<_> = runtime_inputs.iter().copied().collect();
    let embedded_bytes = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .map(|(_, value)| value.bytes.len() as u64)
        .sum::<u64>();
    if embedded_bytes > MAX_EMBEDDED_PROTOBUF_BYTES {
        return Err(ExportError::Serialization(format!(
            "embedded tensor data is {embedded_bytes} bytes, exceeding the protobuf limit of {MAX_EMBEDDED_PROTOBUF_BYTES} bytes"
        )));
    }
    let mut proto = GraphProto::new();
    proto.name = "burn_graph".into();

    for &id in &graph.graph.inputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto.input.push(value_info(tensor, &graph.dynamic_axes)?);
    }
    for &id in &graph.graph.outputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto.output.push(value_info(tensor, &graph.dynamic_axes)?);
    }
    let mut initializers: Vec<_> = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .collect();
    initializers.sort_by(|(lhs, _), (rhs, _)| {
        tensor_name(**lhs, initializer_names).cmp(&tensor_name(**rhs, initializer_names))
    });
    for (&id, data) in initializers {
        let mut initializer = TensorProto::new();
        initializer.name = tensor_name(id, initializer_names);
        initializer.data_type = onnx_data_dtype(id, data.dtype)?;
        initializer.dims = data.shape.iter().map(|dim| *dim as i64).collect();
        initializer.raw_data = bytes::Bytes::copy_from_slice(data.bytes.as_ref());
        proto.initializer.push(initializer);
    }

    for (index, operation) in graph.graph.operations.iter().enumerate() {
        let full = match operation {
            OperationIr::NumericFloat(_, NumericOperationIr::Full(full))
            | OperationIr::NumericInt(_, NumericOperationIr::Full(full)) => Some(full),
            _ => None,
        };
        if let Some(full) = full {
            let shape_name = format!("node_{index}_shape");
            push_i64_initializer(
                &mut proto,
                shape_name.clone(),
                &full
                    .out
                    .shape
                    .iter()
                    .map(|dimension| *dimension as i64)
                    .collect::<Vec<_>>(),
            );
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "ConstantOfShape".into();
            node.input = vec![shape_name];
            node.output = vec![tensor_name(full.out.id, initializer_names)];
            node.attribute.push(Default::default());
            let attribute = node.attribute.last_mut().unwrap();
            attribute.name = "value".into();
            attribute.type_ = EnumOrUnknown::from_i32(4);
            attribute.t =
                MessageField::some(scalar_tensor(full.out.dtype, full.value, full.out.id)?);
            continue;
        }
        let reshape = match operation {
            OperationIr::BaseFloat(BaseOperationIr::Reshape(op))
            | OperationIr::BaseInt(BaseOperationIr::Reshape(op))
            | OperationIr::BaseBool(BaseOperationIr::Reshape(op)) => Some(op),
            _ => None,
        };
        if let Some(reshape) = reshape {
            let resolved = graph
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
                push_i64_initializer(&mut proto, shape_name.clone(), &dimensions);
            } else {
                let mut parts = Vec::with_capacity(resolved.dimensions.len());
                for (axis, dimension) in resolved.dimensions.iter().enumerate() {
                    let part = format!("node_{index}_shape_part_{axis}");
                    match dimension {
                        ShapeExpr::Static(value) => {
                            push_i64_initializer(&mut proto, part.clone(), &[*value as i64]);
                        }
                        ShapeExpr::Infer => {
                            push_i64_initializer(&mut proto, part.clone(), &[-1]);
                        }
                        ShapeExpr::InputDim { input, axis }
                        | ShapeExpr::TensorDim {
                            tensor: input,
                            axis,
                        } => {
                            let source_shape = format!("node_{index}_source_shape_{axis}");
                            proto.node.push(Default::default());
                            let shape = proto.node.last_mut().unwrap();
                            shape.name = source_shape.clone();
                            shape.op_type = "Shape".into();
                            shape.input = vec![tensor_name(*input, initializer_names)];
                            shape.output = vec![source_shape.clone()];

                            let indices = format!("node_{index}_shape_index_{axis}");
                            push_i64_initializer(&mut proto, indices.clone(), &[*axis as i64]);
                            proto.node.push(Default::default());
                            let gather = proto.node.last_mut().unwrap();
                            gather.name = part.clone();
                            gather.op_type = "Gather".into();
                            gather.input = vec![source_shape, indices];
                            gather.output = vec![part.clone()];
                            gather.attribute.push(Default::default());
                            let attribute = gather.attribute.last_mut().unwrap();
                            attribute.name = "axis".into();
                            attribute.type_ = EnumOrUnknown::from_i32(2);
                            attribute.i = 0;
                        }
                    }
                    parts.push(part);
                }
                proto.node.push(Default::default());
                let concat = proto.node.last_mut().unwrap();
                concat.name = shape_name.clone();
                concat.op_type = "Concat".into();
                concat.input = parts;
                concat.output = vec![shape_name.clone()];
                concat.attribute.push(Default::default());
                let attribute = concat.attribute.last_mut().unwrap();
                attribute.name = "axis".into();
                attribute.type_ = EnumOrUnknown::from_i32(2);
                attribute.i = 0;
            }
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "Reshape".into();
            node.input = vec![tensor_name(reshape.input.id, initializer_names), shape_name];
            node.output = vec![tensor_name(reshape.out.id, initializer_names)];
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::Conv2d(conv)) = operation {
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "Conv".into();
            node.input = vec![
                tensor_name(conv.x.id, initializer_names),
                tensor_name(conv.weight.id, initializer_names),
            ];
            if let Some(bias) = &conv.bias {
                node.input.push(tensor_name(bias.id, initializer_names));
            }
            node.output = vec![tensor_name(conv.out.id, initializer_names)];
            push_ints_attribute!(node, "strides", conv.options.stride);
            push_ints_attribute!(node, "dilations", conv.options.dilation);
            push_ints_attribute!(
                node,
                "pads",
                [
                    conv.options.padding[0],
                    conv.options.padding[1],
                    conv.options.padding[0],
                    conv.options.padding[1],
                ]
            );
            push_int_attribute!(node, "group", conv.options.groups);
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::BatchNorm(batch_norm)) = operation {
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "BatchNormalization".into();
            node.input = vec![
                tensor_name(batch_norm.x.id, initializer_names),
                tensor_name(batch_norm.gamma.id, initializer_names),
                tensor_name(batch_norm.beta.id, initializer_names),
                tensor_name(batch_norm.mean.id, initializer_names),
                tensor_name(batch_norm.variance.id, initializer_names),
            ];
            node.output = vec![tensor_name(batch_norm.out.id, initializer_names)];
            push_float_attribute!(node, "epsilon", batch_norm.epsilon.elem::<f64>());
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::Interpolate(interpolate)) = operation {
            let (mode, coordinate_mode, nearest_mode) = match interpolate.options.mode {
                InterpolateModeIr::Nearest => ("nearest", "asymmetric", Some("floor")),
                InterpolateModeIr::NearestExact => {
                    ("nearest", "half_pixel", Some("round_prefer_floor"))
                }
                InterpolateModeIr::Bilinear => (
                    "linear",
                    if interpolate.options.align_corners {
                        "align_corners"
                    } else {
                        "half_pixel"
                    },
                    None,
                ),
                InterpolateModeIr::Bicubic => (
                    "cubic",
                    if interpolate.options.align_corners {
                        "align_corners"
                    } else {
                        "half_pixel"
                    },
                    None,
                ),
                InterpolateModeIr::Lanczos3 => {
                    return Err(ExportError::UnsupportedOperation {
                        operation: index,
                        kind: "Lanczos3 interpolation has no ONNX Resize mode".into(),
                    });
                }
            };
            let sizes = format!("node_{index}_sizes");
            push_i64_initializer(
                &mut proto,
                sizes.clone(),
                &interpolate.output_size.map(|dimension| dimension as i64),
            );
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "Resize".into();
            node.input = vec![
                tensor_name(interpolate.x.id, initializer_names),
                String::new(),
                String::new(),
                sizes,
            ];
            node.output = vec![tensor_name(interpolate.out.id, initializer_names)];
            push_ints_attribute!(node, "axes", [2, 3]);
            push_string_attribute!(node, "mode", mode);
            push_string_attribute!(node, "coordinate_transformation_mode", coordinate_mode);
            if let Some(nearest_mode) = nearest_mode {
                push_string_attribute!(node, "nearest_mode", nearest_mode);
            }
            if matches!(interpolate.options.mode, InterpolateModeIr::Bicubic) {
                push_float_attribute!(node, "cubic_coeff_a", -0.75);
            }
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::MaxPool2d(pool)) = operation {
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "MaxPool".into();
            node.input = vec![tensor_name(pool.x.id, initializer_names)];
            node.output = vec![tensor_name(pool.out.id, initializer_names)];
            push_ints_attribute!(node, "kernel_shape", pool.kernel_size);
            push_ints_attribute!(node, "strides", pool.stride);
            push_ints_attribute!(node, "dilations", pool.dilation);
            push_ints_attribute!(
                node,
                "pads",
                [
                    pool.padding[0],
                    pool.padding[1],
                    pool.padding[0],
                    pool.padding[1],
                ]
            );
            push_int_attribute!(node, "ceil_mode", pool.ceil_mode);
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::AvgPool2d(pool)) = operation {
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "AveragePool".into();
            node.input = vec![tensor_name(pool.x.id, initializer_names)];
            node.output = vec![tensor_name(pool.out.id, initializer_names)];
            push_ints_attribute!(node, "kernel_shape", pool.kernel_size);
            push_ints_attribute!(node, "strides", pool.stride);
            push_ints_attribute!(
                node,
                "pads",
                [
                    pool.padding[0],
                    pool.padding[1],
                    pool.padding[0],
                    pool.padding[1],
                ]
            );
            push_int_attribute!(node, "ceil_mode", pool.ceil_mode);
            push_int_attribute!(node, "count_include_pad", pool.count_include_pad);
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(pool)) = operation
            && pool.output_size == [1, 1]
        {
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = "GlobalAveragePool".into();
            node.input = vec![tensor_name(pool.x.id, initializer_names)];
            node.output = vec![tensor_name(pool.out.id, initializer_names)];
            continue;
        }
        if let OperationIr::Module(ModuleOperationIr::Linear(linear)) = operation {
            let matmul_output = if linear.bias.is_some() {
                format!("node_{index}_matmul")
            } else {
                tensor_name(linear.out.id, initializer_names)
            };
            proto.node.push(Default::default());
            let matmul = proto.node.last_mut().unwrap();
            matmul.name = format!("node_{index}_matmul");
            matmul.op_type = "MatMul".into();
            matmul.input = vec![
                tensor_name(linear.x.id, initializer_names),
                tensor_name(linear.weight.id, initializer_names),
            ];
            matmul.output = vec![matmul_output.clone()];
            if let Some(bias) = &linear.bias {
                proto.node.push(Default::default());
                let add = proto.node.last_mut().unwrap();
                add.name = format!("node_{index}_bias");
                add.op_type = "Add".into();
                add.input = vec![matmul_output, tensor_name(bias.id, initializer_names)];
                add.output = vec![tensor_name(linear.out.id, initializer_names)];
            }
            continue;
        }
        if let Some((op_type, scalar)) = scalar_numeric_op(operation) {
            let scalar_name = format!("node_{index}_scalar");
            push_scalar_initializer(
                &mut proto,
                scalar_name.clone(),
                scalar.lhs.dtype,
                scalar.rhs,
                scalar.lhs.id,
            )?;
            proto.node.push(Default::default());
            let node = proto.node.last_mut().unwrap();
            node.name = format!("node_{index}");
            node.op_type = op_type.into();
            node.input = vec![tensor_name(scalar.lhs.id, initializer_names), scalar_name];
            node.output = vec![tensor_name(scalar.out.id, initializer_names)];
            continue;
        }
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
        node.input = operation
            .inputs()
            .map(|tensor| tensor_name(tensor.id, initializer_names))
            .collect();
        node.output = operation
            .outputs()
            .map(|tensor| tensor_name(tensor.id, initializer_names))
            .collect();
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

fn validate_bindings(
    graph: &ResolvedExportGraph,
    values: &HashMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<(), ExportError> {
    let mut runtime = HashSet::new();
    for &id in runtime_inputs {
        if !runtime.insert(id) {
            return Err(ExportError::InvalidBoundary(format!(
                "duplicate runtime input tensor {id}"
            )));
        }
        if !graph.graph.inputs.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} is not a declared graph input"
            )));
        }
    }
    if runtime.len() != graph.graph.inputs.len() {
        return Err(ExportError::InvalidBoundary(
            "every declared graph input must have one runtime input binding".into(),
        ));
    }
    for (&id, name) in initializer_names {
        if runtime.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} cannot also have an initializer name"
            )));
        }
        if !values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
        if name.is_empty() {
            return Err(ExportError::InvalidBoundary(format!(
                "initializer tensor {id} has an empty name"
            )));
        }
    }

    let mut names = HashMap::<String, TensorId>::new();
    let ids = graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .map(|tensor| tensor.id)
        .chain(values.keys().copied());
    for id in ids {
        let name = tensor_name(id, initializer_names);
        if let Some(previous) = names.insert(name.clone(), id)
            && previous != id
        {
            return Err(ExportError::InvalidBoundary(format!(
                "ONNX value name `{name}` is shared by tensors {previous} and {id}"
            )));
        }
    }

    for (&id, data) in values {
        let Some(tensor) = find_tensor(graph, id) else {
            continue;
        };
        if tensor.dtype != data.dtype || tensor.shape != data.shape {
            return Err(ExportError::InvalidValue {
                tensor: id,
                reason: format!(
                    "graph metadata is {:?} {:?}, initialized value is {:?} {:?}",
                    tensor.dtype, tensor.shape, data.dtype, data.shape
                ),
            });
        }
    }
    Ok(())
}

fn onnx_data_dtype(tensor: TensorId, dtype: DType) -> Result<i32, ExportError> {
    onnx_dtype_parts(tensor, dtype)
}

fn value_info(
    tensor: &TensorIr,
    dynamic_axes: &[crate::DynamicAxis],
) -> Result<ValueInfoProto, ExportError> {
    let mut info = ValueInfoProto::new();
    info.name = name(tensor.id);
    let mut ty = TypeProto::new();
    let tensor_type = ty.mut_tensor_type();
    tensor_type.elem_type = onnx_dtype(tensor)?;
    let shape = tensor_type.shape.mut_or_insert_default();
    for (axis, &dim) in tensor.shape.iter().enumerate() {
        shape.dim.push(Default::default());
        let dimension = shape.dim.last_mut().unwrap();
        if let Some(dynamic) = dynamic_axes
            .iter()
            .find(|dynamic| dynamic.tensor == tensor.id && dynamic.axis == axis)
        {
            dimension.set_dim_param(dynamic.symbol.clone());
        } else {
            dimension.set_dim_value(dim as i64);
        }
    }
    info.type_ = MessageField::some(ty);
    Ok(info)
}

fn onnx_dtype(tensor: &TensorIr) -> Result<i32, ExportError> {
    onnx_dtype_parts(tensor.id, tensor.dtype)
}

fn onnx_dtype_parts(tensor: TensorId, dtype: DType) -> Result<i32, ExportError> {
    // TensorProto.DataType numeric values from the ONNX specification.
    match dtype {
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
            tensor,
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

fn scalar_numeric_op(operation: &OperationIr) -> Option<(&'static str, &ScalarOpIr)> {
    let (OperationIr::NumericFloat(_, operation) | OperationIr::NumericInt(_, operation)) =
        operation
    else {
        return None;
    };
    match operation {
        NumericOperationIr::AddScalar(operation) => Some(("Add", operation)),
        NumericOperationIr::SubScalar(operation) => Some(("Sub", operation)),
        NumericOperationIr::MulScalar(operation) => Some(("Mul", operation)),
        NumericOperationIr::DivScalar(operation) => Some(("Div", operation)),
        _ => None,
    }
}

fn operation_kind(operation: &OperationIr) -> String {
    format!("{operation:?}")
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

fn tensor_name(id: TensorId, initializer_names: &HashMap<TensorId, String>) -> String {
    initializer_names
        .get(&id)
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| name(id))
}

fn push_i64_initializer(proto: &mut GraphProto, name: String, values: &[i64]) {
    let mut tensor = TensorProto::new();
    tensor.name = name;
    tensor.data_type = 7;
    tensor.dims = vec![values.len() as i64];
    let mut raw = Vec::with_capacity(size_of_val(values));
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    tensor.raw_data = bytes::Bytes::from(raw);
    proto.initializer.push(tensor);
}

fn push_scalar_initializer(
    proto: &mut GraphProto,
    name: String,
    dtype: DType,
    value: ScalarIr,
    tensor: TensorId,
) -> Result<(), ExportError> {
    let mut initializer = scalar_tensor(dtype, value, tensor)?;
    initializer.name = name;
    proto.initializer.push(initializer);
    Ok(())
}

fn scalar_tensor(
    dtype: DType,
    value: ScalarIr,
    tensor: TensorId,
) -> Result<TensorProto, ExportError> {
    let mut initializer = TensorProto::new();
    initializer.data_type = onnx_dtype_parts(tensor, dtype)?;
    initializer.dims = vec![1];
    let bytes = match dtype {
        DType::F32 => value.elem::<f32>().to_le_bytes().to_vec(),
        DType::F64 => value.elem::<f64>().to_le_bytes().to_vec(),
        DType::I32 => value.elem::<i32>().to_le_bytes().to_vec(),
        DType::I64 => value.elem::<i64>().to_le_bytes().to_vec(),
        dtype => {
            return Err(ExportError::UnsupportedDType {
                tensor,
                dtype: format!("{dtype:?} scalar initializer"),
            });
        }
    };
    initializer.raw_data = bytes::Bytes::from(bytes);
    Ok(initializer)
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
            dynamic_axes: vec![],
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
