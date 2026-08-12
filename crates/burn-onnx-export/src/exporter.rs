use alloc::{string::String, vec::Vec};

use burn_capture::{CaptureClient, CaptureDevice, CapturedGraph, GraphCapture};
use burn_core::module::{Module, ModuleVisitor, Param};
use burn_ir::{OperationIr, TensorId};
use burn_router::RouterTensor;
use burn_tensor::{Bool, Device, Float, Int, Tensor};
use hashbrown::{HashMap, HashSet};

use crate::{
    ExportError, InputSpec, PairedTraceShapeResolver, ShapeResolver, StaticShapeResolver,
    export_graph_with_bindings,
};

/// Collects tensor IDs from values accepted or returned by a forward function.
///
/// Implementations are provided for capture tensors, vectors, and tuples.
pub trait ExportValues {
    /// Append contained tensor IDs in declaration order.
    fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>);

    /// Return all contained tensor IDs in declaration order.
    fn tensor_ids(&self) -> Vec<TensorId> {
        let mut ids = Vec::new();
        self.collect_tensor_ids(&mut ids);
        ids
    }
}

/// Runtime input values which can be moved to the private capture device.
pub trait ExportInput: ExportValues + Sized {
    /// Move all contained tensors to the capture device.
    fn to_capture_device(self, device: &Device) -> Self;
}

impl ExportValues for RouterTensor<CaptureClient> {
    fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) {
        ids.push(self.id());
    }
}

macro_rules! impl_tensor_value {
    ($kind:ty) => {
        impl<const D: usize> ExportValues for Tensor<D, $kind> {
            fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) {
                let primitive = self
                    .clone()
                    .try_into_primitive::<burn_capture::CaptureBackend>()
                    .expect("export tensor must be on the capture device");
                ids.push(primitive.id());
            }
        }

        impl<const D: usize> ExportInput for Tensor<D, $kind> {
            fn to_capture_device(self, device: &Device) -> Self {
                self.to_device(device)
            }
        }
    };
}

impl_tensor_value!(Float);
impl_tensor_value!(Int);
impl_tensor_value!(Bool);

impl<T: ExportValues> ExportValues for Vec<T> {
    fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) {
        for value in self {
            value.collect_tensor_ids(ids);
        }
    }
}

impl<T: ExportInput> ExportInput for Vec<T> {
    fn to_capture_device(self, device: &Device) -> Self {
        self.into_iter()
            .map(|value| value.to_capture_device(device))
            .collect()
    }
}

macro_rules! impl_export_tuple {
    ($($name:ident),+) => {
        impl<$($name: ExportValues),+> ExportValues for ($($name,)+) {
            #[allow(non_snake_case)]
            fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) {
                let ($($name,)+) = self;
                $($name.collect_tensor_ids(ids);)+
            }
        }
        impl<$($name: ExportInput),+> ExportInput for ($($name,)+) {
            #[allow(non_snake_case)]
            fn to_capture_device(self, device: &Device) -> Self {
                let ($($name,)+) = self;
                ($($name.to_capture_device(device),)+)
            }
        }
    };
}

impl_export_tuple!(A, B);
impl_export_tuple!(A, B, C);
impl_export_tuple!(A, B, C, D);

struct CapturedForward {
    captured: CapturedGraph,
    input_ids: Vec<TensorId>,
    parameter_names: HashMap<TensorId, String>,
}

#[derive(Default)]
struct ParameterNameVisitor {
    path: Vec<String>,
    names: HashMap<TensorId, String>,
}

impl ParameterNameVisitor {
    fn record(&mut self, id: TensorId) {
        self.names.entry(id).or_insert_with(|| self.path.join("."));
    }
}

impl ModuleVisitor for ParameterNameVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.into());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param
            .val()
            .try_into_primitive::<burn_capture::CaptureBackend>()
            .expect("module parameter must be on the capture device");
        self.record(tensor.id());
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        let tensor = param
            .val()
            .try_into_primitive::<burn_capture::CaptureBackend>()
            .expect("module parameter must be on the capture device");
        self.record(tensor.id());
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        let tensor = param
            .val()
            .try_into_primitive::<burn_capture::CaptureBackend>()
            .expect("module parameter must be on the capture device");
        self.record(tensor.id());
    }
}

/// High-level forward-capture ONNX exporter.
///
/// The exporter clones an ordinary Burn module and moves both it and the sample
/// inputs onto a private capture device. It then invokes the supplied forward
/// closure, identifies runtime boundaries, classifies other initialized values
/// as parameters, and emits an embedded-weight ONNX protobuf.
#[derive(Default)]
pub struct OnnxExporter;

impl OnnxExporter {
    /// Create an ONNX exporter.
    pub const fn new() -> Self {
        Self
    }

    /// Create an exporter and its isolated capture device.
    pub fn capture() -> (Device, GraphCapture) {
        let (device, capture) = CaptureDevice::capture();
        (Device::new(device), capture)
    }

    /// Capture one module forward pass and return a serialized static-shape ONNX model.
    pub fn export<M, I, O, F>(
        &self,
        module: &M,
        inputs: I,
        forward: F,
    ) -> Result<Vec<u8>, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportValues,
        F: FnOnce(&M, I) -> O,
    {
        let captured = self.capture_forward(module, inputs, forward)?;
        let resolved = StaticShapeResolver {
            graph: &captured.captured.graph,
        }
        .resolve()?;
        export_graph_with_bindings(
            &resolved,
            &captured.captured.values,
            &captured.input_ids,
            &captured.parameter_names,
        )
    }

    /// Capture two shapes, validate their structure, and export symbolic input axes.
    ///
    /// `input_specs` are positional and must contain one entry per tensor in
    /// `sample_inputs`. The corresponding axes of `validation_inputs` must match
    /// every declared dynamic axis's `validation_value`.
    pub fn export_dynamic<M, I, O, F>(
        &self,
        module: &M,
        sample_inputs: I,
        validation_inputs: I,
        input_specs: &[InputSpec],
        forward: F,
    ) -> Result<Vec<u8>, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportValues,
        F: Fn(&M, I) -> O,
    {
        let sample = self.capture_forward(module, sample_inputs, &forward)?;
        let validation = self.capture_forward(module, validation_inputs, &forward)?;
        let resolved = PairedTraceShapeResolver {
            sample: &sample.captured.graph,
            validation: &validation.captured.graph,
            inputs: input_specs,
        }
        .resolve()?;
        export_graph_with_bindings(
            &resolved,
            &sample.captured.values,
            &sample.input_ids,
            &sample.parameter_names,
        )
    }

    fn capture_forward<M, I, O, F>(
        &self,
        module: &M,
        inputs: I,
        forward: F,
    ) -> Result<CapturedForward, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportValues,
        F: FnOnce(&M, I) -> O,
    {
        let (device, capture) = Self::capture();
        let module = module.clone().to_device(&device);
        let mut visitor = ParameterNameVisitor::default();
        module.visit(&mut visitor);
        let inputs = inputs.to_capture_device(&device);
        let input_ids = inputs.tensor_ids();
        let output_ids = forward(&module, inputs).tensor_ids();
        let mut captured = capture
            .finish(input_ids.iter().copied(), output_ids)
            .map_err(|error| ExportError::InvalidBoundary(error.to_string()))?;
        captured
            .graph
            .operations
            .retain(|operation| !matches!(operation, OperationIr::Init(_) | OperationIr::Drop(_)));
        validate_capture(&captured, &input_ids, &visitor.names)?;
        Ok(CapturedForward {
            captured,
            input_ids,
            parameter_names: visitor.names,
        })
    }
}

fn validate_capture(
    captured: &CapturedGraph,
    runtime_inputs: &[TensorId],
    parameter_names: &HashMap<TensorId, String>,
) -> Result<(), ExportError> {
    for (kind, boundaries) in [
        ("input", captured.graph.inputs.as_slice()),
        ("output", captured.graph.outputs.as_slice()),
    ] {
        let mut unique = HashSet::new();
        if let Some(id) = boundaries.iter().find(|id| !unique.insert(**id)) {
            return Err(ExportError::InvalidBoundary(format!(
                "duplicate graph {kind} tensor {id}"
            )));
        }
    }
    if captured.graph.inputs != runtime_inputs {
        return Err(ExportError::InvalidBoundary(
            "captured graph inputs do not match runtime input declaration order".into(),
        ));
    }
    for &id in runtime_inputs {
        if !captured.values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
    }
    for &id in parameter_names.keys() {
        if !captured.values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
    }

    let mut known: HashSet<_> = captured.values.keys().copied().collect();
    let mut metadata = HashMap::new();
    for (index, operation) in captured.graph.operations.iter().enumerate() {
        for tensor in operation.inputs() {
            metadata.entry(tensor.id).or_insert(tensor);
            if !known.contains(&tensor.id) {
                return Err(ExportError::InvalidBoundary(format!(
                    "operation {index} reads tensor {} before it is initialized or produced",
                    tensor.id
                )));
            }
        }
        for tensor in operation.outputs() {
            metadata.entry(tensor.id).or_insert(tensor);
            known.insert(tensor.id);
        }
    }
    for &id in &captured.graph.outputs {
        if !known.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "graph output tensor {id} is not initialized or produced"
            )));
        }
    }
    for (&id, data) in &captured.values {
        let Some(tensor) = metadata.get(&id) else {
            continue;
        };
        if tensor.dtype != data.dtype || tensor.shape != data.shape {
            return Err(ExportError::InvalidValue {
                tensor: id,
                reason: format!(
                    "captured metadata is {:?} {:?}, initialized value is {:?} {:?}",
                    tensor.dtype, tensor.shape, data.dtype, data.shape
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core as burn;
    use burn_core::module::Param;
    use burn_nn::{Linear, LinearConfig, Relu};
    use onnx_ir::ModelProto;
    use protobuf::Message;

    #[derive(Module, Debug)]
    struct AddModule {
        weight: Param<Tensor<1>>,
    }

    #[derive(Module, Debug)]
    struct Mlp {
        first: Linear,
        activation: Relu,
        second: Linear,
    }

    #[test]
    fn captures_forward_and_embeds_module_value() {
        let device = Device::default();
        let exporter = OnnxExporter::new();
        let module = AddModule {
            weight: Param::from_data([2.0f32, 3.0], &device),
        };
        let input = Tensor::<1>::from_floats([5.0f32, 7.0], &device);

        let bytes = exporter
            .export(&module, input, |module, input| input + module.weight.val())
            .unwrap();
        let model = ModelProto::parse_from_bytes(&bytes).unwrap();
        assert_eq!(model.graph.node[0].op_type, "Add");
        assert_eq!(model.graph.input.len(), 1);
        assert_eq!(model.graph.output.len(), 1);
        assert_eq!(model.graph.initializer.len(), 1);
        assert_eq!(model.graph.initializer[0].name, "weight");
        assert_eq!(
            model.graph.initializer[0].raw_data.len(),
            2 * size_of::<f32>()
        );
    }

    #[test]
    fn exports_two_layer_mlp_forward() {
        let device = Device::default();
        let module = Mlp {
            first: LinearConfig::new(4, 3).init(&device),
            activation: Relu::new(),
            second: LinearConfig::new(3, 2).init(&device),
        };
        let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

        let bytes = OnnxExporter::new()
            .export(&module, input, |module, input| {
                let hidden = module.first.forward(input);
                module.second.forward(module.activation.forward(hidden))
            })
            .unwrap();
        let model = ModelProto::parse_from_bytes(&bytes).unwrap();
        let operations: Vec<_> = model
            .graph
            .node
            .iter()
            .map(|node| node.op_type.as_str())
            .collect();
        assert_eq!(
            operations,
            [
                "MatMul", "Reshape", "Add", "Relu", "MatMul", "Reshape", "Add"
            ]
        );
        // Four module parameters plus two constant bias-reshape operands.
        assert_eq!(model.graph.initializer.len(), 6);
        let initializer_names = model
            .graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert!(initializer_names.contains(&"first.weight"));
        assert!(initializer_names.contains(&"first.bias"));
        assert!(initializer_names.contains(&"second.weight"));
        assert!(initializer_names.contains(&"second.bias"));
        assert_eq!(model.graph.input.len(), 1);
        assert_eq!(model.graph.output.len(), 1);
    }
}
