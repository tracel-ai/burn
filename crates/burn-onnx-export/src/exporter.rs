use alloc::vec::Vec;

use burn_capture::{CaptureClient, CaptureDevice, GraphCapture};
use burn_core::module::Module;
use burn_ir::OperationIr;
use burn_ir::TensorId;
use burn_router::RouterTensor;
use burn_tensor::{Bool, Device, Float, Int, Tensor};

use crate::{ExportError, ShapeResolver, StaticShapeResolver, export_graph_with_values};

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
        let (device, capture) = Self::capture();
        let module = module.clone().to_device(&device);
        let inputs = inputs.to_capture_device(&device);
        let input_ids = inputs.tensor_ids();
        let outputs = forward(&module, inputs);
        let output_ids = outputs.tensor_ids();
        let mut captured = capture
            .finish(input_ids.iter().copied(), output_ids)
            .map_err(|error| ExportError::InvalidBoundary(error.to_string()))?;
        // Initial values are represented by graph inputs or ONNX initializers;
        // lifetime-only drops have no ONNX computation semantics.
        captured
            .graph
            .operations
            .retain(|operation| !matches!(operation, OperationIr::Init(_) | OperationIr::Drop(_)));
        let resolved = StaticShapeResolver {
            graph: &captured.graph,
        }
        .resolve()?;
        export_graph_with_values(&resolved, &captured.values, &input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core as burn;
    use burn_core::module::Param;
    use onnx_ir::ModelProto;
    use protobuf::Message;

    #[derive(Module, Debug)]
    struct AddModule {
        weight: Param<Tensor<1>>,
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
        assert_eq!(
            model.graph.initializer[0].raw_data.len(),
            2 * size_of::<f32>()
        );
    }
}
