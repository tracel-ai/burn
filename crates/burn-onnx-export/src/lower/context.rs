//! Shared state and protobuf-building helpers for operation lowerers.
//!
//! Keeping naming, initializer encoding, attribute encoding, and final model
//! construction here prevents operation-family modules from each implementing
//! subtly different protobuf behavior.

use burn_backend::DType;
use burn_ir::{ScalarIr, TensorId};
use hashbrown::HashMap;
use onnx_ir::{GraphProto, ModelProto, TensorProto};
use protobuf::{EnumOrUnknown, Message, MessageField};

use crate::{ExportError, OnnxModel, Opset, ResolvedExportGraph};

use super::{ONNX_IR_VERSION, scalar_tensor};

/// Mutable protobuf construction state shared by operation lowerers.
///
/// The context is created after graph boundaries, runtime bindings, and
/// captured values have been validated. It owns the partially built ONNX graph,
/// borrows the resolved exporter graph and initializer-name bindings, and
/// carries the selected opset through final model construction.
///
/// Operation lowerers must append a node with [`Self::node`] before calling an
/// attribute helper. Attribute helpers always modify the most recently appended
/// node; keeping those calls adjacent makes that invariant visible at each
/// lowering site. Tensor names must be obtained through [`Self::tensor_name`]
/// so module parameter paths and generated tensor names remain consistent.
pub(super) struct LoweringContext<'a> {
    /// Shape-resolved graph being lowered.
    pub(super) graph: &'a ResolvedExportGraph,
    /// Partially constructed ONNX graph.
    proto: GraphProto,
    /// Stable names assigned to captured module parameters.
    initializer_names: &'a HashMap<TensorId, String>,
    /// Operator-set semantics selected by the caller.
    opset: Opset,
}

impl<'a> LoweringContext<'a> {
    /// Create lowering state around an initialized ONNX graph.
    pub(super) fn new(
        graph: &'a ResolvedExportGraph,
        proto: GraphProto,
        initializer_names: &'a HashMap<TensorId, String>,
        opset: Opset,
    ) -> Self {
        Self {
            graph,
            proto,
            initializer_names,
            opset,
        }
    }

    /// Return the stable ONNX value name for a captured tensor.
    pub(super) fn tensor_name(&self, id: TensorId) -> String {
        self.initializer_names
            .get(&id)
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| super::name(id))
    }

    /// Append an ONNX node without attributes.
    ///
    /// Attributes for this node must be appended immediately afterward through
    /// the attribute helpers on this context.
    pub(super) fn node(
        &mut self,
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) {
        self.proto.node.push(Default::default());
        let node = self.proto.node.last_mut().expect("node was just inserted");
        node.name = name.into();
        node.op_type = op_type.into();
        node.input = inputs;
        node.output = outputs;
    }

    /// Append an `INT64` initializer containing a one-dimensional value list.
    pub(super) fn i64_initializer(&mut self, name: impl Into<String>, values: &[i64]) {
        let mut tensor = TensorProto::new();
        tensor.name = name.into();
        tensor.data_type = 7;
        tensor.dims = vec![values.len() as i64];
        let mut raw = Vec::with_capacity(size_of_val(values));
        for value in values {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        tensor.raw_data = bytes::Bytes::from(raw);
        self.proto.initializer.push(tensor);
    }

    /// Append a scalar initializer encoded according to its Burn dtype.
    pub(super) fn scalar_initializer(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        value: ScalarIr,
        tensor: TensorId,
    ) -> Result<(), ExportError> {
        let mut initializer = scalar_tensor(dtype, value, tensor)?;
        initializer.name = name.into();
        self.proto.initializer.push(initializer);
        Ok(())
    }

    /// Append an integer attribute to the most recently added node.
    pub(super) fn int_attribute(&mut self, name: &str, value: i64) {
        let node = self.proto.node.last_mut().expect("a node must exist");
        node.attribute.push(Default::default());
        let attribute = node.attribute.last_mut().unwrap();
        attribute.name = name.into();
        attribute.type_ = EnumOrUnknown::from_i32(2);
        attribute.i = value;
    }

    /// Append an integer-list attribute to the most recently added node.
    pub(super) fn ints_attribute(&mut self, name: &str, values: impl IntoIterator<Item = usize>) {
        let node = self.proto.node.last_mut().expect("a node must exist");
        node.attribute.push(Default::default());
        let attribute = node.attribute.last_mut().unwrap();
        attribute.name = name.into();
        attribute.type_ = EnumOrUnknown::from_i32(7);
        attribute.ints = values.into_iter().map(|value| value as i64).collect();
    }

    /// Append a float attribute to the most recently added node.
    pub(super) fn float_attribute(&mut self, name: &str, value: f32) {
        let node = self.proto.node.last_mut().expect("a node must exist");
        node.attribute.push(Default::default());
        let attribute = node.attribute.last_mut().unwrap();
        attribute.name = name.into();
        attribute.type_ = EnumOrUnknown::from_i32(1);
        attribute.f = value;
    }

    /// Append a static UTF-8 string attribute to the most recently added node.
    pub(super) fn string_attribute(&mut self, name: &str, value: &'static str) {
        let node = self.proto.node.last_mut().expect("a node must exist");
        node.attribute.push(Default::default());
        let attribute = node.attribute.last_mut().unwrap();
        attribute.name = name.into();
        attribute.type_ = EnumOrUnknown::from_i32(3);
        attribute.s = bytes::Bytes::from_static(value.as_bytes());
    }

    /// Append a tensor attribute to the most recently added node.
    pub(super) fn tensor_attribute(&mut self, name: &str, tensor: TensorProto) {
        let node = self.proto.node.last_mut().expect("a node must exist");
        node.attribute.push(Default::default());
        let attribute = node.attribute.last_mut().unwrap();
        attribute.name = name.into();
        attribute.type_ = EnumOrUnknown::from_i32(4);
        attribute.t = MessageField::some(tensor);
    }

    /// Consume the context, wrap its graph in a model, and serialize it.
    pub(super) fn finish(self) -> Result<OnnxModel, ExportError> {
        let mut model = ModelProto::new();
        model.ir_version = ONNX_IR_VERSION;
        model.producer_name = "burn-onnx-export".into();
        model.graph = MessageField::some(self.proto);
        model.opset_import.push(Default::default());
        model.opset_import[0].version = self.opset.version();
        model
            .write_to_bytes()
            .map(OnnxModel::new)
            .map_err(|error| ExportError::Serialization(error.to_string()))
    }
}
