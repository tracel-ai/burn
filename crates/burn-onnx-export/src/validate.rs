use burn_backend::Shape;
use burn_ir::{GraphIr, IrVisitorMut, TensorId, TensorIr};
use hashbrown::HashMap;

use crate::ExportError;

/// Validates that captures at different shapes have identical topology.
pub struct GraphStructureValidator;

impl GraphStructureValidator {
    /// Compare operation variants, arity, dtypes, boundaries, and non-shape attributes.
    pub fn validate(sample: &GraphIr, validation: &GraphIr) -> Result<(), ExportError> {
        if sample.operations.len() != validation.operations.len() {
            return Err(ExportError::DynamicGraphMismatch {
                operation: sample.operations.len().min(validation.operations.len()),
                reason: format!(
                    "operation count differs ({} != {})",
                    sample.operations.len(),
                    validation.operations.len()
                ),
            });
        }

        let sample = normalize_structure(sample);
        let validation = normalize_structure(validation);
        if sample.inputs != validation.inputs || sample.outputs != validation.outputs {
            return Err(ExportError::DynamicGraphMismatch {
                operation: 0,
                reason: "declared graph boundaries differ".into(),
            });
        }
        for (index, (lhs, rhs)) in sample
            .operations
            .iter()
            .zip(&validation.operations)
            .enumerate()
        {
            if lhs != rhs {
                return Err(ExportError::DynamicGraphMismatch {
                    operation: index,
                    reason: "operation kind, tensor arity/dtype, or static attributes differ"
                        .into(),
                });
            }
        }
        Ok(())
    }
}

/// Produce the structural form used to compare two independent captures.
///
/// Tensor IDs are allocated by a capture and therefore have no meaning across
/// traces. They are replaced with dense IDs (`0..N`) in first-use order. Every
/// concrete dimension is replaced by zero while the tensor rank is retained,
/// since differing dimension values are expected for dynamic-shape captures.
///
/// The operation order, operation variants, tensor connectivity, boundary
/// positions, tensor rank/status/dtype, and all non-shape operation attributes
/// remain unchanged. Shape-sensitive values are compared later by a
/// `ShapeResolver` rather than by this structural representation.
fn normalize_structure(graph: &GraphIr) -> GraphIr {
    let mut ids = HashMap::new();
    let mut next = 0;
    let mut operations = graph.operations.clone();
    for operation in &mut operations {
        operation.visit_mut(&mut StructuralNormalizer {
            ids: &mut ids,
            next: &mut next,
        });
    }
    let map_boundary = |boundary: &[TensorId]| {
        boundary
            .iter()
            .filter_map(|id| ids.get(id).copied())
            .collect()
    };
    GraphIr {
        inputs: map_boundary(&graph.inputs),
        outputs: map_boundary(&graph.outputs),
        operations,
    }
}

/// Operation visitor implementing trace-independent tensor normalization.
///
/// A single instance is shared across the complete operation sequence so every
/// repeated occurrence of a tensor receives the same normalized ID.
struct StructuralNormalizer<'a> {
    ids: &'a mut HashMap<TensorId, TensorId>,
    next: &'a mut u64,
}

impl IrVisitorMut for StructuralNormalizer<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        // Assignment on first use preserves graph connectivity without relying
        // on capture-local allocation order or numeric tensor IDs.
        tensor.id = *self.ids.entry(tensor.id).or_insert_with(|| {
            let id = TensorId::new(*self.next);
            *self.next += 1;
            id
        });
        // Erase dimension values, but preserve rank for structural comparison.
        tensor.shape = core::iter::repeat_n(0, tensor.shape.num_dims()).collect::<Shape>();
    }
}
