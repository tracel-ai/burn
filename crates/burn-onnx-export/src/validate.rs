use burn_backend::Shape;
use burn_ir::{
    BaseOperationIr, GraphIr, IrVisitorMut, ModuleOperationIr, OperationIr, TensorId, TensorIr,
};
use hashbrown::{HashMap, HashSet};

use crate::ExportError;

/// Validates that two captures have compatible computation structure.
///
/// This validator is used before paired-trace shape resolution. It establishes
/// that operations from independent captures can be compared by position even
/// though their tensor IDs and concrete dimensions differ. It checks boundary
/// ordering, operation ordering and variants, tensor connectivity, the number
/// and ordering of tensor inputs and outputs, dtype, rank, tensor status, and
/// non-shape operation attributes.
///
/// It does not validate initialized tensor values, ONNX operator support, or
/// prove relationships between dynamic dimensions. Shape-sensitive operands
/// are ignored here and must subsequently be resolved or rejected by a
/// shape resolver.
pub struct GraphStructureValidator;

impl GraphStructureValidator {
    /// Compare two captures and return their first structural mismatch.
    pub fn validate(sample: &GraphIr, validation: &GraphIr) -> Result<(), ExportError> {
        validate_boundaries(sample, "sample")?;
        validate_boundaries(validation, "validation")?;
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
        if sample.inputs.len() != validation.inputs.len()
            || sample.outputs.len() != validation.outputs.len()
        {
            return Err(ExportError::DynamicGraphMismatch {
                operation: 0,
                reason: format!(
                    "number of graph inputs or outputs differs (inputs {} != {}, outputs {} != {})",
                    sample.inputs.len(),
                    validation.inputs.len(),
                    sample.outputs.len(),
                    validation.outputs.len()
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
                    reason: "operation variant, tensor inputs or outputs, dtype, rank, tensor status, or non-shape attributes differ".into(),
                });
            }
        }
        Ok(())
    }
}

fn validate_boundaries(graph: &GraphIr, trace: &str) -> Result<(), ExportError> {
    let known: HashSet<_> = graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .map(|tensor| tensor.id)
        .collect();
    for (kind, boundaries) in [("input", &graph.inputs), ("output", &graph.outputs)] {
        let mut unique = HashSet::new();
        for &id in boundaries {
            if !unique.insert(id) {
                return Err(ExportError::DynamicGraphMismatch {
                    operation: 0,
                    reason: format!("{trace} trace has duplicate {kind} boundary {id}"),
                });
            }
            if !known.contains(&id) {
                return Err(ExportError::DynamicGraphMismatch {
                    operation: 0,
                    reason: format!("{trace} trace has unknown {kind} boundary {id}"),
                });
            }
        }
    }
    Ok(())
}

/// Produce the structural form used to compare two independent captures.
///
/// Tensor IDs are allocated by a capture and therefore have no meaning across
/// traces. They are replaced with dense IDs (`0..N`) in first-use order. Every
/// concrete dimension is replaced by zero while the tensor rank is retained,
/// since differing dimension values are expected for dynamic-shape captures.
///
/// The operation order, operation variants, tensor connectivity, boundary
/// positions, tensor rank, status, dtype, and all non-shape operation attributes
/// remain unchanged. Shape-sensitive values are compared later by a
/// shape resolver rather than by this structural representation.
fn normalize_structure(graph: &GraphIr) -> GraphIr {
    let mut ids = HashMap::new();
    let mut next = 0;
    let mut operations = graph.operations.clone();
    for operation in &mut operations {
        operation.visit_mut(&mut StructuralNormalizer {
            ids: &mut ids,
            next: &mut next,
        });
        erase_shape_sensitive_attributes(operation);
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

/// Remove operands whose values are allowed to vary between shape traces.
///
/// The shape resolver is responsible for either producing an explicit runtime
/// expression for these operands or returning `DynamicShapeLost`.
fn erase_shape_sensitive_attributes(operation: &mut OperationIr) {
    match operation {
        OperationIr::BaseFloat(operation)
        | OperationIr::BaseInt(operation)
        | OperationIr::BaseBool(operation) => match operation {
            BaseOperationIr::Slice(operation) => {
                for range in &mut operation.ranges {
                    range.start = 0;
                    range.end = None;
                }
            }
            BaseOperationIr::SliceAssign(operation) => {
                for range in &mut operation.ranges {
                    range.start = 0;
                    range.end = None;
                }
            }
            _ => {}
        },
        OperationIr::Module(ModuleOperationIr::Interpolate(operation)) => {
            operation.output_size = [0; 2];
        }
        _ => {}
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
