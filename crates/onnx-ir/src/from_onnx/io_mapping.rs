//! Input/Output name mapping for ONNX graph conversion
//!
//! This module manages the mapping between ONNX names and IR names during
//! graph conversion, tracking which graph inputs have been used by nodes.

use std::collections::{HashMap, HashSet};

/// Represents where an input comes from - either a graph input or a node output
#[derive(Debug, Clone)]
pub(crate) enum IOEntry {
    /// Input from a graph input at the given index
    In(usize),
    /// Input from a node output (node_index, output_index)
    Node(usize, usize),
}

/// Manages input/output name mapping during ONNX graph conversion
#[derive(Debug)]
pub(super) struct IOMapper {
    /// Maps the original input name to a graph input or node output
    input_name_map: HashMap<String, IOEntry>,
    /// Maps the updated input name to the original input name
    input_key_map: HashMap<String, String>,
    /// Tracks which graph inputs have been used by nodes
    passed_inputs: HashSet<usize>,
}

impl IOMapper {
    /// Create a new empty IOMapper
    pub(super) fn new() -> Self {
        Self {
            input_name_map: HashMap::new(),
            input_key_map: HashMap::new(),
            passed_inputs: HashSet::new(),
        }
    }

    /// Register an initializer as a node output
    pub(super) fn register_initializer(&mut self, name: String, node_idx: usize) {
        self.input_name_map.insert(name, IOEntry::Node(node_idx, 0));
    }

    /// Register a graph input
    pub(super) fn register_input(&mut self, original_name: String, new_name: String, idx: usize) {
        // Only add to input_name_map if not already mapped to a constant
        if !self.input_name_map.contains_key(&original_name) {
            self.input_name_map
                .insert(original_name.clone(), IOEntry::In(idx));
        }
        self.input_key_map.insert(new_name, original_name);
    }

    /// Register a node output
    pub(super) fn register_node_output(
        &mut self,
        output_name: String,
        node_idx: usize,
        output_idx: usize,
    ) {
        self.input_name_map
            .insert(output_name, IOEntry::Node(node_idx, output_idx));
    }

    /// Look up the source of an input by original name
    pub(super) fn lookup(&self, original_name: &str) -> Option<&IOEntry> {
        self.input_name_map.get(original_name)
    }

    /// Mark a graph input as used
    pub(super) fn mark_input_used(&mut self, new_name: &str) {
        if let Some(old_input_name) = self.input_key_map.get(new_name)
            && let Some(IOEntry::In(i)) = self.input_name_map.get(old_input_name)
        {
            // Only In entries are graph inputs; Node entries are initializers/node outputs
            self.passed_inputs.insert(*i);
        }
    }

    /// Get the set of used graph input indices
    pub(super) fn passed_inputs(&self) -> &HashSet<usize> {
        &self.passed_inputs
    }
}
