use onnx_ir::parse_onnx;
use std::path::Path;

const CURRENT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parse_special_chars");

#[test]
fn parse_model_with_special_chars() {
    // The path to the test model with special characters in node names
    let model_path = Path::new(CURRENT_DIR).join("special_char_nodes.onnx");

    // This should not panic with "invalid identifier" errors if our fix is working
    let graph = parse_onnx(&model_path);

    // Verify that the graph was loaded successfully
    assert!(!graph.nodes.is_empty(), "Graph should have nodes");

    // Verify that input/output names were properly sanitized
    let problematic_outputs = graph
        .nodes
        .iter()
        .filter(|node| {
            node.outputs.iter().any(|output| {
                output.name.contains('/')
                    || output.name.contains(':')
                    || !is_valid_identifier(&output.name)
            })
        })
        .count();

    assert_eq!(
        problematic_outputs, 0,
        "All node outputs should have valid sanitized names without ':' or '/' characters"
    );

    // Check that input names were properly sanitized
    let problematic_inputs = graph
        .nodes
        .iter()
        .filter(|node| {
            node.inputs.iter().any(|input| {
                input.name.contains('/')
                    || input.name.contains(':')
                    || !is_valid_identifier(&input.name)
            })
        })
        .count();

    assert_eq!(
        problematic_inputs, 0,
        "All node inputs should have valid sanitized names without ':' or '/' characters"
    );

    // Ensure node names don't contain problematic characters
    let problematic_node_names = graph
        .nodes
        .iter()
        .filter(|node| {
            node.name.contains('/') || node.name.contains(':') || !is_valid_identifier(&node.name)
        })
        .count();

    assert_eq!(
        problematic_node_names, 0,
        "All node names should be valid identifiers without ':' or '/' characters"
    );
}

// Helper function to check if a string is a valid Rust identifier
fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let first_char = s.chars().next().unwrap();
    if !first_char.is_alphabetic() && first_char != '_' {
        return false;
    }

    s.chars().all(|c| c.is_alphanumeric() || c == '_')
}
