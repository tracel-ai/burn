/// Master node registry - auto-generates ONNX to Burn conversion infrastructure.
///
/// Generates: Node<PS> enum, imports, match_all! macro, name() method, try_convert_onnx_node()
///
/// # Syntax
///
/// Single mapping: `Add => add as AddNode,`
/// - `Add` = ONNX operation (from `onnx_ir::NodeType`)
/// - `add` = module name
/// - `AddNode` = node struct type
///
/// Grouped mapping: `[ReduceMax, ReduceMin, ...] => ReduceMax: reduce as ReduceNode,`
/// - Maps multiple ONNX ops to one node type
///
/// # Adding a Node
///
/// 1. Implement in `node/<op>.rs`: `<Op>Node` struct with `OnnxIntoNode` + `NodeCodegen` traits
/// 2. Add module in `mod.rs`: `pub(crate) mod <op>;`
/// 3. Add one line to registry in `node_registry.rs`
///
/// See: `contributor-book/src/guides/onnx-to-burn-conversion-tool.md`
macro_rules! node_registry {
    (
        $(
            // Single ONNX op -> Single node type
            $single_onnx:ident => $single_module:ident as $single_node_type:ident
        ),* $(,)?
        $(
            // Multiple ONNX ops -> Single node type (grouped)
            [$($group_onnx:ident),+ $(,)?] => $group_variant:ident: $group_module:ident as $group_node_type:ident
        ),* $(,)?
    ) => {
        // Generate imports (from both single and grouped)
        $(
            pub use super::node::$single_module::$single_node_type;
        )*
        $(
            pub use super::node::$group_module::$group_node_type;
        )*

        // Generate Node enum (one variant per unique node type)
        // Variant names use ONNX operation names for consistency
        #[derive(Debug, Clone)]
        pub enum Node<PS: burn::record::PrecisionSettings> {
            $(
                $single_onnx($single_node_type),
            )*
            $(
                $group_variant($group_node_type),
            )*
            // For now, we have to keep the precision settings in order to correctly serialize the fields
            // into the right data types.
            _Unreachable(std::convert::Infallible, std::marker::PhantomData<PS>),
        }

        // Generate match_all! macro for dispatching on Node variants
        macro_rules! match_all {
            ($self:expr, $func:expr) => {{
                #[allow(clippy::redundant_closure_call)]
                match $self {
                    $(
                        Node::$single_onnx(node) => $func(node),
                    )*
                    $(
                        Node::$group_variant(node) => $func(node),
                    )*
                    Node::_Unreachable(_, _) => unimplemented!(),
                }
            }};
        }

        // Re-export match_all! so it's accessible from parent module
        pub(crate) use match_all;

        // Generate name() method
        impl<PS: burn::record::PrecisionSettings> Node<PS> {
            pub fn name(&self) -> &str {
                match self {
                    $(
                        Node::$single_onnx(_) => stringify!($single_module),
                    )*
                    $(
                        Node::$group_variant(_) => stringify!($group_module),
                    )*
                    Node::_Unreachable(_, _) => unimplemented!(),
                }
            }
        }

        // Generate ONNX registry dispatcher (expands grouped mappings)
        pub(crate) fn try_convert_onnx_node<PS: burn::record::PrecisionSettings>(
            node: onnx_ir::Node,
        ) -> Option<Node<PS>> {
            use onnx_ir::NodeType;
            use super::node_codegen::NodeCodegen;
            use super::node_codegen::OnnxIntoNode;

            match node.node_type {
                // Single mappings
                $(
                    NodeType::$single_onnx => {
                        Some(NodeCodegen::into_node($single_node_type::from_onnx(node)))
                    }
                )*
                // Grouped mappings (expands each ONNX op in the group)
                $(
                    $(
                        NodeType::$group_onnx => {
                            Some(NodeCodegen::into_node($group_node_type::from_onnx(node)))
                        }
                    )+
                )*
                _ => None,
            }
        }
    };
}
