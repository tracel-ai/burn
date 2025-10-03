/// Master node registry - single source of truth for all ONNX nodes
///
/// To add a new node:
/// 1. Create the node file (e.g., my_node.rs) with from_onnx() implementation
/// 2. Add module declaration in mod.rs: pub(crate) mod my_node;
/// 3. Add one line to registry:
///    - Single mapping: `Add => add as AddNode,`
///    - Grouped mapping: `[ReduceMax, ReduceMin, ...] => ReduceMax: reduce as ReduceNode,`
///                                                         ^^^^^^^^ variant name (uses first ONNX op by convention)
/// 4. Done! Node enum, imports, dispatch, and ONNX conversion all auto-generated.
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
            pub use super::$single_module::$single_node_type;
        )*
        $(
            pub use super::$group_module::$group_node_type;
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
            use super::NodeCodegen;
            use super::OnnxIntoNode;

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

// Master registry - ALL nodes declared here
node_registry! {
    // Binary ops
    Add => add as AddNode,
    Sub => sub as SubNode,
    Mul => mul as MulNode,
    Div => div as DivNode,
    Max => max_pair as MaxPairNode,
    Min => min_pair as MinPairNode,
    MatMul => matmul as MatmulNode,

    // Comparison ops
    Equal => equal as EqualNode,
    Greater => greater as GreaterNode,
    GreaterOrEqual => greater_equal as GreaterEqualNode,
    Less => lower as LowerNode,
    LessOrEqual => lower_equal as LowerEqualNode,

    // Boolean ops
    And => bool_and as BoolAndNode,
    Or => bool_or as BoolOrNode,
    Xor => bool_xor as BoolXorNode,

    // Unary ops
    Abs => abs as AbsNode,
    Ceil => ceil as CeilNode,
    Cos => cos as CosNode,
    Cosh => cosh as CoshNode,
    Erf => erf as ErfNode,
    Exp => exp as ExpNode,
    Floor => floor as FloorNode,
    Identity => identity as IdentityNode,
    Log => log as LogNode,
    Neg => neg as NegNode,
    Not => not as NotNode,
    Reciprocal => reciprocal as ReciprocalNode,
    Round => round as RoundNode,
    Sigmoid => sigmoid as SigmoidNode,
    Sign => sign as SignNode,
    Sin => sin as SinNode,
    Sinh => sinh as SinhNode,
    Sqrt => sqrt as SqrtNode,
    Tan => tan as TanNode,
    Tanh => tanh as TanhNode,

    // Activation ops
    Relu => relu as ReluNode,
    Gelu => gelu as GeluNode,
    LeakyRelu => leaky_relu as LeakyReluNode,
    HardSigmoid => hard_sigmoid as HardSigmoidNode,
    Softmax => softmax as SoftmaxNode,
    LogSoftmax => log_softmax as LogSoftmaxNode,
    PRelu => prelu as PReluNode,

    // Shape ops
    Reshape => reshape as ReshapeNode,
    Flatten => flatten as FlattenNode,
    Squeeze => squeeze as SqueezeNode,
    Unsqueeze => unsqueeze as UnsqueezeNode,
    Transpose => transpose as TransposeNode,
    Shape => shape as ShapeNode,
    Size => size as SizeNode,

    // Tensor ops
    Concat => concat as ConcatNode,
    Split => split as SplitNode,
    Slice => slice as SliceNode,
    Gather => gather as GatherNode,
    GatherElements => gather_elements as GatherElementsNode,
    Tile => tile as TileNode,
    Expand => expand as ExpandNode,
    Pad => pad as PadNode,

    // Convolution ops
    Conv1d => conv1d as Conv1dNode,
    Conv2d => conv2d as Conv2dNode,
    Conv3d => conv3d as Conv3dNode,
    ConvTranspose1d => conv_transpose_1d as ConvTranspose1dNode,
    ConvTranspose2d => conv_transpose_2d as ConvTranspose2dNode,
    ConvTranspose3d => conv_transpose_3d as ConvTranspose3dNode,

    // Pooling ops
    AveragePool1d => avg_pool1d as AvgPool1dNode,
    AveragePool2d => avg_pool2d as AvgPool2dNode,
    MaxPool1d => max_pool1d as MaxPool1dNode,
    MaxPool2d => max_pool2d as MaxPool2dNode,
    GlobalAveragePool => global_avg_pool as GlobalAvgPoolNode,

    // Normalization ops
    BatchNormalization => batch_norm as BatchNormNode,
    LayerNormalization => layer_norm as LayerNormNode,
    GroupNormalization => group_norm as GroupNormNode,
    InstanceNormalization => instance_norm as InstanceNormNode,

    // Other ops
    Cast => cast as CastNode,
    Clip => clip as ClipNode,
    Dropout => dropout as DropoutNode,
    Where => where_op as WhereNode,
    ArgMax => argmax as ArgMaxNode,
    ArgMin => argmin as ArgMinNode,
    TopK => top_k as TopKNode,
    NonZero => nonzero as NonZeroNode,
    OneHot => one_hot as OneHotNode,
    Pow => pow as PowNode,
    Mod => modulo as ModNode,
    Trilu => trilu as TriluNode,

    // Bitwise ops
    BitShift => bitshift as BitShiftNode,
    BitwiseAnd => bitwiseand as BitwiseAndNode,
    BitwiseOr => bitwiseor as BitwiseOrNode,
    BitwiseXor => bitwisexor as BitwiseXorNode,
    BitwiseNot => bitwisenot as BitwiseNotNode,

    // Math ops
    Sum => sum as SumNode,
    Mean => mean as MeanNode,
    Gemm => gemm as GemmNode,
    Linear => linear as LinearNode,
    MatMulInteger => matmul_integer as MatMulIntegerNode,

    // Constant ops
    Constant => constant as ConstantNode,
    ConstantOfShape => constant_of_shape as ConstantOfShapeNode,
    EyeLike => eye_like as EyeLikeNode,
    Range => range as RangeNode,

    // Random ops
    RandomNormal => random_normal as RandomNormalNode,
    RandomNormalLike => random_normal_like as RandomNormalLikeNode,
    RandomUniform => random_uniform as RandomUniformNode,
    RandomUniformLike => random_uniform_like as RandomUniformLikeNode,
    Bernoulli => bernoulli as BernoulliNode,

    // Spatial ops
    DepthToSpace => depth_to_space as DepthToSpaceNode,
    SpaceToDepth => space_to_depth as SpaceToDepthNode,
    Resize => resize as ResizeNode,

    // Test ops
    IsInf => is_inf as IsInfNode,
    IsNaN => is_nan as IsNanNode,

    // Special ops
    Attention => attention as AttentionNode,

    // Grouped mappings: Multiple ONNX ops -> Single node type
    [ReduceMax, ReduceMin, ReduceMean, ReduceProd, ReduceSum,
     ReduceSumSquare, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp]
        => ReduceMax: reduce as ReduceNode,
}
