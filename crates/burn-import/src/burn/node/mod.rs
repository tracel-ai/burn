mod base;

pub(crate) mod abs;
pub(crate) mod add;
pub(crate) mod argmax;
pub(crate) mod argmin;
pub(crate) mod attention;
pub(crate) mod avg_pool1d;
pub(crate) mod avg_pool2d;
pub(crate) mod batch_norm;
pub(crate) mod bernoulli;
pub(crate) mod bitshift;
pub(crate) mod bitwiseand;
pub(crate) mod bitwisenot;
pub(crate) mod bitwiseor;
pub(crate) mod bitwisexor;
pub(crate) mod bool_and;
pub(crate) mod bool_or;
pub(crate) mod bool_xor;
pub(crate) mod cast;
pub(crate) mod ceil;
pub(crate) mod clip;
pub(crate) mod concat;
pub(crate) mod constant;
pub(crate) mod constant_of_shape;
pub(crate) mod conv1d;
pub(crate) mod conv2d;
pub(crate) mod conv3d;
pub(crate) mod conv_transpose_1d;
pub(crate) mod conv_transpose_2d;
pub(crate) mod conv_transpose_3d;
pub(crate) mod cos;
pub(crate) mod cosh;
pub(crate) mod depth_to_space;
pub(crate) mod div;
pub(crate) mod dropout;
pub(crate) mod equal;
pub(crate) mod erf;
pub(crate) mod exp;
pub(crate) mod expand;
pub(crate) mod eye_like;
pub(crate) mod flatten;
pub(crate) mod floor;
pub(crate) mod gather;
pub(crate) mod gather_elements;
pub(crate) mod gelu;
pub(crate) mod gemm;
pub(crate) mod global_avg_pool;
pub(crate) mod greater;
pub(crate) mod greater_equal;
pub(crate) mod group_norm;
pub(crate) mod hard_sigmoid;
pub(crate) mod identity;
pub(crate) mod instance_norm;
pub(crate) mod is_inf;
pub(crate) mod is_nan;
pub(crate) mod layer_norm;
pub(crate) mod leaky_relu;
pub(crate) mod linear;
pub(crate) mod log;
pub(crate) mod log_softmax;
pub(crate) mod lower;
pub(crate) mod lower_equal;
pub(crate) mod matmul;
pub(crate) mod matmul_integer;
pub(crate) mod max_pair;
pub(crate) mod max_pool1d;
pub(crate) mod max_pool2d;
pub(crate) mod mean;
pub(crate) mod min_pair;
pub(crate) mod modulo;
pub(crate) mod mul;
pub(crate) mod neg;
pub(crate) mod nonzero;
pub(crate) mod not;
pub(crate) mod one_hot;
pub(crate) mod pad;
pub(crate) mod pow;
pub(crate) mod powf;
pub(crate) mod powi;
pub(crate) mod prelu;
pub(crate) mod random_normal;
pub(crate) mod random_normal_like;
pub(crate) mod random_uniform;
pub(crate) mod random_uniform_like;
pub(crate) mod range;
pub(crate) mod reciprocal;
pub(crate) mod reduce;
pub(crate) mod relu;
pub(crate) mod reshape;
pub(crate) mod resize;
pub(crate) mod round;
pub(crate) mod shape;
pub(crate) mod sigmoid;
pub(crate) mod sign;
pub(crate) mod sin;
pub(crate) mod sinh;
pub(crate) mod size;
pub(crate) mod slice;
pub(crate) mod softmax;
pub(crate) mod space_to_depth;
pub(crate) mod split;
pub(crate) mod sqrt;
pub(crate) mod squeeze;
pub(crate) mod sub;
pub(crate) mod sum;
pub(crate) mod tan;
pub(crate) mod tanh;
pub(crate) mod tile;
pub(crate) mod top_k;
pub(crate) mod transpose;
pub(crate) mod trilu;
pub(crate) mod unsqueeze;
pub(crate) mod where_op;
pub(crate) use base::*;

// Auto-generated ONNX node dispatchers
burn_import_macros::onnx_node_registry! {
    Add => add,
    Sub => sub,
    Mul => mul,
    Div => div,
    Max => max_pair,
    Min => min_pair,
    MatMul => matmul,
    Equal => equal,
    Greater => greater,
    GreaterOrEqual => greater_equal,
    Less => lower,
    LessOrEqual => lower_equal,
}

burn_import_macros::onnx_node_registry! {
    And => bool_and,
    Or => bool_or,
    Xor => bool_xor,
    Abs => abs,
    Ceil => ceil,
    Cos => cos,
    Cosh => cosh,
    Erf => erf,
    Exp => exp,
    Floor => floor,
    Identity => identity,
}

burn_import_macros::onnx_node_registry! {
    Log => log,
    Neg => neg,
    Not => not,
    Reciprocal => reciprocal,
    Round => round,
    Sigmoid => sigmoid,
    Sign => sign,
    Sin => sin,
    Sinh => sinh,
    Sqrt => sqrt,
    Tan => tan,
    Tanh => tanh,
}

burn_import_macros::onnx_node_registry! {
    ArgMax => argmax,
    ArgMin => argmin,
    Attention => attention,
    AveragePool1d => avg_pool1d,
    AveragePool2d => avg_pool2d,
    BatchNormalization => batch_norm,
    Bernoulli => bernoulli,
    BitShift => bitshift,
    BitwiseAnd => bitwiseand,
    BitwiseNot => bitwisenot,
    BitwiseOr => bitwiseor,
    BitwiseXor => bitwisexor,
}

burn_import_macros::onnx_node_registry! {
    Cast => cast,
    Clip => clip,
    Concat => concat,
    Constant => constant,
    ConstantOfShape => constant_of_shape,
    Conv1d => conv1d,
    Conv2d => conv2d,
    Conv3d => conv3d,
    ConvTranspose1d => conv_transpose_1d,
    ConvTranspose2d => conv_transpose_2d,
    ConvTranspose3d => conv_transpose_3d,
}

burn_import_macros::onnx_node_registry! {
    DepthToSpace => depth_to_space,
    Dropout => dropout,
    Expand => expand,
    EyeLike => eye_like,
    Flatten => flatten,
    Gather => gather,
    GatherElements => gather_elements,
    Gelu => gelu,
    Gemm => gemm,
    GlobalAveragePool => global_avg_pool,
    GroupNormalization => group_norm,
    HardSigmoid => hard_sigmoid,
}

burn_import_macros::onnx_node_registry! {
    InstanceNormalization => instance_norm,
    IsInf => is_inf,
    IsNaN => is_nan,
    LayerNormalization => layer_norm,
    LeakyRelu => leaky_relu,
    Linear => linear,
    LogSoftmax => log_softmax,
    MatMulInteger => matmul_integer,
    MaxPool1d => max_pool1d,
    MaxPool2d => max_pool2d,
    Mean => mean,
    Mod => modulo,
}

burn_import_macros::onnx_node_registry! {
    NonZero => nonzero,
    OneHot => one_hot,
    Pad => pad,
    Pow => pow,
    PRelu => prelu,
    RandomNormal => random_normal,
    RandomNormalLike => random_normal_like,
    RandomUniform => random_uniform,
    RandomUniformLike => random_uniform_like,
    Range => range,
    Relu => relu,
    Reshape => reshape,
    Resize => resize,
}

burn_import_macros::onnx_node_registry! {
    ReduceMax => reduce,
    ReduceMin => reduce,
    ReduceMean => reduce,
    ReduceProd => reduce,
    ReduceSum => reduce,
    ReduceSumSquare => reduce,
    ReduceL1 => reduce,
    ReduceL2 => reduce,
    ReduceLogSum => reduce,
    ReduceLogSumExp => reduce,
    Shape => shape,
    Size => size,
}

burn_import_macros::onnx_node_registry! {
    Slice => slice,
    Softmax => softmax,
    SpaceToDepth => space_to_depth,
    Split => split,
    Squeeze => squeeze,
    Sum => sum,
    Tile => tile,
    TopK => top_k,
    Transpose => transpose,
    Trilu => trilu,
    Unsqueeze => unsqueeze,
    Where => where_op,
}

// Combined dispatcher
pub(crate) fn try_convert_onnx_node<PS: burn::record::PrecisionSettings>(
    node: onnx_ir::Node,
) -> Option<Node<PS>> {
    onnx_dispatch_add::try_convert_onnx_node_add(node.clone())
        .or_else(|| onnx_dispatch_and::try_convert_onnx_node_and(node.clone()))
        .or_else(|| onnx_dispatch_log::try_convert_onnx_node_log(node.clone()))
        .or_else(|| onnx_dispatch_argmax::try_convert_onnx_node_argmax(node.clone()))
        .or_else(|| onnx_dispatch_cast::try_convert_onnx_node_cast(node.clone()))
        .or_else(|| onnx_dispatch_depthtospace::try_convert_onnx_node_depthtospace(node.clone()))
        .or_else(|| {
            onnx_dispatch_instancenormalization::try_convert_onnx_node_instancenormalization(
                node.clone(),
            )
        })
        .or_else(|| onnx_dispatch_nonzero::try_convert_onnx_node_nonzero(node.clone()))
        .or_else(|| onnx_dispatch_reducemax::try_convert_onnx_node_reducemax(node.clone()))
        .or_else(|| onnx_dispatch_slice::try_convert_onnx_node_slice(node))
}

#[cfg(test)]
pub(crate) mod test;
