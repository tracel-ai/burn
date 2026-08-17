//! Lowering for fused operations from [`burn_ir::ModuleOperationIr`].
//!
//! A Burn module operation may emit one or several ONNX nodes. This module also
//! owns opset-specific attributes for convolution, normalization, resize, and
//! pooling operations.

use burn_ir::{InterpolateModeIr, ModuleOperationIr, OperationIr};

use crate::ExportError;

use super::context::LoweringContext;

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    let OperationIr::Module(operation) = operation else {
        return Ok(false);
    };
    match operation {
        ModuleOperationIr::Conv2d(conv) => {
            let mut inputs = vec![
                context.tensor_name(conv.x.id),
                context.tensor_name(conv.weight.id),
            ];
            if let Some(bias) = &conv.bias {
                inputs.push(context.tensor_name(bias.id));
            }
            let output = context.tensor_name(conv.out.id);
            context.node(format!("node_{index}"), "Conv", inputs, vec![output]);
            context.ints_attribute("strides", conv.options.stride);
            context.ints_attribute("dilations", conv.options.dilation);
            context.ints_attribute(
                "pads",
                [
                    conv.options.padding[0],
                    conv.options.padding[1],
                    conv.options.padding[0],
                    conv.options.padding[1],
                ],
            );
            context.int_attribute("group", conv.options.groups as i64);
            Ok(true)
        }
        ModuleOperationIr::BatchNorm(batch_norm) => {
            let inputs = vec![
                context.tensor_name(batch_norm.x.id),
                context.tensor_name(batch_norm.gamma.id),
                context.tensor_name(batch_norm.beta.id),
                context.tensor_name(batch_norm.mean.id),
                context.tensor_name(batch_norm.variance.id),
            ];
            let output = context.tensor_name(batch_norm.out.id);
            context.node(
                format!("node_{index}"),
                "BatchNormalization",
                inputs,
                vec![output],
            );
            context.float_attribute("epsilon", batch_norm.epsilon.elem::<f32>());
            Ok(true)
        }
        ModuleOperationIr::Interpolate(interpolate) => {
            let (mode, coordinate_mode, nearest_mode) = match interpolate.options.mode {
                InterpolateModeIr::Nearest => ("nearest", "asymmetric", Some("floor")),
                InterpolateModeIr::NearestExact => {
                    ("nearest", "half_pixel", Some("round_prefer_floor"))
                }
                InterpolateModeIr::Bilinear => (
                    "linear",
                    if interpolate.options.align_corners {
                        "align_corners"
                    } else {
                        "half_pixel"
                    },
                    None,
                ),
                InterpolateModeIr::Bicubic => (
                    "cubic",
                    if interpolate.options.align_corners {
                        "align_corners"
                    } else {
                        "half_pixel"
                    },
                    None,
                ),
                InterpolateModeIr::Lanczos3 => {
                    return Err(ExportError::UnsupportedOperation {
                        operation: index,
                        kind: "Lanczos3 interpolation has no ONNX Resize mode".into(),
                    });
                }
            };
            let sizes = format!("node_{index}_sizes");
            context.i64_initializer(
                sizes.clone(),
                &interpolate.output_size.map(|dimension| dimension as i64),
            );
            let input = context.tensor_name(interpolate.x.id);
            let output = context.tensor_name(interpolate.out.id);
            context.node(
                format!("node_{index}"),
                "Resize",
                vec![input, String::new(), String::new(), sizes],
                vec![output],
            );
            context.ints_attribute("axes", [2, 3]);
            context.string_attribute("mode", mode);
            context.string_attribute("coordinate_transformation_mode", coordinate_mode);
            if let Some(nearest_mode) = nearest_mode {
                context.string_attribute("nearest_mode", nearest_mode);
            }
            if matches!(interpolate.options.mode, InterpolateModeIr::Bicubic) {
                context.float_attribute("cubic_coeff_a", -0.75);
            }
            Ok(true)
        }
        ModuleOperationIr::MaxPool2d(pool) => {
            let input = context.tensor_name(pool.x.id);
            let output = context.tensor_name(pool.out.id);
            context.node(
                format!("node_{index}"),
                "MaxPool",
                vec![input],
                vec![output],
            );
            context.ints_attribute("kernel_shape", pool.kernel_size);
            context.ints_attribute("strides", pool.stride);
            context.ints_attribute("dilations", pool.dilation);
            context.ints_attribute(
                "pads",
                [
                    pool.padding[0],
                    pool.padding[1],
                    pool.padding[0],
                    pool.padding[1],
                ],
            );
            context.int_attribute("ceil_mode", pool.ceil_mode as i64);
            Ok(true)
        }
        ModuleOperationIr::AvgPool2d(pool) => {
            let input = context.tensor_name(pool.x.id);
            let output = context.tensor_name(pool.out.id);
            context.node(
                format!("node_{index}"),
                "AveragePool",
                vec![input],
                vec![output],
            );
            context.ints_attribute("kernel_shape", pool.kernel_size);
            context.ints_attribute("strides", pool.stride);
            context.ints_attribute(
                "pads",
                [
                    pool.padding[0],
                    pool.padding[1],
                    pool.padding[0],
                    pool.padding[1],
                ],
            );
            context.int_attribute("ceil_mode", pool.ceil_mode as i64);
            context.int_attribute("count_include_pad", pool.count_include_pad as i64);
            Ok(true)
        }
        ModuleOperationIr::AdaptiveAvgPool2d(pool) if pool.output_size == [1, 1] => {
            let input = context.tensor_name(pool.x.id);
            let output = context.tensor_name(pool.out.id);
            context.node(
                format!("node_{index}"),
                "GlobalAveragePool",
                vec![input],
                vec![output],
            );
            Ok(true)
        }
        ModuleOperationIr::Linear(linear) => {
            let output = context.tensor_name(linear.out.id);
            let matmul_output = if linear.bias.is_some() {
                format!("node_{index}_matmul")
            } else {
                output.clone()
            };
            let x = context.tensor_name(linear.x.id);
            let weight = context.tensor_name(linear.weight.id);
            context.node(
                format!("node_{index}_matmul"),
                "MatMul",
                vec![x, weight],
                vec![matmul_output.clone()],
            );
            if let Some(bias) = &linear.bias {
                let bias = context.tensor_name(bias.id);
                context.node(
                    format!("node_{index}_bias"),
                    "Add",
                    vec![matmul_output, bias],
                    vec![output],
                );
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
