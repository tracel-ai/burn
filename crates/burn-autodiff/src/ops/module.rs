use crate::checkpoint::base::Checkpointer;
use crate::checkpoint::strategy::CheckpointStrategy;
use crate::grads::Gradients;
use crate::graph::NodeID;
use crate::ops::{unary, Backward, Ops};
use crate::tensor::AutodiffTensor;
use crate::Autodiff;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

use super::OpsKind;

impl<B: Backend, C: CheckpointStrategy> ModuleOps<Autodiff<B, C>> for Autodiff<B, C> {
    fn embedding(weights: AutodiffTensor<B>, indices: IntTensor<B>) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Embedding;

        impl<B: Backend> Backward<B, 1> for Embedding {
            type State = (B::FloatTensorPrimitive, IntTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (weights, indices) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::embedding_backward(weights, grad, indices)
                });
            }
        }

        match Embedding
            .prepare::<C>([weights.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (weights.primitive.clone(), indices.clone()),
                B::embedding(weights.primitive, indices),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::embedding(weights.primitive, indices)),
        }
    }

    fn embedding_backward(
        _weights: AutodiffTensor<B>,
        _output: AutodiffTensor<B>,
        _indices: IntTensor<B>,
    ) -> AutodiffTensor<B> {
        panic!("Can't differentiate embedding backward.");
    }

    fn conv1d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<1>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv1DWithBias;
        #[derive(Debug)]
        struct Conv1DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv1DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv1d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv1d_weight_backward(x.clone(), weight, grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv1d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv1DNoBias {
            type State = (NodeID, NodeID, ConvOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv1d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv1d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }
        match bias {
            Some(bias) => match Conv1DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv1d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv1d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv1DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv1d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv1d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn conv_transpose1d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose1DWithBias;
        #[derive(Debug)]
        struct ConvTranspose1DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose1DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvTransposeOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose1d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose1d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose1d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose1DNoBias {
            type State = (NodeID, NodeID, ConvTransposeOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose1d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose1d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose1DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose1d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose1DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose1d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    fn conv2d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv2DWithBias;
        #[derive(Debug)]
        struct Conv2DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv2DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv2d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad =
                        B::conv2d_weight_backward(x.clone(), weight.clone(), grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv2d_bias_backward(x, weight, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv2DNoBias {
            type State = (NodeID, NodeID, ConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv2d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv2d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match Conv2DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv2d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv2d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv2DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv2d(x.primitive, weight.primitive, None, options),
                    )
                }

                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv2d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn deform_conv2d(
        x: AutodiffTensor<B>,
        offset: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        mask: Option<AutodiffTensor<B>>,
        bias: Option<AutodiffTensor<B>>,
        options: DeformConvOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct DeformConv2DWithMaskWithBias;
        #[derive(Debug)]
        struct DeformConv2DWithMaskNoBias;
        #[derive(Debug)]
        struct DeformConv2DNoMaskWithBias;
        #[derive(Debug)]
        struct DeformConv2DNoMaskNoBias;

        impl<B: Backend> Backward<B, 5> for DeformConv2DWithMaskWithBias {
            type State = (NodeID, NodeID, NodeID, NodeID, NodeID, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_mask, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, mask_state, bias_state, options) =
                    ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let mask = Some(checkpointer.retrieve_node_output(mask_state));
                let bias = Some(checkpointer.retrieve_node_output(bias_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, mask, bias, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_mask {
                    grads.register::<B>(node.id, backward.mask_grad.unwrap())
                }
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 4> for DeformConv2DWithMaskNoBias {
            type State = (NodeID, NodeID, NodeID, NodeID, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_mask] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, mask_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let mask = Some(checkpointer.retrieve_node_output(mask_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, mask, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_mask {
                    grads.register::<B>(node.id, backward.mask_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 4> for DeformConv2DNoMaskWithBias {
            type State = (NodeID, NodeID, NodeID, NodeID, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let bias = Some(checkpointer.retrieve_node_output(bias_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, None, bias, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 3> for DeformConv2DNoMaskNoBias {
            type State = (NodeID, NodeID, NodeID, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, None, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
            }
        }

        match (mask, bias) {
            (Some(mask), Some(bias)) => match DeformConv2DWithMaskWithBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    mask.node.clone(),
                    bias.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let mask_state = prep.checkpoint(&mask);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            mask_state,
                            bias_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            Some(mask.primitive),
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    Some(mask.primitive),
                    Some(bias.primitive),
                    options,
                )),
            },
            (Some(mask), None) => match DeformConv2DWithMaskNoBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    mask.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let mask_state = prep.checkpoint(&mask);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            mask_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            Some(mask.primitive),
                            None,
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    Some(mask.primitive),
                    None,
                    options,
                )),
            },
            (None, Some(bias)) => match DeformConv2DNoMaskWithBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    bias.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            bias_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            None,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    None,
                    Some(bias.primitive),
                    options,
                )),
            },
            (None, None) => match DeformConv2DNoMaskNoBias
                .prepare::<C>([x.node.clone(), offset.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, offset_state, weight_state, options.clone()),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            None,
                            None,
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    None,
                    None,
                    options,
                )),
            },
        }
    }

    fn deform_conv2d_backward(
        _x: AutodiffTensor<B>,
        _offset: AutodiffTensor<B>,
        _weight: AutodiffTensor<B>,
        _mask: Option<AutodiffTensor<B>>,
        _bias: Option<AutodiffTensor<B>>,
        _output_grad: AutodiffTensor<B>,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        panic!("Can't differentiate deform conv 2d backward.");
    }

    fn conv_transpose2d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose2DWithBias;
        #[derive(Debug)]
        struct ConvTranspose2DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose2DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvTransposeOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose2d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose2d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose2d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose2DNoBias {
            type State = (NodeID, NodeID, ConvTransposeOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose2d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose2d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose2DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);

                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose2d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose2DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);

                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose2d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    fn conv3d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<3>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv3DWithBias;
        #[derive(Debug)]
        struct Conv3DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv3DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv3d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad =
                        B::conv3d_weight_backward(x.clone(), weight.clone(), grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv3d_bias_backward(x, weight, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv3DNoBias {
            type State = (NodeID, NodeID, ConvOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv3d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv3d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match Conv3DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv3d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv3d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv3DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv3d(x.primitive, weight.primitive, None, options),
                    )
                }

                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv3d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn conv_transpose3d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<3>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose3DWithBias;
        #[derive(Debug)]
        struct ConvTranspose3DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose3DWithBias {
            type State = (NodeID, NodeID, NodeID, ConvTransposeOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose3d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose3d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose3d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose3DNoBias {
            type State = (NodeID, NodeID, ConvTransposeOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose3d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose3d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose3DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);

                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose3d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose3d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose3DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);

                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose3d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose3d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    // TODO: Support a custom unfold4d operation by overriding the default implementation.
    //
    // We don't override it now because the fold operation isn't available for the backward pass.
    // This implies that when autodiff is enabled, custom unfold operations defined by backends
    // won't be used. Instead, the conv2d operation with custom weights matrix will be used.
    // Therefore, the conv2d backward pass will be used for the unfold4d backward pass.
    //
    // fn unfold4d(
    //     x:AutodiffTensor<B>,
    //     kernel_size: [usize; 2],
    //     options: UnfoldOptions,
    // ) -> AutodiffTensor<B> {
    //     todo!()
    // }

    fn avg_pool1d(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AvgPool1D;

        impl<B: Backend> Backward<B, 1> for AvgPool1D {
            type State = (NodeID, usize, usize, usize, bool);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, kernel_size, stride, padding, count_include_pad) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::avg_pool1d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    );
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AvgPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(
                    (x_state, kernel_size, stride, padding, count_include_pad),
                    B::avg_pool1d(
                        x.primitive.clone(),
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    ),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::avg_pool1d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            )),
        }
    }

    fn avg_pool2d(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AvgPool2D;

        impl<B: Backend> Backward<B, 1> for AvgPool2D {
            type State = (NodeID, [usize; 2], [usize; 2], [usize; 2], bool);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, kernel_size, stride, padding, count_include_pad) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::avg_pool2d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    );
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AvgPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(
                    (x_state, kernel_size, stride, padding, count_include_pad),
                    B::avg_pool2d(
                        x.primitive.clone(),
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    ),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::avg_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            )),
        }
    }

    fn avg_pool2d_backward(
        _x: AutodiffTensor<B>,
        _grad: AutodiffTensor<B>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
    ) -> AutodiffTensor<B> {
        panic!("Can't differentiate avg pool 2d backward.");
    }

    fn max_pool1d(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> AutodiffTensor<B> {
        match MaxPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output =
                    B::max_pool1d_with_indices(x.primitive, kernel_size, stride, padding, dilation);
                prep.finish(
                    (
                        x_state,
                        output.indices,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    ),
                    output.output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::max_pool1d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
            )),
        }
    }

    fn max_pool1d_with_indices(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Self> {
        match MaxPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output =
                    B::max_pool1d_with_indices(x.primitive, kernel_size, stride, padding, dilation);

                let output_tensor = prep.finish(
                    (
                        x_state,
                        output.indices.clone(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    ),
                    output.output,
                );

                MaxPool1dWithIndices::new(output_tensor, output.indices)
            }
            OpsKind::UnTracked(prep) => {
                let output =
                    B::max_pool1d_with_indices(x.primitive, kernel_size, stride, padding, dilation);
                let output_tensor = prep.finish(output.output);

                MaxPool1dWithIndices::new(output_tensor, output.indices)
            }
        }
    }

    fn max_pool1d_with_indices_backward(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: AutodiffTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool1dBackward<Self> {
        let output = B::max_pool1d_with_indices_backward(
            x.primitive,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad.primitive,
            indices,
        );
        MaxPool1dBackward::new(AutodiffTensor::new(output.x_grad))
    }

    fn max_pool2d(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> AutodiffTensor<B> {
        match MaxPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output =
                    B::max_pool2d_with_indices(x.primitive, kernel_size, stride, padding, dilation);
                prep.finish(
                    (
                        x_state,
                        output.indices,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    ),
                    output.output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::max_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
            )),
        }
    }

    fn max_pool2d_with_indices(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        match MaxPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);

                let output =
                    B::max_pool2d_with_indices(x.primitive, kernel_size, stride, padding, dilation);

                let output_tensor = prep.finish(
                    (
                        x_state,
                        output.indices.clone(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                    ),
                    output.output,
                );

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
            OpsKind::UnTracked(prep) => {
                let output =
                    B::max_pool2d_with_indices(x.primitive, kernel_size, stride, padding, dilation);
                let output_tensor = prep.finish(output.output);

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
        }
    }

    fn max_pool2d_with_indices_backward(
        _x: AutodiffTensor<B>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _output_grad: AutodiffTensor<B>,
        _indices: IntTensor<B>,
    ) -> MaxPool2dBackward<Self> {
        panic!("Can't differentiate max pool2d with indices backward.");
    }
    fn adaptive_avg_pool1d(x: AutodiffTensor<B>, output_size: usize) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AdaptiveAvgPool1D;

        impl<B: Backend> Backward<B, 1> for AdaptiveAvgPool1D {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let state = checkpointer.retrieve_node_output(ops.state);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool1d_backward(state, grad);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AdaptiveAvgPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(x_state, B::adaptive_avg_pool1d(x.primitive, output_size))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool1d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d(x: AutodiffTensor<B>, output_size: [usize; 2]) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AdaptiveAvgPool2D;

        impl<B: Backend> Backward<B, 1> for AdaptiveAvgPool2D {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let state = checkpointer.retrieve_node_output(ops.state);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool2d_backward(state, grad);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AdaptiveAvgPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(x_state, B::adaptive_avg_pool2d(x.primitive, output_size))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool2d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d_backward(
        _x: AutodiffTensor<B>,
        _grad: AutodiffTensor<B>,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive {
        panic!("Can't differentiate adaptive avg pool2d backward.");
    }

    fn interpolate(
        x: AutodiffTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Interpolate;
        impl<B: Backend> Backward<B, 1> for Interpolate {
            type State = (NodeID, [usize; 2], InterpolateOptions);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, output_size, options) = ops.state;
                let state = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::interpolate_backward(state, grad, output_size, options);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match Interpolate
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output = B::interpolate(x.primitive.clone(), output_size, options.clone());
                prep.finish((x_state, output_size, options), output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::interpolate(x.primitive, output_size, options))
            }
        }
    }

    fn interpolate_backward(
        _x: FloatTensor<Autodiff<B, C>>,
        _grad: FloatTensor<Autodiff<B, C>>,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive {
        panic!("Can't differentiate interpolate backward.");
    }
}

#[derive(Debug)]
struct MaxPool1D;

impl<B: Backend> Backward<B, 1> for MaxPool1D {
    type State = (NodeID, IntTensor<B>, usize, usize, usize, usize);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (x_state, indices, kernel_size, stride, padding, dilation) = ops.state;
        let x = checkpointer.retrieve_node_output(x_state);

        if let Some(node) = node_parent {
            let grad = B::max_pool1d_with_indices_backward(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                grad,
                indices,
            );

            grads.register::<B>(node.id, grad.x_grad);
        }
    }
}

#[derive(Debug)]
struct MaxPool2D;

impl<B: Backend> Backward<B, 1> for MaxPool2D {
    type State = (
        NodeID,
        IntTensor<B>,
        [usize; 2],
        [usize; 2],
        [usize; 2],
        [usize; 2],
    );

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (x_state, indices, kernel_size, stride, padding, dilation) = ops.state;
        let x = checkpointer.retrieve_node_output(x_state);

        if let Some(node) = node_parent {
            let grad = B::max_pool2d_with_indices_backward(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                grad,
                indices,
            );

            grads.register::<B>(node.id, grad.x_grad);
        }
    }
}
