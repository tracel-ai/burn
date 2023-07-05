use crate::grads::Gradients;
use crate::ops::{unary, Backward, Ops};
use crate::tensor::{ADTensor, IntTensor};
use crate::ADBackendDecorator;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

use super::OpsKind;

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(weights: ADTensor<B, 2>, indices: IntTensor<B, 2>) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct Embedding;

        impl<B: Backend> Backward<B, 3, 1> for Embedding {
            type State = (B::TensorPrimitive<2>, IntTensor<B, 2>);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (weights, indices) = ops.state;

                unary::<B, 3, 2, _>(ops.parents, ops.node, grads, |grad| {
                    B::embedding_backward(weights, grad, indices)
                });
            }
        }

        match Embedding
            .prepare([weights.node], [weights.graph])
            .statefull()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (weights.primitive.clone(), indices.clone()),
                B::embedding(weights.primitive, indices),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::embedding(weights.primitive, indices)),
        }
    }

    fn embedding_backward(
        weights: ADTensor<B, 2>,
        output: ADTensor<B, 3>,
        indices: IntTensor<B, 2>,
    ) -> ADTensor<B, 2> {
        let tensor = B::embedding_backward(weights.primitive, output.primitive, indices);
        ADTensor::new(tensor)
    }

    fn conv2d(
        x: ADTensor<B, 4>,
        weight: ADTensor<B, 4>,
        bias: Option<ADTensor<B, 1>>,
        options: ConvOptions<2>,
    ) -> ADTensor<B, 4> {
        #[derive(Debug)]
        struct Conv2DWithBias;
        #[derive(Debug)]
        struct Conv2DNoBias;

        impl<B: Backend> Backward<B, 4, 3> for Conv2DWithBias {
            type State = (
                B::TensorPrimitive<4>,
                B::TensorPrimitive<4>,
                B::TensorPrimitive<1>,
                ConvOptions<2>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, bias, options) = ops.state;
                let backward = B::conv2d_backward(x, weight, Some(bias), grad, options);

                if let Some(node) = node_x {
                    grads.register::<B, 4>(node, backward.x_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B, 4>(node, backward.weights_grad)
                }
                if let Some(node) = node_bias {
                    grads.register::<B, 1>(node, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 4, 2> for Conv2DNoBias {
            type State = (B::TensorPrimitive<4>, B::TensorPrimitive<4>, ConvOptions<2>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, options) = ops.state;
                let backward = B::conv2d_backward(x, weight, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B, 4>(node, backward.x_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B, 4>(node, backward.weights_grad)
                }
            }
        }

        match bias {
            Some(bias) => {
                match Conv2DWithBias
                    .prepare(
                        [x.node, weight.node, bias.node],
                        [x.graph, weight.graph, bias.graph],
                    )
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            bias.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv2d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv2d(
                        x.primitive,
                        weight.primitive,
                        Some(bias.primitive),
                        options,
                    )),
                }
            }
            None => {
                match Conv2DNoBias
                    .prepare([x.node, weight.node], [x.graph, weight.graph])
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv2d(x.primitive, weight.primitive, None, options),
                    ),
                    OpsKind::UnTracked(prep) => {
                        prep.finish(B::conv2d(x.primitive, weight.primitive, None, options))
                    }
                }
            }
        }
    }

    fn conv_transpose2d(
        _x: ADTensor<B, 4>,
        _weight: ADTensor<B, 4>,
        _bias: Option<ADTensor<B, 1>>,
        _options: ConvTransposeOptions<2>,
    ) -> ADTensor<B, 4> {
        todo!("Transposed 2D convolution doesn't yet support backward.");
    }

    fn conv1d(
        x: ADTensor<B, 3>,
        weight: ADTensor<B, 3>,
        bias: Option<ADTensor<B, 1>>,
        options: ConvOptions<1>,
    ) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct Conv1DWithBias;
        #[derive(Debug)]
        struct Conv1DNoBias;

        impl<B: Backend> Backward<B, 3, 3> for Conv1DWithBias {
            type State = (
                B::TensorPrimitive<3>,
                B::TensorPrimitive<3>,
                B::TensorPrimitive<1>,
                ConvOptions<1>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, bias, options) = ops.state;
                let backward = B::conv1d_backward(x, weight, Some(bias), grad, options);

                if let Some(node) = node_x {
                    grads.register::<B, 3>(node, backward.x_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B, 3>(node, backward.weights_grad)
                }
                if let Some(node) = node_bias {
                    grads.register::<B, 1>(node, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 3, 2> for Conv1DNoBias {
            type State = (B::TensorPrimitive<3>, B::TensorPrimitive<3>, ConvOptions<1>);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, options) = ops.state;
                let backward = B::conv1d_backward(x, weight, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B, 3>(node, backward.x_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B, 3>(node, backward.weights_grad)
                }
            }
        }
        match bias {
            Some(bias) => {
                match Conv1DWithBias
                    .prepare(
                        [x.node, weight.node, bias.node],
                        [x.graph, weight.graph, bias.graph],
                    )
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            bias.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv1d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv1d(
                        x.primitive,
                        weight.primitive,
                        Some(bias.primitive),
                        options,
                    )),
                }
            }
            None => {
                match Conv1DNoBias
                    .prepare([x.node, weight.node], [x.graph, weight.graph])
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv1d(x.primitive, weight.primitive, None, options),
                    ),
                    OpsKind::UnTracked(prep) => {
                        prep.finish(B::conv1d(x.primitive, weight.primitive, None, options))
                    }
                }
            }
        }
    }

    fn conv_transpose1d(
        _x: ADTensor<B, 3>,
        _weight: ADTensor<B, 3>,
        _bias: Option<ADTensor<B, 1>>,
        _options: ConvTransposeOptions<1>,
    ) -> ADTensor<B, 3> {
        todo!("Transposed 1D convolution doesn't yet support backward.");
    }
    fn avg_pool1d(
        x: ADTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct AvgPool1D;

        impl<B: Backend> Backward<B, 3, 1> for AvgPool1D {
            type State = (B::TensorPrimitive<3>, usize, usize, usize);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);
                let (x, kernel_size, stride, padding) = ops.state;

                if let Some(node) = node_parent {
                    let grad = B::avg_pool1d_backward(x, grad, kernel_size, stride, padding);
                    grads.register::<B, 3>(node, grad);
                }
            }
        }

        match AvgPool1D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::avg_pool1d(x.primitive.clone(), kernel_size, stride, padding);
                prep.finish((x.primitive, kernel_size, stride, padding), output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::avg_pool1d(x.primitive, kernel_size, stride, padding))
            }
        }
    }

    fn avg_pool2d(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        #[derive(Debug)]
        struct AvgPool2D;

        impl<B: Backend> Backward<B, 4, 1> for AvgPool2D {
            type State = (B::TensorPrimitive<4>, [usize; 2], [usize; 2], [usize; 2]);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);
                let (x, kernel_size, stride, padding) = ops.state;

                if let Some(node) = node_parent {
                    let grad = B::avg_pool2d_backward(x, grad, kernel_size, stride, padding);
                    grads.register::<B, 4>(node, grad);
                }
            }
        }

        match AvgPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::avg_pool2d(x.primitive.clone(), kernel_size, stride, padding);
                prep.finish((x.primitive, kernel_size, stride, padding), output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::avg_pool2d(x.primitive, kernel_size, stride, padding))
            }
        }
    }
    fn avg_pool2d_backward(
        x: ADTensor<B, 4>,
        grad: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        let tensor =
            B::avg_pool2d_backward(x.primitive, grad.primitive, kernel_size, stride, padding);
        ADTensor::new(tensor)
    }

    fn max_pool2d(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        match MaxPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output =
                    B::max_pool2d_with_indices(x.primitive.clone(), kernel_size, stride, padding);
                prep.finish(
                    (x.primitive, output.indices, kernel_size, stride, padding),
                    output.output,
                )
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::max_pool2d(x.primitive, kernel_size, stride, padding))
            }
        }
    }

    fn max_pool2d_with_indices(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> MaxPool2dWithIndices<ADBackendDecorator<B>> {
        match MaxPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output =
                    B::max_pool2d_with_indices(x.primitive.clone(), kernel_size, stride, padding);

                let output_tensor = prep.finish(
                    (
                        x.primitive,
                        output.indices.clone(),
                        kernel_size,
                        stride,
                        padding,
                    ),
                    output.output,
                );

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::max_pool2d_with_indices(x.primitive, kernel_size, stride, padding);
                let output_tensor = prep.finish(output.output);

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
        }
    }

    fn max_pool2d_with_indices_backward(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_grad: ADTensor<B, 4>,
        indices: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<ADBackendDecorator<B>> {
        let output = B::max_pool2d_with_indices_backward(
            x.primitive,
            kernel_size,
            stride,
            padding,
            output_grad.primitive,
            indices,
        );
        MaxPool2dBackward::new(ADTensor::new(output.x_grad))
    }
}

#[derive(Debug)]
struct MaxPool2D;

impl<B: Backend> Backward<B, 4, 1> for MaxPool2D {
    type State = (
        B::TensorPrimitive<4>,
        IntTensor<B, 4>,
        [usize; 2],
        [usize; 2],
        [usize; 2],
    );

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B, 4>(&ops.node);
        let (x, indices, kernel_size, stride, padding) = ops.state;

        if let Some(node) = node_parent {
            let grad =
                B::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, grad, indices);

            grads.register::<B, 4>(node, grad.x_grad);
        }
    }
}
