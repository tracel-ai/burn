use crate::grads::Gradients;
use crate::ops::{Backward, Ops};
use crate::tensor::{ADTensor, IntTensor};
use crate::ADBackendDecorator;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(weights: ADTensor<B, 2>, indexes: IntTensor<B, 2>) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct Embedding;

        impl<B: Backend> Backward<B, 3, 1> for Embedding {
            type State = (B::TensorPrimitive<2>, IntTensor<B, 2>);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);
                let (weights, indexes) = ops.state;

                node_parent.run(|node, _| {
                    let grad = B::embedding_backward(weights, grad, indexes);
                    grads.register::<B, 2>(node, grad);
                });
            }
        }

        Embedding.run(
            (weights.primitive.clone(), indexes.clone()),
            B::embedding(weights.primitive, indexes),
            [weights.node],
            [weights.graph],
        )
    }

    fn embedding_backward(
        weights: ADTensor<B, 2>,
        output: ADTensor<B, 3>,
        indexes: IntTensor<B, 2>,
    ) -> ADTensor<B, 2> {
        let tensor = B::embedding_backward(weights.primitive, output.primitive, indexes);
        ADTensor::new(tensor)
    }

    fn conv2d(
        x: ADTensor<B, 4>,
        weight: ADTensor<B, 4>,
        bias: Option<ADTensor<B, 1>>,
        stride: [usize; 2],
        padding: [usize; 2],
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
                [usize; 2],
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, bias, stride) = ops.state;
                let backward = B::conv2d_backward(x, weight, Some(bias), stride, grad);

                node_x.run(|node, _| grads.register::<B, 4>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 4>(node, backward.weights_grad));
                node_bias.run(|node, _| grads.register::<B, 1>(node, backward.bias_grad.unwrap()));
            }
        }

        impl<B: Backend> Backward<B, 4, 2> for Conv2DNoBias {
            type State = (B::TensorPrimitive<4>, B::TensorPrimitive<4>, [usize; 2]);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, stride) = ops.state;
                let backward = B::conv2d_backward(x, weight, None, stride, grad);

                node_x.run(|node, _| grads.register::<B, 4>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 4>(node, backward.weights_grad));
            }
        }

        match bias {
            Some(bias) => Conv2DWithBias.run(
                (
                    x.primitive.clone(),
                    weight.primitive.clone(),
                    bias.primitive.clone(),
                    stride,
                ),
                B::conv2d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    stride,
                    padding,
                ),
                [x.node, weight.node, bias.node],
                [x.graph, weight.graph, bias.graph],
            ),
            None => Conv2DNoBias.run(
                (x.primitive.clone(), weight.primitive.clone(), stride),
                B::conv2d(x.primitive, weight.primitive, None, stride, padding),
                [x.node, weight.node],
                [x.graph, weight.graph],
            ),
        }
    }
    fn conv1d(
        x: ADTensor<B, 3>,
        weight: ADTensor<B, 3>,
        bias: Option<ADTensor<B, 1>>,
        stride: usize,
        padding: usize,
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
                usize,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, bias, stride) = ops.state;
                let backward = B::conv1d_backward(x, weight, Some(bias), stride, grad);

                node_x.run(|node, _| grads.register::<B, 3>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 3>(node, backward.weights_grad));
                node_bias.run(|node, _| grads.register::<B, 1>(node, backward.bias_grad.unwrap()));
            }
        }

        impl<B: Backend> Backward<B, 3, 2> for Conv1DNoBias {
            type State = (B::TensorPrimitive<3>, B::TensorPrimitive<3>, usize);

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, stride) = ops.state;
                let backward = B::conv1d_backward(x, weight, None, stride, grad);

                node_x.run(|node, _| grads.register::<B, 3>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 3>(node, backward.weights_grad));
            }
        }

        match bias {
            Some(bias) => Conv1DWithBias.run(
                {
                    (
                        x.primitive.clone(),
                        weight.primitive.clone(),
                        bias.primitive.clone(),
                        stride,
                    )
                },
                B::conv1d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    stride,
                    padding,
                ),
                [x.node, weight.node, bias.node],
                [x.graph, weight.graph, bias.graph],
            ),
            None => Conv1DNoBias.run(
                (x.primitive.clone(), weight.primitive.clone(), stride),
                B::conv1d(x.primitive, weight.primitive, None, stride, padding),
                [x.node, weight.node],
                [x.graph, weight.graph],
            ),
        }
    }

    fn max_pool2d(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        if !x.is_tracked() {
            return ADTensor::new(B::max_pool2d(x.primitive, kernel_size, stride, padding));
        }

        let output = B::max_pool2d_with_indexes(x.primitive.clone(), kernel_size, stride, padding);
        let state = (x.primitive, output.indexes, kernel_size, stride, padding);

        MaxPool2D.run(state, output.output, [x.node], [x.graph])
    }

    fn max_pool2d_with_indexes(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> MaxPool2dWithIndexes<ADBackendDecorator<B>> {
        if !x.is_tracked() {
            let output = B::max_pool2d_with_indexes(x.primitive, kernel_size, stride, padding);
            return MaxPool2dWithIndexes::new(ADTensor::new(output.output), output.indexes);
        }

        let output = B::max_pool2d_with_indexes(x.primitive.clone(), kernel_size, stride, padding);
        let indexes = output.indexes;
        let state = (x.primitive, indexes.clone(), kernel_size, stride, padding);

        let output = MaxPool2D.run(state, output.output, [x.node], [x.graph]);
        MaxPool2dWithIndexes::new(output, indexes)
    }

    fn max_pool2d_with_indexes_backward(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_grad: ADTensor<B, 4>,
        indexes: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<ADBackendDecorator<B>> {
        let output = B::max_pool2d_with_indexes_backward(
            x.primitive,
            kernel_size,
            stride,
            padding,
            output_grad.primitive,
            indexes,
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
        let (x, indexes, kernel_size, stride, padding) = ops.state;

        node_parent.run(|node, _| {
            let grad =
                B::max_pool2d_with_indexes_backward(x, kernel_size, stride, padding, grad, indexes);
            grads.register::<B, 4>(node, grad.x_grad);
        });
    }
}
