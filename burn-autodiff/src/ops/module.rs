use crate::grads::Gradients;
use crate::ops::{Backward, OpsNodes};
use crate::tensor::{ADTensor, BackwardTensor, IntTensor};
use crate::ADBackendDecorator;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(weights: ADTensor<B, 2>, indexes: IntTensor<B, 2>) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct Embedding;

        impl<B: Backend> Backward<B, 3, 1> for Embedding {
            type State = Option<(B::TensorPrimitive<2>, IntTensor<B, 2>)>;

            fn backward(
                self,
                [node]: OpsNodes<1>,
                output: BackwardTensor<B, 3>,
                grads: &mut Gradients,
                state: Self::State,
            ) {
                let grad = grads.consume(&output);

                node.requirements([state])
                    .run(|node, [(weights, indexes)]| {
                        let grad = B::embedding_backward(weights, grad, indexes.clone());
                        grads.register::<B, 2>(node, grad);
                    });
            }
        }

        Embedding.run(
            weights
                .is_tracked()
                .then(|| (weights.primitive.clone(), indexes.clone())),
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
            type State = Option<(
                B::TensorPrimitive<4>,
                B::TensorPrimitive<4>,
                B::TensorPrimitive<1>,
                [usize; 2],
            )>;

            fn backward(
                self,
                [node_x, node_weight, node_bias]: OpsNodes<3>,
                output: BackwardTensor<B, 4>,
                grads: &mut Gradients,
                state: Self::State,
            ) {
                let grad = grads.consume(&output);

                let (x, weight, bias, stride) = state.unwrap();
                let backward = B::conv2d_backward(x, weight, Some(bias), stride, grad);

                node_x.run(|node, _| grads.register::<B, 4>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 4>(node, backward.weights_grad));
                node_bias.run(|node, _| grads.register::<B, 1>(node, backward.bias_grad.unwrap()));
            }
        }

        impl<B: Backend> Backward<B, 4, 2> for Conv2DNoBias {
            type State = Option<(B::TensorPrimitive<4>, B::TensorPrimitive<4>, [usize; 2])>;

            fn backward(
                self,
                [node_x, node_weight]: OpsNodes<2>,
                output: BackwardTensor<B, 4>,
                grads: &mut Gradients,
                state: Self::State,
            ) {
                let grad = grads.consume(&output);

                let (x, weight, stride) = state.unwrap();
                let backward = B::conv2d_backward(x, weight, None, stride, grad);

                node_x.run(|node, _| grads.register::<B, 4>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 4>(node, backward.weights_grad));
            }
        }

        match bias {
            Some(bias) => Conv2DWithBias.run(
                weight.is_tracked().then(|| {
                    (
                        x.primitive.clone(),
                        weight.primitive.clone(),
                        bias.primitive.clone(),
                        stride,
                    )
                }),
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
                weight
                    .is_tracked()
                    .then(|| (x.primitive.clone(), weight.primitive.clone(), stride)),
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
            type State = Option<(
                B::TensorPrimitive<3>,
                B::TensorPrimitive<3>,
                B::TensorPrimitive<1>,
                usize,
            )>;

            fn backward(
                self,
                [node_x, node_weight, node_bias]: OpsNodes<3>,
                output: BackwardTensor<B, 3>,
                grads: &mut Gradients,
                state: Self::State,
            ) {
                let grad = grads.consume(&output);

                let (x, weight, bias, stride) = state.unwrap();
                let backward = B::conv1d_backward(x, weight, Some(bias), stride, grad);

                node_x.run(|node, _| grads.register::<B, 3>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 3>(node, backward.weights_grad));
                node_bias.run(|node, _| grads.register::<B, 1>(node, backward.bias_grad.unwrap()));
            }
        }

        impl<B: Backend> Backward<B, 3, 2> for Conv1DNoBias {
            type State = Option<(B::TensorPrimitive<3>, B::TensorPrimitive<3>, usize)>;

            fn backward(
                self,
                [node_x, node_weight]: OpsNodes<2>,
                output: BackwardTensor<B, 3>,
                grads: &mut Gradients,
                state: Self::State,
            ) {
                let grad = grads.consume(&output);

                let (x, weight, stride) = state.unwrap();
                let backward = B::conv1d_backward(x, weight, None, stride, grad);

                node_x.run(|node, _| grads.register::<B, 3>(node, backward.x_grad));
                node_weight.run(|node, _| grads.register::<B, 3>(node, backward.weights_grad));
            }
        }

        match bias {
            Some(bias) => Conv1DWithBias.run(
                weight.is_tracked().then(|| {
                    (
                        x.primitive.clone(),
                        weight.primitive.clone(),
                        bias.primitive.clone(),
                        stride,
                    )
                }),
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
                weight
                    .is_tracked()
                    .then(|| (x.primitive.clone(), weight.primitive.clone(), stride)),
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
        return MaxPool2dWithIndexes::new(output, indexes);
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

    fn backward(
        self,
        [node]: OpsNodes<1>,
        output: BackwardTensor<B, 4>,
        grads: &mut Gradients,
        (x, indexes, kernel_size, stride, padding): Self::State,
    ) {
        let grad = grads.consume(&output);

        node.run(|node, _| {
            let grad =
                B::max_pool2d_with_indexes_backward(x, kernel_size, stride, padding, grad, indexes);
            grads.register::<B, 4>(node, grad.x_grad);
        });
    }
}
