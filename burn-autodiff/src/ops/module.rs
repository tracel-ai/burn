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
        _weights: ADTensor<B, 2>,
        _output: ADTensor<B, 3>,
        _indices: IntTensor<B, 2>,
    ) -> ADTensor<B, 2> {
        panic!("Can't differentiate embedding backward.");
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
        x: ADTensor<B, 4>,
        weight: ADTensor<B, 4>,
        bias: Option<ADTensor<B, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> ADTensor<B, 4> {
        #[derive(Debug)]
        struct ConvTranspose2DWithBias;
        #[derive(Debug)]
        struct ConvTranspose2DNoBias;

        impl<B: Backend> Backward<B, 4, 3> for ConvTranspose2DWithBias {
            type State = (
                B::TensorPrimitive<4>,
                B::TensorPrimitive<4>,
                B::TensorPrimitive<1>,
                ConvTransposeOptions<2>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, bias, options) = ops.state;
                let backward = B::conv_transpose2d_backward(x, weight, Some(bias), grad, options);

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

        impl<B: Backend> Backward<B, 4, 2> for ConvTranspose2DNoBias {
            type State = (
                B::TensorPrimitive<4>,
                B::TensorPrimitive<4>,
                ConvTransposeOptions<2>,
            );

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                let (x, weight, options) = ops.state;
                let backward = B::conv_transpose2d_backward(x, weight, None, grad, options);

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
                match ConvTranspose2DWithBias
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
                        B::conv_transpose2d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                        x.primitive,
                        weight.primitive,
                        Some(bias.primitive),
                        options,
                    )),
                }
            }
            None => {
                match ConvTranspose2DNoBias
                    .prepare([x.node, weight.node], [x.graph, weight.graph])
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv_transpose2d(x.primitive, weight.primitive, None, options),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                        x.primitive,
                        weight.primitive,
                        None,
                        options,
                    )),
                }
            }
        }
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
        x: ADTensor<B, 3>,
        weight: ADTensor<B, 3>,
        bias: Option<ADTensor<B, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct ConvTranspose1DWithBias;
        #[derive(Debug)]
        struct ConvTranspose1DNoBias;

        impl<B: Backend> Backward<B, 3, 3> for ConvTranspose1DWithBias {
            type State = (
                B::TensorPrimitive<3>,
                B::TensorPrimitive<3>,
                B::TensorPrimitive<1>,
                ConvTransposeOptions<1>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, bias, options) = ops.state;
                let backward = B::conv_transpose1d_backward(x, weight, Some(bias), grad, options);

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

        impl<B: Backend> Backward<B, 3, 2> for ConvTranspose1DNoBias {
            type State = (
                B::TensorPrimitive<3>,
                B::TensorPrimitive<3>,
                ConvTransposeOptions<1>,
            );

            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                let (x, weight, options) = ops.state;
                let backward = B::conv_transpose1d_backward(x, weight, None, grad, options);

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
                match ConvTranspose1DWithBias
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
                        B::conv_transpose1d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                        x.primitive,
                        weight.primitive,
                        Some(bias.primitive),
                        options,
                    )),
                }
            }
            None => {
                match ConvTranspose1DNoBias
                    .prepare([x.node, weight.node], [x.graph, weight.graph])
                    .statefull()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (
                            x.primitive.clone(),
                            weight.primitive.clone(),
                            options.clone(),
                        ),
                        B::conv_transpose1d(x.primitive, weight.primitive, None, options),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                        x.primitive,
                        weight.primitive,
                        None,
                        options,
                    )),
                }
            }
        }
    }

    fn avg_pool1d(
        x: ADTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct AvgPool1D;

        impl<B: Backend> Backward<B, 3, 1> for AvgPool1D {
            type State = (B::TensorPrimitive<3>, usize, usize, usize, bool);

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);
                let (x, kernel_size, stride, padding, count_include_pad) = ops.state;

                if let Some(node) = node_parent {
                    let grad = B::avg_pool1d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    );
                    grads.register::<B, 3>(node, grad);
                }
            }
        }

        match AvgPool1D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::avg_pool1d(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                );
                prep.finish(
                    (x.primitive, kernel_size, stride, padding, count_include_pad),
                    output,
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
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> ADTensor<B, 4> {
        #[derive(Debug)]
        struct AvgPool2D;

        impl<B: Backend> Backward<B, 4, 1> for AvgPool2D {
            type State = (
                B::TensorPrimitive<4>,
                [usize; 2],
                [usize; 2],
                [usize; 2],
                bool,
            );

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);
                let (x, kernel_size, stride, padding, count_include_pad) = ops.state;

                if let Some(node) = node_parent {
                    let grad = B::avg_pool2d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                    );
                    grads.register::<B, 4>(node, grad);
                }
            }
        }

        match AvgPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::avg_pool2d(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                );
                prep.finish(
                    (x.primitive, kernel_size, stride, padding, count_include_pad),
                    output,
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
        _x: ADTensor<B, 4>,
        _grad: ADTensor<B, 4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
    ) -> ADTensor<B, 4> {
        panic!("Can't differentiate avg pool 2d backward.");
    }

    fn max_pool1d(
        x: ADTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> ADTensor<B, 3> {
        match MaxPool1D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::max_pool1d_with_indices(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                );
                prep.finish(
                    (
                        x.primitive,
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
        x: ADTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<ADBackendDecorator<B>> {
        match MaxPool1D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::max_pool1d_with_indices(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                );

                let output_tensor = prep.finish(
                    (
                        x.primitive,
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
        x: ADTensor<B, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: ADTensor<B, 3>,
        indices: IntTensor<B, 3>,
    ) -> MaxPool1dBackward<ADBackendDecorator<B>> {
        let output = B::max_pool1d_with_indices_backward(
            x.primitive,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad.primitive,
            indices,
        );
        MaxPool1dBackward::new(ADTensor::new(output.x_grad))
    }

    fn max_pool2d(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> ADTensor<B, 4> {
        match MaxPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::max_pool2d_with_indices(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                );
                prep.finish(
                    (
                        x.primitive,
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
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<ADBackendDecorator<B>> {
        match MaxPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::max_pool2d_with_indices(
                    x.primitive.clone(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                );

                let output_tensor = prep.finish(
                    (
                        x.primitive,
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
        _x: ADTensor<B, 4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _output_grad: ADTensor<B, 4>,
        _indices: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<ADBackendDecorator<B>> {
        panic!("Can't differentiate max pool2d with indices backward.");
    }
    fn adaptive_avg_pool1d(x: ADTensor<B, 3>, output_size: usize) -> ADTensor<B, 3> {
        #[derive(Debug)]
        struct AdaptiveAvgPool1D;

        impl<B: Backend> Backward<B, 3, 1> for AdaptiveAvgPool1D {
            type State = B::TensorPrimitive<3>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 3>(&ops.node);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool1d_backward(ops.state, grad);
                    grads.register::<B, 3>(node, grad);
                }
            }
        }

        match AdaptiveAvgPool1D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                x.primitive.clone(),
                B::adaptive_avg_pool1d(x.primitive, output_size),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool1d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d(x: ADTensor<B, 4>, output_size: [usize; 2]) -> ADTensor<B, 4> {
        #[derive(Debug)]
        struct AdaptiveAvgPool2D;

        impl<B: Backend> Backward<B, 4, 1> for AdaptiveAvgPool2D {
            type State = B::TensorPrimitive<4>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B, 4>(&ops.node);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool2d_backward(ops.state, grad);
                    grads.register::<B, 4>(node, grad);
                }
            }
        }

        match AdaptiveAvgPool2D.prepare([x.node], [x.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(
                x.primitive.clone(),
                B::adaptive_avg_pool2d(x.primitive, output_size),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool2d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d_backward(
        _x: ADTensor<B, 4>,
        _grad: ADTensor<B, 4>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<4> {
        panic!("Can't differentiate adaptive avg pool2d backward.");
    }
}

#[derive(Debug)]
struct MaxPool1D;

impl<B: Backend> Backward<B, 3, 1> for MaxPool1D {
    type State = (
        B::TensorPrimitive<3>,
        IntTensor<B, 3>,
        usize,
        usize,
        usize,
        usize,
    );

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B, 3>(&ops.node);
        let (x, indices, kernel_size, stride, padding, dilation) = ops.state;

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

            grads.register::<B, 3>(node, grad.x_grad);
        }
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
        [usize; 2],
    );

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B, 4>(&ops.node);
        let (x, indices, kernel_size, stride, padding, dilation) = ops.state;

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

            grads.register::<B, 4>(node, grad.x_grad);
        }
    }
}
