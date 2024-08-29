use std::marker::PhantomData;

use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Element, ElementConversion,
};
use cubecl::tune::{local_tuner, AutotuneOperation, AutotuneOperationSet, LocalTuner};

use crate::{
    element::{FloatElement, IntElement},
    kernel::prng::random_like_uniform,
    tensor::JitTensor,
    tune_key::JitAutotuneKey,
    JitRuntime, JitTuneId,
};

use super::key::Conv2dAutotuneKey;

/// Set of conv2d implementations available for autotune
pub struct Conv2dOperationsSet<R: JitRuntime, E: FloatElement, I: IntElement> {
    key: JitAutotuneKey,
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
    _int_element: PhantomData<I>,
}
impl<R: JitRuntime, E: FloatElement, I: IntElement> Conv2dOperationsSet<R, E, I> {
    fn new(
        input: JitTensor<R, E, 4>,
        weights: JitTensor<R, E, 4>,
        bias: Option<JitTensor<R, E, 1>>,
        options: ConvOptions<2>,
    ) -> Self {
        Self {
            key: JitAutotuneKey::Conv2d(Conv2dAutotuneKey::new(
                &input.shape,
                &weights.shape,
                &options,
            )),
            input,
            weights,
            bias,
            options,
            _int_element: PhantomData,
        }
    }
}

/// Set of conv_transpose2d implementations available for autotune
pub struct ConvTranspose2dOperationsSet<R: JitRuntime, E: FloatElement, I: IntElement> {
    key: JitAutotuneKey,
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvTransposeOptions<2>,
    _int_element: PhantomData<I>,
}
impl<R: JitRuntime, E: FloatElement, I: IntElement> ConvTranspose2dOperationsSet<R, E, I> {
    fn new(
        input: JitTensor<R, E, 4>,
        weights: JitTensor<R, E, 4>,
        bias: Option<JitTensor<R, E, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> Self {
        Self {
            key: JitAutotuneKey::Conv2d(Conv2dAutotuneKey::new(
                &input.shape,
                &weights.shape,
                &ConvOptions {
                    stride: options.stride,
                    padding: options.padding,
                    dilation: options.dilation,
                    groups: options.groups,
                },
            )),
            input,
            weights,
            bias,
            options,
            _int_element: PhantomData,
        }
    }
}
impl<R: JitRuntime, E: FloatElement, I: IntElement>
    AutotuneOperationSet<JitAutotuneKey, JitTensor<R, E, 4>> for Conv2dOperationsSet<R, E, I>
{
    fn key(&self) -> JitAutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<JitTensor<R, E, 4>>>> {
        let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
        let input = random_like_uniform(&self.input, random_bounds.0, random_bounds.1);
        let output = random_like_uniform(&self.weights, random_bounds.0, random_bounds.1);
        let bias = self
            .bias
            .as_ref()
            .map(|bias| random_like_uniform(bias, random_bounds.0, random_bounds.1));

        vec![
            Box::new(DirectConv2d::<R, E, I>::new(
                input.clone(),
                output.clone(),
                bias.clone(),
                self.options.clone(),
            )),
            Box::new(GemmConv2d::<R, E, I>::new(
                input.clone(),
                output.clone(),
                bias.clone(),
                self.options.clone(),
            )),
        ]
    }

    fn fastest(
        self: Box<Self>,
        fastest_index: usize,
    ) -> Box<dyn AutotuneOperation<JitTensor<R, E, 4>>> {
        match fastest_index {
            0 => Box::new(DirectConv2d::<R, E, I>::new(
                self.input,
                self.weights,
                self.bias,
                self.options,
            )),
            1 => Box::new(GemmConv2d::<R, E, I>::new(
                self.input,
                self.weights,
                self.bias,
                self.options,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }

    fn compute_checksum(&self) -> String {
        cubecl::tune::compute_checksum(&self.autotunables())
    }
}

impl<R: JitRuntime, E: FloatElement, I: IntElement>
    AutotuneOperationSet<JitAutotuneKey, JitTensor<R, E, 4>>
    for ConvTranspose2dOperationsSet<R, E, I>
{
    fn key(&self) -> JitAutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<JitTensor<R, E, 4>>>> {
        let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
        let input = random_like_uniform(&self.input, random_bounds.0, random_bounds.1);
        let output = random_like_uniform(&self.weights, random_bounds.0, random_bounds.1);
        let bias = self
            .bias
            .as_ref()
            .map(|bias| random_like_uniform(bias, random_bounds.0, random_bounds.1));

        vec![
            Box::new(DirectConvTranspose2d::<R, E, I>::new(
                input.clone(),
                output.clone(),
                bias.clone(),
                self.options.clone(),
            )),
            Box::new(GemmConvTranspose2d::<R, E, I>::new(
                input.clone(),
                output.clone(),
                bias.clone(),
                self.options.clone(),
            )),
        ]
    }

    fn fastest(
        self: Box<Self>,
        fastest_index: usize,
    ) -> Box<dyn AutotuneOperation<JitTensor<R, E, 4>>> {
        match fastest_index {
            0 => Box::new(DirectConvTranspose2d::<R, E, I>::new(
                self.input,
                self.weights,
                self.bias,
                self.options,
            )),
            1 => Box::new(GemmConvTranspose2d::<R, E, I>::new(
                self.input,
                self.weights,
                self.bias,
                self.options,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }

    fn compute_checksum(&self) -> String {
        cubecl::tune::compute_checksum(&self.autotunables())
    }
}

/// Executes autotune on conv2d operations
pub fn conv2d_autotune<R: JitRuntime, E: FloatElement + Element, I: IntElement>(
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let client = input.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!("conv2d");

    TUNER.execute(
        &JitTuneId::new::<R>(&input.device),
        &client,
        Box::new(Conv2dOperationsSet::<R, E, I>::new(
            input, weights, bias, options,
        )),
    )
}

/// Executes autotune on conv2d operations
pub fn conv_transpose2d_autotune<R: JitRuntime, E: FloatElement + Element, I: IntElement>(
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E, 4> {
    let client = input.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!("conv-transpose2d");

    TUNER.execute(
        &JitTuneId::new::<R>(&input.device),
        &client,
        Box::new(ConvTranspose2dOperationsSet::<R, E, I>::new(
            input, weights, bias, options,
        )),
    )
}

macro_rules! conv2d_tune_ops {
    ($name:ident, $func:expr, $options:ident) => {
        #[derive(new)]
        pub(crate) struct $name<R: JitRuntime, E: FloatElement, I: IntElement> {
            input: JitTensor<R, E, 4>,
            weights: JitTensor<R, E, 4>,
            bias: Option<JitTensor<R, E, 1>>,
            options: $options<2>,
            _int_element: PhantomData<I>,
        }

        impl<R: JitRuntime, E: FloatElement, I: IntElement> AutotuneOperation<JitTensor<R, E, 4>>
            for $name<R, E, I>
        {
            fn execute(self: Box<Self>) -> JitTensor<R, E, 4> {
                #[allow(clippy::redundant_closure_call)]
                $func(self.input, self.weights, self.bias, self.options)
            }

            fn clone(&self) -> Box<dyn AutotuneOperation<JitTensor<R, E, 4>>> {
                Box::new(Self {
                    input: self.input.clone(),
                    weights: self.weights.clone(),
                    bias: self.bias.clone(),
                    options: self.options.clone(),
                    _int_element: PhantomData,
                })
            }
        }
    };
}

// Lower memory footprint
conv2d_tune_ops!(
    DirectConv2d,
    crate::kernel::conv::conv2d::conv2d_direct,
    ConvOptions
);

// Faster if memory is sufficient
conv2d_tune_ops!(
    GemmConv2d,
    crate::kernel::conv::conv2d::conv2d_im2col::<R, E, I>,
    ConvOptions
);

conv2d_tune_ops!(
    DirectConvTranspose2d,
    crate::kernel::conv::conv2d::conv_transpose2d_direct,
    ConvTransposeOptions
);

conv2d_tune_ops!(
    GemmConvTranspose2d,
    crate::kernel::conv::conv2d::conv_transpose2d_col2im::<R, E, I>,
    ConvTransposeOptions
);
