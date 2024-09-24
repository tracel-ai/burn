use std::marker::PhantomData;

use burn_tensor::{Element, ElementConversion};
use cubecl::tune::{local_tuner, AutotuneOperation, AutotuneOperationSet, LocalTuner};

use crate::{
    element::JitElement,
    kernel::{
        prng::random_like_uniform,
        reduce::{
            init_reduce_output, naive::shader::reduce_dim_naive, shared::shader::reduce_dim_shared,
            ReduceDimAlgorithm,
        },
    },
    ops::numeric::empty_device,
    tensor::JitTensor,
    tune_key::JitAutotuneKey,
    JitRuntime, JitTuneId,
};

use super::ReduceAutotuneKey;

/// Set of reduce_dim implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of
/// dim to reduce, and product of others
pub(crate) struct ReduceDimAutotuneOperationSet<
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
> {
    key: JitAutotuneKey,
    input: JitTensor<R, EI>,
    output: JitTensor<R, EO>,
    reduce_dim: usize,
    _algorithm: PhantomData<RD>,
}
impl<RD: ReduceDimAlgorithm<EI>, R: JitRuntime, EI: JitElement, EO: JitElement>
    ReduceDimAutotuneOperationSet<RD, R, EI, EO>
{
    fn new(input: JitTensor<R, EI>, output: JitTensor<R, EO>, reduce_dim: usize) -> Self {
        Self {
            key: JitAutotuneKey::ReduceDim(ReduceAutotuneKey::new(
                &input.shape,
                &input.strides,
                reduce_dim,
            )),
            input,
            output,
            reduce_dim,
            _algorithm: PhantomData,
        }
    }
}

impl<RD: ReduceDimAlgorithm<EI>, R, EI, EO> AutotuneOperationSet<JitAutotuneKey>
    for ReduceDimAutotuneOperationSet<RD, R, EI, EO>
where
    R: JitRuntime,
    EI: JitElement + Element,
    EO: JitElement + Element,
{
    fn key(&self) -> JitAutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        let random_bounds: (EI, EI) = ((-10.0).elem::<EI>(), (10.0).elem::<EI>());
        let input = random_like_uniform(&self.input, random_bounds.0, random_bounds.1);

        let output: JitTensor<R, EO> = empty_device(
            self.output.client.clone(),
            self.output.device.clone(),
            self.output.shape.clone(),
        );

        vec![
            Box::new(ReduceDimNaiveAutotune::<RD, R, EI, EO>::new(
                input.clone(),
                output.clone(),
                self.reduce_dim,
            )),
            Box::new(ReduceDimSharedAutotune::<RD, R, EI, EO>::new(
                input.clone(),
                output.clone(),
                self.reduce_dim,
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(ReduceDimNaiveAutotune::<RD, R, EI, EO>::new(
                self.input,
                self.output,
                self.reduce_dim,
            )),
            1 => Box::new(ReduceDimSharedAutotune::<RD, R, EI, EO>::new(
                self.input,
                self.output,
                self.reduce_dim,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

/// Executes autotune on reduce_dim operation
pub(crate) fn reduce_dim_autotune<
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement + Element,
    EO: JitElement + Element,
>(
    input: JitTensor<R, EI>,
    reduce_dim: usize,
) -> JitTensor<R, EO> {
    let client = input.client.clone();

    let output = init_reduce_output(&input, reduce_dim);
    let id = JitTuneId::new::<R>(&input.device);

    let operation_set = Box::new(ReduceDimAutotuneOperationSet::<RD, R, EI, EO>::new(
        input,
        output.clone(),
        reduce_dim,
    ));

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(&id, &client, operation_set);

    output
}

#[derive(new)]
// Probably better on balanced tensor shapes
pub(crate) struct ReduceDimNaiveAutotune<
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
> {
    input: JitTensor<R, EI>,
    output: JitTensor<R, EO>,
    reduce_dim: usize,
    _algorithm: PhantomData<RD>,
}

impl<RD, R, EI, EO> AutotuneOperation for ReduceDimNaiveAutotune<RD, R, EI, EO>
where
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
{
    fn execute(self: Box<Self>) {
        #[allow(clippy::redundant_closure_call)]
        reduce_dim_naive::<RD, R, EI, EO>(self.input, self.output, self.reduce_dim);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            input: self.input.clone(),
            output: self.output.clone(),
            reduce_dim: self.reduce_dim,
            _algorithm: PhantomData,
        })
    }
}

#[derive(new)]
// Probably better on tensors large along reduce dim
pub(crate) struct ReduceDimSharedAutotune<
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
> {
    input: JitTensor<R, EI>,
    output: JitTensor<R, EO>,
    reduce_dim: usize,
    _algorithm: PhantomData<RD>,
}

impl<RD, R, EI, EO> AutotuneOperation for ReduceDimSharedAutotune<RD, R, EI, EO>
where
    RD: ReduceDimAlgorithm<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
{
    fn execute(self: Box<Self>) {
        #[allow(clippy::redundant_closure_call)]
        reduce_dim_shared::<RD, R, EI, EO>(self.input, self.output, self.reduce_dim);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            input: self.input.clone(),
            output: self.output.clone(),
            reduce_dim: self.reduce_dim,
            _algorithm: PhantomData,
        })
    }
}
