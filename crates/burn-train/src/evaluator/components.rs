use std::marker::PhantomData;

use burn_core::{module::Module, prelude::Backend};

use crate::metric::{ItemLazy, processor::EventProcessor};

/// All components necessary to train a model grouped in one trait.
pub trait EvaluatorComponentTypes {
    /// The backend in used for the training.
    type Backend: Backend;
    /// The model to train.
    type Model: Module<Self::Backend>
        + TestStep<Self::TestInput, Self::TestOutput>
        + core::fmt::Display
        + 'static;
    type EventProcessor: EventProcessor<ItemTrain = Self::TestOutput, ItemValid = ()> + 'static;
    /// Type of input to the evaluation step
    type TestInput: Send + 'static;
    /// Type of output of the evaluation step
    type TestOutput: ItemLazy + 'static;
}

// Trait to be implemented for validating models.
pub trait TestStep<TI, TO> {
    /// Runs a test step.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to validate on.
    ///
    /// # Returns
    ///
    /// The test output.
    fn step(&self, item: TI) -> TO;
}

pub struct EvaluatorComponentTypesMarker<B, M, E, TI, TO> {
    _p: PhantomData<(B, M, E, TI, TO)>,
}

impl<B, M, E, TI, TO> EvaluatorComponentTypes for EvaluatorComponentTypesMarker<B, M, E, TI, TO>
where
    B: Backend,
    M: Module<B> + TestStep<TI, TO> + core::fmt::Display + 'static,
    E: EventProcessor<ItemTrain = TO, ItemValid = ()> + 'static,
    TI: Send + 'static,
    TO: ItemLazy + 'static,
{
    type Backend = B;
    type Model = M;
    type EventProcessor = E;
    type TestInput = TI;
    type TestOutput = TO;
}
