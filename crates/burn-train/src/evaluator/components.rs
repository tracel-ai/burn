use crate::InferenceStep;
use burn_core::{module::Module, prelude::Backend};
use std::marker::PhantomData;

/// All components necessary to evaluate a model grouped in one trait.
pub trait EvaluatorComponentTypes {
    /// The backend in used for the evaluation.
    type Backend: Backend;
    /// The model to evaluate.
    type Model: Module<Self::Backend> + InferenceStep + core::fmt::Display + 'static;
}

/// A marker type used to provide [evaluation components](EvaluatorComponentTypes).
pub struct EvaluatorComponentTypesMarker<B, M> {
    _p: PhantomData<(B, M)>,
}

impl<B, M> EvaluatorComponentTypes for EvaluatorComponentTypesMarker<B, M>
where
    B: Backend,
    M: Module<B> + InferenceStep + core::fmt::Display + 'static,
{
    type Backend = B;
    type Model = M;
}
