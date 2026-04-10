use crate::InferenceStep;
use burn_core::module::Module;
use std::marker::PhantomData;

/// All components necessary to evaluate a model grouped in one trait.
pub trait EvaluatorComponentTypes {
    /// The model to evaluate.
    type Model: Module + InferenceStep + core::fmt::Display + 'static;
}

/// A marker type used to provide [evaluation components](EvaluatorComponentTypes).
pub struct EvaluatorComponentTypesMarker<M> {
    _p: PhantomData<M>,
}

impl<M> EvaluatorComponentTypes for EvaluatorComponentTypesMarker<M>
where
    M: Module + InferenceStep + core::fmt::Display + 'static,
{
    type Model = M;
}
