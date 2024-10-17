use crate::metric::processor::EventProcessor;
use burn_core::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use std::marker::PhantomData;

/// All components necessary to train a model grouped in one trait.
pub trait TesterComponents {
    /// The backend in used for the training.
    type Backend: AutodiffBackend;
    /// The model to train.
    type Model: AutodiffModule<Self::Backend> + core::fmt::Display + 'static;
    type EventProcessor: EventProcessor + 'static;
}

/// Concrete type that implements [training components trait](TrainingComponents).
pub struct TesterComponentsMarker<B, M, EP> {
    _backend: PhantomData<B>,
    _model: PhantomData<M>,
    _event_processor: PhantomData<EP>,
}

impl<B, M, EP> TesterComponents for TesterComponentsMarker<B, M, EP>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    EP: EventProcessor + 'static,
{
    type Backend = B;
    type Model = M;
    type EventProcessor = EP;
}
