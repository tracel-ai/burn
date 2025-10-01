//! # Dataset Transformations
//!
//! This module provides a collection of [`crate::Dataset`] composition wrappers;
//! providing composition, subset selection, sampling, random shuffling, and windowing.
//!
//! * [`ComposedDataset`] - composes a list of datasets.
//! * [`PartialDataset`] - selects a contiguous index range subset of a dataset.
//! * [`ShuffledDataset`] - a randomly shuffled / mutably shuffle-able dataset;
//!   a thin wrapper around [`SelectionDataset`].
//! * [`SamplerDataset`] - samples a dataset; support for with/without replacement,
//!   and under/oversampling.
//! * [`SelectionDataset`] - selects a subset of a dataset via indices; support for shuffling.
//! * [`WindowsDataset`] - creates a sliding window over a dataset.
mod composed;
mod mapper;
mod options;
mod partial;
mod sampler;
mod selection;
mod shuffle;
mod window;

pub use composed::*;
pub use mapper::*;
pub use options::*;
pub use partial::*;
pub use sampler::*;
pub use selection::*;
pub use shuffle::*;
pub use window::*;
