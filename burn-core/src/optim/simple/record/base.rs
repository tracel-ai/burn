use super::{AdaptorRecordItemV1, AdaptorRecordV1};
use crate::{
    optim::SimpleOptimizer,
    record::{PrecisionSettings, Record},
};
use burn_tensor::backend::Backend;
use serde::{Deserialize, Serialize};

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record.
///
/// Records are versioned for backward compatibility, so old records can be loaded.
pub enum AdaptorRecord<O: SimpleOptimizer<B>, B: Backend> {
    /// Version 1.
    V1(AdaptorRecordV1<O, B>),
}

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record item.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub enum AdaptorRecordItem<O: SimpleOptimizer<B>, B: Backend, S: PrecisionSettings> {
    /// Version 1.
    V1(AdaptorRecordItemV1<O, B, S>),
}

impl<O, B> Record for AdaptorRecord<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    type Item<S: PrecisionSettings> = AdaptorRecordItem<O, B, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            AdaptorRecord::V1(record) => AdaptorRecordItem::V1(record.into_item()),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        match item {
            AdaptorRecordItem::V1(item) => Self::V1(AdaptorRecordV1::from_item(item)),
        }
    }
}

impl<O, B> Clone for AdaptorRecord<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    fn clone(&self) -> Self {
        match self {
            AdaptorRecord::V1(record) => Self::V1(record.clone()),
        }
    }
}

impl<O, B> AdaptorRecord<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    /// Converts the record into the optimizer state.
    ///
    /// # Returns
    ///
    /// The optimizer state.
    pub fn into_state<const D: usize>(self) -> O::State<D> {
        match self {
            AdaptorRecord::V1(record) => record.into_state(),
        }
    }

    /// Converts the optimizer state into the record.
    ///
    /// # Arguments
    ///
    /// * `state`: The optimizer state.
    ///
    /// # Returns
    ///
    /// The record.
    pub fn from_state<const D: usize>(state: O::State<D>) -> Self {
        Self::V1(AdaptorRecordV1::from_state(state))
    }
}
