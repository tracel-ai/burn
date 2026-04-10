use burn_core as burn;

use super::{AdaptorRecordItemV1, AdaptorRecordV1};
use crate::optim::SimpleOptimizer;
use burn::record::{PrecisionSettings, Record};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record.
///
/// Records are versioned for backward compatibility, so old records can be loaded.
pub enum AdaptorRecord<O>
where
    O: SimpleOptimizer,
{
    /// Version 1.
    V1(AdaptorRecordV1<O>),
}

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record item.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub enum AdaptorRecordItem<O: SimpleOptimizer, S: PrecisionSettings> {
    /// Version 1.
    V1(AdaptorRecordItemV1<O, S>),
}

impl<O> Record for AdaptorRecord<O>
where
    O: SimpleOptimizer,
{
    type Item<S: PrecisionSettings> = AdaptorRecordItem<O, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            AdaptorRecord::V1(record) => AdaptorRecordItem::V1(record.into_item()),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &Device) -> Self {
        match item {
            AdaptorRecordItem::V1(item) => Self::V1(AdaptorRecordV1::from_item(item, device)),
        }
    }
}

impl<O> Clone for AdaptorRecord<O>
where
    O: SimpleOptimizer,
{
    fn clone(&self) -> Self {
        match self {
            AdaptorRecord::V1(record) => Self::V1(record.clone()),
        }
    }
}

impl<O> AdaptorRecord<O>
where
    O: SimpleOptimizer,
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
