use burn_core as burn;

use crate::optim::SimpleOptimizer;
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::Backend;
use core::any::Any;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record item.
pub enum AdaptorRecordV1<O: SimpleOptimizer<B>, B: Backend> {
    /// Rank 0.
    Rank0(O::State<0>),

    /// Rank 1.
    Rank1(O::State<1>),

    /// Rank 2.
    Rank2(O::State<2>),

    /// Rank 3.
    Rank3(O::State<3>),

    /// Rank 4.
    Rank4(O::State<4>),

    /// Rank 5.
    Rank5(O::State<5>),

    /// Rank 6.
    Rank6(O::State<6>),

    /// Rank 7.
    Rank7(O::State<7>),

    /// Rank 8.
    Rank8(O::State<8>),
}

impl<O: SimpleOptimizer<B>, B: Backend> Clone for AdaptorRecordV1<O, B> {
    fn clone(&self) -> Self {
        match self {
            AdaptorRecordV1::Rank0(record) => AdaptorRecordV1::Rank0(record.clone()),
            AdaptorRecordV1::Rank1(record) => AdaptorRecordV1::Rank1(record.clone()),
            AdaptorRecordV1::Rank2(record) => AdaptorRecordV1::Rank2(record.clone()),
            AdaptorRecordV1::Rank3(record) => AdaptorRecordV1::Rank3(record.clone()),
            AdaptorRecordV1::Rank4(record) => AdaptorRecordV1::Rank4(record.clone()),
            AdaptorRecordV1::Rank5(record) => AdaptorRecordV1::Rank5(record.clone()),
            AdaptorRecordV1::Rank6(record) => AdaptorRecordV1::Rank6(record.clone()),
            AdaptorRecordV1::Rank7(record) => AdaptorRecordV1::Rank7(record.clone()),
            AdaptorRecordV1::Rank8(record) => AdaptorRecordV1::Rank8(record.clone()),
        }
    }
}

/// [Optimizer adaptor](crate::optim::simple::adaptor::OptimizerAdaptor) record item.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub enum AdaptorRecordItemV1<O: SimpleOptimizer<B>, B: Backend, S: PrecisionSettings> {
    /// Rank 0.
    Rank0(<O::State<0> as Record<B>>::Item<S>),

    /// Rank 1.
    Rank1(<O::State<1> as Record<B>>::Item<S>),

    /// Rank 2.
    Rank2(<O::State<2> as Record<B>>::Item<S>),

    /// Rank 3.
    Rank3(<O::State<3> as Record<B>>::Item<S>),

    /// Rank 4.
    Rank4(<O::State<4> as Record<B>>::Item<S>),

    /// Rank 5.
    Rank5(<O::State<5> as Record<B>>::Item<S>),

    /// Rank 6.
    Rank6(<O::State<6> as Record<B>>::Item<S>),

    /// Rank 7.
    Rank7(<O::State<7> as Record<B>>::Item<S>),

    /// Rank 8.
    Rank8(<O::State<8> as Record<B>>::Item<S>),
}

impl<O, B> AdaptorRecordV1<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    /// Convert the record into the state.
    ///
    /// # Returns
    ///
    /// The state.
    ///
    /// # Panics
    ///
    /// Panics if the state dimension is not supported.
    pub fn into_state<const D: usize>(self) -> O::State<D> {
        let boxed_state: Box<dyn Any> = match self {
            AdaptorRecordV1::Rank0(s) => Box::new(s),
            AdaptorRecordV1::Rank1(s) => Box::new(s),
            AdaptorRecordV1::Rank2(s) => Box::new(s),
            AdaptorRecordV1::Rank3(s) => Box::new(s),
            AdaptorRecordV1::Rank4(s) => Box::new(s),
            AdaptorRecordV1::Rank5(s) => Box::new(s),
            AdaptorRecordV1::Rank6(s) => Box::new(s),
            AdaptorRecordV1::Rank7(s) => Box::new(s),
            AdaptorRecordV1::Rank8(s) => Box::new(s),
        };
        let state = boxed_state
            .downcast::<O::State<D>>()
            .expect("Unsupported state dimension, dimension up to 8 are supported.");
        *state
    }

    /// Convert the state into the record.
    ///
    /// # Arguments
    ///
    /// * `state`: The state.
    ///
    /// # Returns
    ///
    /// The record.
    pub fn from_state<const D: usize>(state: O::State<D>) -> Self {
        let state: Box<dyn Any> = Box::new(state);

        match D {
            0 => AdaptorRecordV1::Rank0(*state.downcast().unwrap()),
            1 => AdaptorRecordV1::Rank1(*state.downcast().unwrap()),
            2 => AdaptorRecordV1::Rank2(*state.downcast().unwrap()),
            3 => AdaptorRecordV1::Rank3(*state.downcast().unwrap()),
            4 => AdaptorRecordV1::Rank4(*state.downcast().unwrap()),
            5 => AdaptorRecordV1::Rank5(*state.downcast().unwrap()),
            6 => AdaptorRecordV1::Rank6(*state.downcast().unwrap()),
            7 => AdaptorRecordV1::Rank7(*state.downcast().unwrap()),
            8 => AdaptorRecordV1::Rank8(*state.downcast().unwrap()),
            _ => panic!("Unsupported state dimension, dimension up to 8 are supported."),
        }
    }
}

impl<O, B> Record<B> for AdaptorRecordV1<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    type Item<S: PrecisionSettings> = AdaptorRecordItemV1<O, B, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            AdaptorRecordV1::Rank0(record) => AdaptorRecordItemV1::Rank0(record.into_item()),
            AdaptorRecordV1::Rank1(record) => AdaptorRecordItemV1::Rank1(record.into_item()),
            AdaptorRecordV1::Rank2(record) => AdaptorRecordItemV1::Rank2(record.into_item()),
            AdaptorRecordV1::Rank3(record) => AdaptorRecordItemV1::Rank3(record.into_item()),
            AdaptorRecordV1::Rank4(record) => AdaptorRecordItemV1::Rank4(record.into_item()),
            AdaptorRecordV1::Rank5(record) => AdaptorRecordItemV1::Rank5(record.into_item()),
            AdaptorRecordV1::Rank6(record) => AdaptorRecordItemV1::Rank6(record.into_item()),
            AdaptorRecordV1::Rank7(record) => AdaptorRecordItemV1::Rank7(record.into_item()),
            AdaptorRecordV1::Rank8(record) => AdaptorRecordItemV1::Rank8(record.into_item()),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        match item {
            AdaptorRecordItemV1::Rank0(item) => {
                AdaptorRecordV1::Rank0(<O::State<0> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank1(item) => {
                AdaptorRecordV1::Rank1(<O::State<1> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank2(item) => {
                AdaptorRecordV1::Rank2(<O::State<2> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank3(item) => {
                AdaptorRecordV1::Rank3(<O::State<3> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank4(item) => {
                AdaptorRecordV1::Rank4(<O::State<4> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank5(item) => {
                AdaptorRecordV1::Rank5(<O::State<5> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank6(item) => {
                AdaptorRecordV1::Rank6(<O::State<6> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank7(item) => {
                AdaptorRecordV1::Rank7(<O::State<7> as Record<B>>::from_item(item, device))
            }
            AdaptorRecordItemV1::Rank8(item) => {
                AdaptorRecordV1::Rank8(<O::State<8> as Record<B>>::from_item(item, device))
            }
        }
    }
}
