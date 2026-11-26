use crate::{
    Adam, AdamState, AdamW, AdamWState, AdaptiveMomentumState, OptimizerStateStorage, RecordState,
    StatisticCodec,
};
use burn_core::tensor::{Device, FloatDType};
use serde::{Deserialize, Serialize};

/// Storage policy for the moments shared by [`Adam`] and [`AdamW`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdamMomentStorage<First, Second, Maximum = Second> {
    first: First,
    second: Second,
    maximum: Maximum,
}

impl<First, Second: Clone> AdamMomentStorage<First, Second> {
    /// Choose codecs for the first and second moments. AMSGrad's running maximum uses
    /// the second-moment codec unless overridden with [`Self::with_maximum`].
    pub fn new(first: First, second: Second) -> Self {
        Self {
            first,
            maximum: second.clone(),
            second,
        }
    }
}

impl<First, Second, Maximum> AdamMomentStorage<First, Second, Maximum> {
    /// Choose a separate codec for AMSGrad's running maximum.
    pub fn with_maximum<M>(self, maximum: M) -> AdamMomentStorage<First, Second, M> {
        AdamMomentStorage {
            first: self.first,
            second: self.second,
            maximum,
        }
    }
}

/// Checkpoint record preserving the Adam momentum namespace and step counter.
#[derive(Clone, RecordState)]
pub struct StoredAdamState<F: RecordState, S: RecordState, M: RecordState> {
    momentum: StoredMoments<F, S, M>,
}

#[derive(Clone, RecordState)]
struct StoredMoments<F: RecordState, S: RecordState, M: RecordState> {
    time: usize,
    moment_1: F,
    moment_2: S,
    max_moment_2: Option<M>,
}

type EncodedAdamState<const D: usize, F, S, M> = StoredAdamState<
    <F as StatisticCodec>::Encoded<D>,
    <S as StatisticCodec>::Encoded<D>,
    <M as StatisticCodec>::Encoded<D>,
>;

impl<F: StatisticCodec, S: StatisticCodec, M: StatisticCodec> AdamMomentStorage<F, S, M> {
    fn encode<const D: usize>(
        &self,
        state: AdaptiveMomentumState<D>,
    ) -> EncodedAdamState<D, F, S, M> {
        StoredAdamState {
            momentum: StoredMoments {
                time: state.time,
                moment_1: self.first.encode(state.moment_1),
                moment_2: self.second.encode(state.moment_2),
                max_moment_2: state.max_moment_2.map(|t| self.maximum.encode(t)),
            },
        }
    }

    fn decode<const D: usize>(
        &self,
        stored: EncodedAdamState<D, F, S, M>,
        compute: FloatDType,
    ) -> AdaptiveMomentumState<D> {
        let state = stored.momentum;
        AdaptiveMomentumState {
            time: state.time,
            moment_1: self.first.decode(state.moment_1, compute),
            moment_2: self.second.decode(state.moment_2, compute),
            max_moment_2: state.max_moment_2.map(|s| self.maximum.decode(s, compute)),
        }
    }

    fn move_record<const D: usize>(
        stored: EncodedAdamState<D, F, S, M>,
        device: &Device,
    ) -> EncodedAdamState<D, F, S, M> {
        let state = stored.momentum;
        StoredAdamState {
            momentum: StoredMoments {
                time: state.time,
                moment_1: F::to_device::<D>(state.moment_1, device),
                moment_2: S::to_device::<D>(state.moment_2, device),
                max_moment_2: state.max_moment_2.map(|s| M::to_device::<D>(s, device)),
            },
        }
    }
}

macro_rules! impl_adam_storage {
    ($optim:ty, $state:ident) => {
        impl<F: StatisticCodec, S: StatisticCodec, M: StatisticCodec> OptimizerStateStorage<$optim>
            for AdamMomentStorage<F, S, M>
        {
            type Stored<const D: usize> = EncodedAdamState<D, F, S, M>;

            fn store<const D: usize>(&self, state: $state<D>) -> Self::Stored<D> {
                self.encode(state.momentum)
            }

            fn restore<const D: usize>(
                &self,
                stored: Self::Stored<D>,
                compute: FloatDType,
            ) -> $state<D> {
                $state::new(self.decode(stored, compute))
            }

            fn to_device<const D: usize>(
                stored: Self::Stored<D>,
                device: &Device,
            ) -> Self::Stored<D> {
                Self::move_record::<D>(stored, device)
            }
        }
    };
}
impl_adam_storage!(Adam, AdamState);
impl_adam_storage!(AdamW, AdamWState);
