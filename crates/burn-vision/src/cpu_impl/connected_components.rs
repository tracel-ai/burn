use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
};

use crate::{ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity};

pub fn connected_components<B: Backend>(
    _img: BoolTensor<B>,
    _connectivity: Connectivity,
) -> IntTensor<B> {
    todo!()
}

pub fn connected_components_with_stats<B: Backend>(
    _img: BoolTensor<B>,
    _connectivity: Connectivity,
    _options: ConnectedStatsOptions,
) -> (IntTensor<B>, ConnectedStatsPrimitive<B>) {
    todo!()
}
