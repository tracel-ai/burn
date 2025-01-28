use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
    Int, Tensor,
};

use crate::cpu_impl;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Connectivity {
    Four,
    Eight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnectedStatsOptions {
    pub area_enabled: bool,
    pub top_enabled: bool,
    pub left_enabled: bool,
    pub right_enabled: bool,
    pub bottom_enabled: bool,
}

#[derive(Clone, Debug)]
pub struct ConnectedStats<B: Backend> {
    pub area: Tensor<B, 3, Int>,
    pub top: Tensor<B, 3, Int>,
    pub left: Tensor<B, 3, Int>,
    pub right: Tensor<B, 3, Int>,
    pub bottom: Tensor<B, 3, Int>,
}

pub struct ConnectedStatsPrimitive<B: Backend> {
    pub area: IntTensor<B>,
    pub left: IntTensor<B>,
    pub top: IntTensor<B>,
    pub right: IntTensor<B>,
    pub bottom: IntTensor<B>,
}

impl<B: Backend> From<ConnectedStatsPrimitive<B>> for ConnectedStats<B> {
    fn from(value: ConnectedStatsPrimitive<B>) -> Self {
        ConnectedStats {
            area: Tensor::from_primitive(value.area),
            top: Tensor::from_primitive(value.top),
            left: Tensor::from_primitive(value.left),
            right: Tensor::from_primitive(value.right),
            bottom: Tensor::from_primitive(value.bottom),
        }
    }
}

impl Default for ConnectedStatsOptions {
    fn default() -> Self {
        Self::all()
    }
}

impl ConnectedStatsOptions {
    pub fn none() -> Self {
        Self {
            area_enabled: false,
            top_enabled: false,
            left_enabled: false,
            right_enabled: false,
            bottom_enabled: false,
        }
    }

    pub fn all() -> Self {
        Self {
            area_enabled: true,
            top_enabled: true,
            left_enabled: true,
            right_enabled: true,
            bottom_enabled: true,
        }
    }
}

pub trait VisionOps<B: Backend> {
    fn connected_components(img: BoolTensor<B>, connectivity: Connectivity) -> IntTensor<B> {
        cpu_impl::connected_components::<B>(img, connectivity)
    }

    fn connected_components_with_stats(
        img: BoolTensor<B>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<B>, ConnectedStatsPrimitive<B>) {
        cpu_impl::connected_components_with_stats(img, connectivity, opts)
    }
}
