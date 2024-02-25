use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{
    backend::Backend, BasicOps, Bool, Device, DynRankData, Float, Int, Tensor, TensorKind,
};

#[derive(new, Clone, Debug)]
pub struct DynRankTensor<B, K = Float>
where
    B: Backend,
    K: DynRankTensorOps<B>,
{
    pub(crate) primitive: K::DynRankPrimitive,
}

impl<B, K> DynRankTensor<B, K>
where
    B: Backend,
    K: DynRankTensorOps<B>,
{
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<DynRankData<K::Elem>>,
    {
        Self::new(K::from_data(data.into(), device))
    }

    pub fn into_data(self) -> DynRankData<K::Elem> {
        K::into_data(self.primitive)
    }

    pub fn to_data(&self) -> DynRankData<K::Elem> {
        self.clone().into_data()
    }

    pub fn from_primitive(primitive: K::DynRankPrimitive) -> Self {
        Self { primitive }
    }

    pub fn into_primitive(self) -> K::DynRankPrimitive {
        self.primitive
    }
}

impl<B, const D: usize, K> From<DynRankTensor<B, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: DynRankTensorOps<B> + BasicOps<B>,
{
    fn from(value: DynRankTensor<B, K>) -> Self {
        Tensor::from_primitive(K::from_dyn_rank(value.primitive))
    }
}

impl<B, const D: usize, K> From<Tensor<B, D, K>> for DynRankTensor<B, K>
where
    B: Backend,
    K: DynRankTensorOps<B> + BasicOps<B>,
{
    fn from(value: Tensor<B, D, K>) -> Self {
        Self::from_primitive(K::into_dyn_rank(value.primitive).read())
    }
}

impl<B, K> Serialize for DynRankTensor<B, K>
where
    B: Backend,
    K: DynRankTensorOps<B>,
    K::Elem: Debug + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_data().serialize(serializer)
    }
}

impl<'de, B, K> Deserialize<'de> for DynRankTensor<B, K>
where
    B: Backend,
    K: DynRankTensorOps<B>,
    K::Elem: Debug + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::from_data(
            DynRankData::<K::Elem>::deserialize(deserializer)?,
            &Default::default(),
        ))
    }
}

pub trait DynRankTensorOps<B: Backend>: TensorKind<B> {
    type Elem: 'static;

    fn into_data(tensor: Self::DynRankPrimitive) -> DynRankData<Self::Elem>;

    fn from_data(data: DynRankData<Self::Elem>, device: &Device<B>) -> Self::DynRankPrimitive;
}

impl<B: Backend> DynRankTensorOps<B> for Float {
    type Elem = B::FloatElem;

    fn into_data(tensor: Self::DynRankPrimitive) -> DynRankData<Self::Elem> {
        B::float_dyn_rank_into_data(tensor).read()
    }

    fn from_data(data: DynRankData<Self::Elem>, device: &Device<B>) -> Self::DynRankPrimitive {
        B::float_dyn_rank_from_data(data, device)
    }
}

impl<B: Backend> DynRankTensorOps<B> for Int {
    type Elem = B::IntElem;

    fn into_data(tensor: Self::DynRankPrimitive) -> DynRankData<Self::Elem> {
        B::int_dyn_rank_into_data(tensor).read()
    }

    fn from_data(data: DynRankData<Self::Elem>, device: &Device<B>) -> Self::DynRankPrimitive {
        B::int_dyn_rank_from_data(data, device)
    }
}

impl<B: Backend> DynRankTensorOps<B> for Bool {
    type Elem = bool;

    fn into_data(tensor: Self::DynRankPrimitive) -> DynRankData<Self::Elem> {
        B::bool_dyn_rank_into_data(tensor).read()
    }

    fn from_data(data: DynRankData<Self::Elem>, device: &Device<B>) -> Self::DynRankPrimitive {
        B::bool_dyn_rank_from_data(data, device)
    }
}
