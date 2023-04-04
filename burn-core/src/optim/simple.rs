use core::{any::Any, marker::PhantomData};

use super::{GradientsParams, Optimizer};
use crate::{
    module::{ADModule, ModuleMapper, ParamId},
    record::{Record, RecordSettings},
};
use burn_tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple optimizer is a more opinionated trait where the state can be generic over the
/// dimension D and implements Record. This allows for simpler optimizer implementations where they
/// don't have to handle missing gradients, loading and exporting records, and navigate the
/// module parameter structure.
pub trait SimpleOptimizer<B>: Send + Sync
where
    B: Backend,
{
    type State<const D: usize>: Record + Clone + 'static;

    fn step<const D: usize>(
        &self,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>);

    fn to_device<const D: usize>(state: Self::State<D>, device: &B::Device) -> Self::State<D>;
}

pub struct SimpleModuleOptimizer<O, M, B>
where
    O: SimpleOptimizer<B::InnerBackend>,
    M: ADModule<B>,
    B: ADBackend,
{
    optim: O,
    records: HashMap<ParamId, SimpleOptimizerRecord<O, B::InnerBackend>>,
    module: PhantomData<M>,
}

impl<O, B, M> SimpleModuleOptimizer<O, M, B>
where
    B: ADBackend,
    M: ADModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    pub fn new(optim: O) -> Self {
        Self {
            optim,
            records: HashMap::new(),
            module: PhantomData::default(),
        }
    }
}

impl<O, B, M> Optimizer<M, B> for SimpleModuleOptimizer<O, M, B>
where
    B: ADBackend,
    M: ADModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    type Record = HashMap<ParamId, SimpleOptimizerRecord<O, B::InnerBackend>>;

    fn step(&mut self, module: M, mut grads: GradientsParams) -> M {
        let mut mapper =
            SimpleModuleOptimizerMapper::<M, B, O>::new(&self.optim, &mut self.records, &mut grads);
        module.map(&mut mapper)
    }

    fn to_record(&self) -> Self::Record {
        self.records.clone()
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.records = record;
        self
    }
}

#[derive(new)]
pub struct SimpleModuleOptimizerMapper<'a, M, B, O>
where
    M: ADModule<B>,
    B: ADBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    optimizer: &'a O,
    records: &'a mut HashMap<ParamId, SimpleOptimizerRecord<O, B::InnerBackend>>,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<'a, M, B, O> ModuleMapper<B> for SimpleModuleOptimizerMapper<'a, M, B, O>
where
    M: ADModule<B>,
    B: ADBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let grad = self.grads.remove(id);

        if let Some(grad) = grad {
            let device = grad.device();
            let (key, record) = self.records.remove_entry(id).unzip();
            let (tensor, state) = self.optimizer.step(
                tensor.inner(),
                grad,
                record.map(|record| O::to_device(record.into_state(), &device)),
            );

            if let Some(state) = state {
                self.records.insert(
                    key.unwrap_or_else(|| id.clone()),
                    SimpleOptimizerRecord::from_state(state),
                );
            }

            return Tensor::from_inner(tensor);
        }

        tensor
    }
}

pub enum SimpleOptimizerRecord<O: SimpleOptimizer<B>, B: Backend> {
    Rank1(O::State<1>),
    Rank2(O::State<2>),
    Rank3(O::State<3>),
    Rank4(O::State<4>),
    Rank5(O::State<5>),
    Rank6(O::State<6>),
    Rank7(O::State<7>),
    Rank8(O::State<8>),
}

impl<O: SimpleOptimizer<B>, B: Backend> Clone for SimpleOptimizerRecord<O, B> {
    fn clone(&self) -> Self {
        match self {
            SimpleOptimizerRecord::Rank1(record) => SimpleOptimizerRecord::Rank1(record.clone()),
            SimpleOptimizerRecord::Rank2(record) => SimpleOptimizerRecord::Rank2(record.clone()),
            SimpleOptimizerRecord::Rank3(record) => SimpleOptimizerRecord::Rank3(record.clone()),
            SimpleOptimizerRecord::Rank4(record) => SimpleOptimizerRecord::Rank4(record.clone()),
            SimpleOptimizerRecord::Rank5(record) => SimpleOptimizerRecord::Rank5(record.clone()),
            SimpleOptimizerRecord::Rank6(record) => SimpleOptimizerRecord::Rank6(record.clone()),
            SimpleOptimizerRecord::Rank7(record) => SimpleOptimizerRecord::Rank7(record.clone()),
            SimpleOptimizerRecord::Rank8(record) => SimpleOptimizerRecord::Rank8(record.clone()),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "<O::State<1> as Record>::Item<S>: Serialize + serde::de::DeserializeOwned")]
pub enum SimpleOptimizerRecordItem<O: SimpleOptimizer<B>, B: Backend, S: RecordSettings> {
    Rank1(<O::State<1> as Record>::Item<S>),
    Rank2(<O::State<2> as Record>::Item<S>),
    Rank3(<O::State<3> as Record>::Item<S>),
    Rank4(<O::State<4> as Record>::Item<S>),
    Rank5(<O::State<5> as Record>::Item<S>),
    Rank6(<O::State<6> as Record>::Item<S>),
    Rank7(<O::State<7> as Record>::Item<S>),
    Rank8(<O::State<8> as Record>::Item<S>),
}

impl<O, B> SimpleOptimizerRecord<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    pub fn into_state<const D: usize>(self) -> O::State<D> {
        let boxed_state: Box<dyn Any> = match self {
            SimpleOptimizerRecord::Rank1(s) => Box::new(s),
            SimpleOptimizerRecord::Rank2(s) => Box::new(s),
            SimpleOptimizerRecord::Rank3(s) => Box::new(s),
            SimpleOptimizerRecord::Rank4(s) => Box::new(s),
            SimpleOptimizerRecord::Rank5(s) => Box::new(s),
            SimpleOptimizerRecord::Rank6(s) => Box::new(s),
            SimpleOptimizerRecord::Rank7(s) => Box::new(s),
            SimpleOptimizerRecord::Rank8(s) => Box::new(s),
        };
        let state = boxed_state
            .downcast::<O::State<D>>()
            .expect("Unsupported state dimension");
        *state
    }
    pub fn from_state<const D: usize>(state: O::State<D>) -> Self {
        let state: Box<dyn Any> = Box::new(state);

        match D {
            1 => SimpleOptimizerRecord::Rank1(*state.downcast().unwrap()),
            2 => SimpleOptimizerRecord::Rank2(*state.downcast().unwrap()),
            3 => SimpleOptimizerRecord::Rank3(*state.downcast().unwrap()),
            4 => SimpleOptimizerRecord::Rank4(*state.downcast().unwrap()),
            5 => SimpleOptimizerRecord::Rank5(*state.downcast().unwrap()),
            6 => SimpleOptimizerRecord::Rank6(*state.downcast().unwrap()),
            7 => SimpleOptimizerRecord::Rank7(*state.downcast().unwrap()),
            8 => SimpleOptimizerRecord::Rank8(*state.downcast().unwrap()),
            _ => panic!("Unsupported state dimension"),
        }
    }
}

impl<O, B> Record for SimpleOptimizerRecord<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    type Item<S: RecordSettings> = SimpleOptimizerRecordItem<O, B, S>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        match self {
            SimpleOptimizerRecord::Rank1(record) => {
                SimpleOptimizerRecordItem::Rank1(record.into_item())
            }
            SimpleOptimizerRecord::Rank2(record) => {
                SimpleOptimizerRecordItem::Rank2(record.into_item())
            }
            SimpleOptimizerRecord::Rank3(record) => {
                SimpleOptimizerRecordItem::Rank3(record.into_item())
            }
            SimpleOptimizerRecord::Rank4(record) => {
                SimpleOptimizerRecordItem::Rank4(record.into_item())
            }
            SimpleOptimizerRecord::Rank5(record) => {
                SimpleOptimizerRecordItem::Rank5(record.into_item())
            }
            SimpleOptimizerRecord::Rank6(record) => {
                SimpleOptimizerRecordItem::Rank6(record.into_item())
            }
            SimpleOptimizerRecord::Rank7(record) => {
                SimpleOptimizerRecordItem::Rank7(record.into_item())
            }
            SimpleOptimizerRecord::Rank8(record) => {
                SimpleOptimizerRecordItem::Rank8(record.into_item())
            }
        }
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        match item {
            SimpleOptimizerRecordItem::Rank1(item) => {
                SimpleOptimizerRecord::Rank1(<O::State<1> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank2(item) => {
                SimpleOptimizerRecord::Rank2(<O::State<2> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank3(item) => {
                SimpleOptimizerRecord::Rank3(<O::State<3> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank4(item) => {
                SimpleOptimizerRecord::Rank4(<O::State<4> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank5(item) => {
                SimpleOptimizerRecord::Rank5(<O::State<5> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank6(item) => {
                SimpleOptimizerRecord::Rank6(<O::State<6> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank7(item) => {
                SimpleOptimizerRecord::Rank7(<O::State<7> as Record>::from_item(item))
            }
            SimpleOptimizerRecordItem::Rank8(item) => {
                SimpleOptimizerRecord::Rank8(<O::State<8> as Record>::from_item(item))
            }
        }
    }
}
