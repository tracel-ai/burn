use core::marker::PhantomData;

use alloc::sync::Arc;
use burn_common::benchmark::Benchmark;
use hashbrown::HashMap;

use crate::{channel::ComputeChannel, server::ComputeServer};

use super::{InputHashable, Operation};

pub trait TuneBenchmark: Benchmark {
    type Args: InputHashable;
}

#[derive(new)]
pub struct KernelType<TB, O, S: ComputeServer, C> {
    pub kernel: Arc<S::Kernel>,
    pub tune_benchmark: Arc<TB>,
    pub _operation: PhantomData<O>,
    pub _channel: PhantomData<C>,
}

impl<TB, O, S: ComputeServer, C> Clone for KernelType<TB, O, S, C> {
    fn clone(&self) -> KernelType<TB, O, S, C> {
        KernelType {
            kernel: self.kernel.clone(),
            tune_benchmark: self.tune_benchmark.clone(),
            _operation: PhantomData,
            _channel: PhantomData,
        }
    }
}

impl<TB, O, S, C> KernelType<TB, O, S, C>
where
    TB: TuneBenchmark,
    O: Operation<S>,
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    pub(crate) fn to_benchmark(&self) -> Arc<TB> {
        self.tune_benchmark.clone()
    }
}

#[derive(new)]
pub struct KernelPool<TB, O, S, C>
where
    O: Operation<S>,
    S: ComputeServer,
{
    cache: HashMap<String, usize>,
    pub kernel_types: Vec<KernelType<TB, O, S, C>>,
    _operation: PhantomData<O>,
}

impl<TB, O: Operation<S>, S: ComputeServer, C> KernelPool<TB, O, S, C> {
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<KernelType<TB, O, S, C>> {
        let index = self.cache.get(&input.custom_hash());
        index.map(|i| self.kernel_types[*i].clone())
    }

    pub(crate) fn get(&self, index: usize) -> KernelType<TB, O, S, C> {
        (*self.kernel_types.get(index).unwrap()).clone()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
