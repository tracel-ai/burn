use burn_common::benchmark::Benchmark;
use core::marker::PhantomData;
use hashbrown::HashMap;

use crate::server::ComputeServer;

use super::{InputHashable, Operation};

pub trait TuneBenchmark<O, S: ComputeServer>: Benchmark {
    fn take_kernel(&self) -> S::Kernel;
}

#[derive(new)]
pub struct KernelPool<TB, O, S> {
    cache: HashMap<String, usize>,
    pub tune_benchmarks: Vec<TB>,
    _operation: PhantomData<O>,
    _server: PhantomData<S>,
}

impl<TB: TuneBenchmark<O, S>, O: Operation, S: ComputeServer> KernelPool<TB, O, S> {
    pub(crate) fn try_cache(&self, input: &O::Input) -> Option<S::Kernel> {
        let index = self.cache.get(&input.custom_hash());
        if let Some(&i) = index {
            return Some(self.tune_benchmarks[i].take_kernel());
        }
        None
    }

    pub(crate) fn get(&self, index: usize) -> S::Kernel {
        (*self.tune_benchmarks.get(index).unwrap()).take_kernel()
    }

    pub(crate) fn add_to_cache(&mut self, input: &O::Input, index: usize) -> () {
        self.cache.insert(input.custom_hash(), index);
    }
}
