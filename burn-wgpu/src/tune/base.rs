use std::{collections::HashMap, sync::Arc};

use burn_common::stub::{Mutex, RwLock};

use crate::benchmark::Benchmark;

#[derive(new, Hash, Clone, Debug, PartialEq, Eq)]
pub struct AutoTuneKey {
    /// List all shapes used for the autotuned kernel.
    shapes: Vec<Vec<usize>>,
    /// Name of the operation.
    ops_name: String,
}

pub trait KernelFunction: Send + Sync + 'static {
    type Input;
    type Output;

    fn call(&self, input: Self::Input) -> Self::Output;
}

pub type AutoTuneValue = Box<dyn core::any::Any + Send + Sync>;
pub type AutoTuneFunction<I, O> = Arc<dyn KernelFunction<Input = I, Output = O>>;

#[derive(Debug)]
pub struct Tuner {
    cache: RwLock<HashMap<AutoTuneKey, AutoTuneValue>>,
}

#[derive(Debug)]
pub enum Execution<I, O> {
    Executed(O),
    NoCacheFound(I),
}

#[derive(new)]
pub struct Tunable<G, I, O> {
    func: AutoTuneFunction<I, O>,
    benchmark: Arc<dyn Benchmark<G, Args = I>>,
}

impl Tuner {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn execute<I, O>(&self, id: &AutoTuneKey, input: I) -> Execution<I, O>
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
    {
        let cache = self.cache.read().unwrap();
        let obj = cache.get(id);

        let obj = match obj {
            None => return Execution::NoCacheFound(input),
            Some(value) => value,
        };

        let func: &Arc<dyn KernelFunction<Input = I, Output = O>> = obj.downcast_ref().unwrap();

        let output = func.call(input);

        Execution::Executed(output)
    }

    pub fn tune<G, I, O>(&self, id: AutoTuneKey, input: I, mut tunables: Vec<Tunable<G, I, O>>) -> O
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
    {
        let mut cache = self.cache.write().unwrap();

        cache.insert(id.clone(), Box::new(tunables.pop().unwrap().func));

        match self.execute(&id, input) {
            Execution::Executed(output) => output,
            _ => panic!(""),
        }
    }
}
