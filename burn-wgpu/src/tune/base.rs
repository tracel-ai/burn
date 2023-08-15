use std::{collections::HashMap, fmt::Display, hash::Hash, sync::Arc, time::Duration};

use burn_common::stub::RwLock;

use crate::{
    benchmark::{Benchmark, BenchmarkResult},
    context::Context,
    GraphicsApi, WgpuDevice,
};

/// Key used for caching.
#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub struct AutoTuneKey {
    /// List all shapes used for the autotuned kernel.
    shapes: Vec<Vec<usize>>,
    /// Operation name.
    ops: String,
    /// Device name used to benchmark.
    device: String,
    /// Graphics api name.
    graphics_api: String,
}

impl AutoTuneKey {
    pub fn new(shapes: Vec<Vec<usize>>, ops: String, context: &Context) -> Self {
        let device = format!("{:?}", context.info.name);
        let graphics_api = format!("{:?}", context.info.backend);

        Self {
            shapes,
            ops,
            device,
            graphics_api,
        }
    }
}

impl Display for AutoTuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = format!(
            "(AutoTuneKey) Kernel {} - Shapes {:?} - Device {} - API {}",
            self.ops, self.shapes, self.device, self.graphics_api,
        );
        f.write_str(&message)
    }
}

/// Objects that are stored in the tuner cache. Can have any inputs and outputs.
pub type AutoTuneValue = Box<dyn core::any::Any + Send + Sync>;

/// Executable function
pub trait KernelFunction: Send + Sync + 'static {
    type Input;
    type Output;

    fn call(&self, input: Self::Input) -> Self::Output;
    fn description(&self) -> String;
}

/// Encapsulates kernel functions, with specified inputs and outputs
pub type AutoTuneFunction<I, O> = Arc<dyn KernelFunction<Input = I, Output = O>>;

/// The tunable links an executable function to its corresponding benchmark
#[derive(new)]
pub struct Tunable<G, I, O> {
    func: AutoTuneFunction<I, O>,
    benchmark: Arc<dyn Benchmark<G, Args = I>>,
}

impl<G, I, O> std::fmt::Display for Tunable<G, I, O>
where
    G: GraphicsApi,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.func.description().as_str())
    }
}

/// Output of the tuner execution. If execution succeeded, the output of
/// the execution is contained. Otherwise, the function must be tuned and
/// the input is given back to preserve ownership.
#[derive(Debug)]
pub enum Execution<I, O> {
    Executed(O),
    NoCacheFound(I),
}

/// The tuner allows to find the best version of a kernel by benchmarking
/// different versions. It keeps the best version found in a cache, so the best
/// function is reused automatically in similar circumstances.
#[derive(Debug)]
pub struct Tuner {
    cache: RwLock<HashMap<AutoTuneKey, AutoTuneValue>>,
}

impl Tuner {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Executes the function stored in the cache at key id, on specified input,
    /// and returns its output. If cache has no such id, returns NoCacheFound.
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

    /// Finds the best tunable and writes it to the cache.
    pub fn tune<G: GraphicsApi, I, O>(
        &self,
        id: AutoTuneKey,
        input: I,
        tunables: Vec<Tunable<G, I, O>>,
        device: &WgpuDevice,
    ) -> O
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
    {
        let mut cache = self.cache.write().unwrap();

        let results = benchmark(&tunables, device);
        let kernel = find_best(&id, tunables, results);
        cache.insert(id.clone(), kernel);
        drop(cache);

        match self.execute(&id, input) {
            Execution::Executed(output) => output,
            _ => panic!("Should have found a kernel to execute. "),
        }
    }
}

/// Finds the best kernel by keeping the one with the smallest median duration.
fn find_best<G: GraphicsApi, I, O>(
    id: &AutoTuneKey,
    tunables: Vec<Tunable<G, I, O>>,
    results: Vec<BenchmarkResult>,
) -> AutoTuneValue
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    let mut best_duration = Duration::MAX;
    let mut best_tunable = None;

    for (tunable, result) in tunables.into_iter().zip(results) {
        let duration = result.median_duration();

        if duration < best_duration {
            best_duration = duration;
            best_tunable = Some(tunable);
        }
    }

    let tunable = best_tunable.expect("At least one tunable needed. ");

    log::info!("{} => {}", id, tunable);
    Box::new(tunable.func)
}

/// Run benchmarks.
fn benchmark<G: GraphicsApi, I, O>(
    tunables: &[Tunable<G, I, O>],
    device: &WgpuDevice,
) -> Vec<BenchmarkResult>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    tunables
        .iter()
        .map(|tunable| tunable.benchmark.run(device))
        .collect()
}
