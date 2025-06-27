use std::time::{Duration, Instant};
use burn::tensor::backend::Backend;

/// Configuration for inference timing
pub struct TimingConfig {
    /// Number of warm-up iterations before timing
    pub num_warmup: usize,
    /// Number of iterations to measure
    pub num_iterations: usize,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            num_warmup: 10,
            num_iterations: 100,
        }
    }
}

/// Results from timing measurements
#[derive(Debug)]
pub struct TimingResults {
    /// Individual timing measurements
    pub measurements: Vec<Duration>,
    /// Mean duration
    pub mean: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum duration
    pub min: Duration,
    /// Maximum duration
    pub max: Duration,
}

impl TimingResults {
    fn new(measurements: Vec<Duration>) -> Self {
        let len = measurements.len() as f64;
        let sum: Duration = measurements.iter().sum();
        let mean = sum / measurements.len() as u32;

        let variance_sum: f64 = measurements
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum();
        let std_dev = Duration::from_secs_f64((variance_sum / len).sqrt());

        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();

        Self {
            measurements,
            mean,
            std_dev,
            min,
            max,
        }
    }
}

/// Measures inference time for any model that can process an input
pub trait InferenceMeasurable<B: Backend, Input, Output> {
    /// Run a single inference pass
    fn infer(&self, input: &Input) -> Output;
}

/// Measure inference time with proper GPU synchronization
pub fn measure_inference_time<B, M, I, O>(
    model: &M,
    input: &I,
    device: &B::Device,
    config: TimingConfig,
) -> TimingResults 
where
    B: Backend,
    M: InferenceMeasurable<B, I, O>,
{
    // Warm-up phase
    for _ in 0..config.num_warmup {
        let _ = model.infer(input);
        B::sync(device);
    }
    
    // Measurement phase
    let mut timings = Vec::with_capacity(config.num_iterations);
    
    for _ in 0..config.num_iterations {
        let start = Instant::now();
        let _ = model.infer(input);
        B::sync(device);
        timings.push(start.elapsed());
    }
    
    TimingResults::new(timings)
}

/// Helper macro to implement InferenceMeasurable for a type with a forward method
#[macro_export]
macro_rules! impl_inference_measurable {
    ($type:ty, $input:ty, $output:ty) => {
        impl<B: Backend> InferenceMeasurable<B, $input, $output> for $type {
            fn infer(&self, input: &$input) -> $output {
                self.forward(input)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::Tensor;

    struct DummyModel;
    
    impl<B: Backend> InferenceMeasurable<B, Tensor<B, 2>, Tensor<B, 2>> for DummyModel {
        fn infer(&self, input: &Tensor<B, 2>) -> Tensor<B, 2> {
            input.clone()
        }
    }

    #[test]
    fn test_timing_measurement() {
        let device = NdArrayDevice::Cpu;
        let model = DummyModel;
        let input: Tensor<NdArray, 2> = Tensor::zeros([1, 10]).to_device(&device);
        
        let config = TimingConfig {
            num_warmup: 2,
            num_iterations: 5,
        };
        
        let results = measure_inference_time::<NdArray, _, _, _>(&model, &input, &device, config);
        
        assert_eq!(results.measurements.len(), 5);
        assert!(results.mean > Duration::from_nanos(0));
        assert!(results.std_dev >= Duration::from_nanos(0));
    }
} 