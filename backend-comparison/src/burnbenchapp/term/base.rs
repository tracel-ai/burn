use crate::burnbenchapp::{
    run_backend_comparison_benchmarks, Application, BackendValues, BenchmarkValues,
};

use derive_new::new;

#[derive(new)]
pub struct TermApplication;

impl Application for TermApplication {
    fn init(&mut self) {}

    fn run(
        &mut self,
        benches: &[BenchmarkValues],
        backends: &[BackendValues],
        token: Option<&str>,
    ) {
        run_backend_comparison_benchmarks(benches, backends, token)
    }

    fn cleanup(&mut self) {}
}
