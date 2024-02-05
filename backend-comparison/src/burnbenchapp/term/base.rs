use crate::burnbenchapp::{run_cargo, Application, BackendValues, BenchmarkValues};

use derive_new::new;

#[derive(new)]
pub struct TermApplication;

impl Application for TermApplication {
    fn init(&mut self) {}

    fn run(&mut self, benches: &[BenchmarkValues], backends: &[BackendValues]) {
        // Iterate over each combination of backend and bench
        for backend in backends.iter() {
            for bench in benches.iter() {
                run_cargo(
                    "bench",
                    &[
                        "--bench",
                        &bench.to_string(),
                        "--features",
                        &backend.to_string(),
                    ],
                );
            }
        }
    }

    fn cleanup(&mut self) {}
}
