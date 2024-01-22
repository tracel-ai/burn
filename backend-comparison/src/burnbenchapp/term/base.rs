use crate::burnbenchapp::{Application, execute_cargo_bench, BenchmarkValues, BackendValues};

use derive_new::new;

#[derive(new)]
pub struct TermApplication;

impl Application for TermApplication {
    fn init(&mut self) {}

    fn run(&mut self, benches: &Vec<BenchmarkValues>, backends: &Vec<BackendValues>) {
        // Iterate over each combination of backend and bench
        for backend in backends.iter() {
            for bench in benches.iter() {
                execute_cargo_bench(&bench.to_string(), &backend.to_string()).unwrap();
            }
        }
    }

    fn cleanup(&mut self) {}
}
