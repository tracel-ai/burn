

enum Backends {
    WgpuFusion,
    Wgpu,
    TchGpu,
    TchCpu,
    Ndarray,
    NdarrayBlasNetLib,
    NdarrayBlasOpenblas,
    NdarrayBlasAccelerate,
    
}

enum Benches {
    Unary,
    Binary,
    MatMul,
    Data,
    CustomGelu,
}

enum AppState {
    Running,
    Stopped,
}

struct Result {
    
}

#[derive(new)]
struct Model {
    backends: Vec<String>,
    benchmarks: Vec<String>,
    selected_backends: Vec<String>,
    selected_benchmarks: Vec<String>,
    state: AppState,
    results: String,
    completed_benches: u8,
}


impl Model {
    pub fn benches_to_run_count(&self) -> usize {
        self.selected_backends.len() * self.selected_benchmarks.len()

    }

    pub fn bench_complete(&mut self) {
       self.completed_benches += 1;
    }
}
