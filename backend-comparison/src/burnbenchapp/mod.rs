mod auth;
mod base;
mod progressbar;
mod reports;
mod runner;

pub use base::*;

const BENCHMARKS_TARGET_DIR: &str = "target/benchmarks";
const USER_BENCHMARK_SERVER_URL: &str = if cfg!(debug_assertions) {
    // development
    "http://localhost:8000/"
} else {
    // production
    "https://user-benchmark-server-gvtbw64teq-nn.a.run.app/"
};
