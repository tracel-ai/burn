mod auth;
mod base;

pub use base::*;

#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
use tui::TuiApplication as App;

#[cfg(not(feature = "tui"))]
mod term;
#[cfg(not(feature = "tui"))]
use term::TermApplication as App;

const BENCHMARKS_TARGET_DIR: &str = "target/benchmarks";
const USER_BENCHMARK_SERVER_URL: &str = if cfg!(debug_assertions) {
    // development
    "http://localhost:8000/"
} else {
    // production
    "https://user-benchmark-server-gvtbw64teq-nn.a.run.app/"
};
