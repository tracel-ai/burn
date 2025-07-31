mod global;
pub use global::*;

mod config;
pub use config::*;

mod api;
pub use api::*;

mod local;

#[cfg(all(
    test,
    any(
        feature = "test-ndarray",
        feature = "test-wgpu",
        feature = "test-cuda",
        feature = "test-metal"
    )
))]
mod tests;
