/// Command line interface module for the dashboard.
#[cfg(feature = "cli")]
mod cli;
#[cfg(not(feature = "cli"))]
mod cli_stub;

mod base;
mod plot;

pub use base::*;
pub use plot::*;

#[cfg(feature = "cli")]
pub use cli::CLIDashboardRenderer;
#[cfg(not(feature = "cli"))]
pub use cli_stub::CLIDashboardRenderer;
