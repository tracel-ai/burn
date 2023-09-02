/// Command line interface module for the dashboard.
#[cfg(feature = "ui")]
mod cli;
#[cfg(not(feature = "ui"))]
mod cli_stub;

mod base;
mod plot;

pub use base::*;
pub use plot::*;

#[cfg(feature = "ui")]
pub use cli::CLIDashboardRenderer;
#[cfg(not(feature = "ui"))]
pub use cli_stub::CLIDashboardRenderer;
