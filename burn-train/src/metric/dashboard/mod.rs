mod base;

pub use base::*;

#[cfg(not(feature = "tui"))]
mod cli_stub;
#[cfg(not(feature = "tui"))]
pub use cli_stub::CLIDashboardRenderer as SelectedDashboardRenderer;

#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
pub use tui::TuiDashboardRenderer as SelectedDashboardRenderer;
