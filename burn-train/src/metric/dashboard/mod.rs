mod base;

pub use base::*;

#[cfg(not(feature = "tui"))]
mod cli_stub;
#[cfg(not(feature = "tui"))]
pub use cli_stub::CLIDashboardRenderer as SelectedDashboardRenderer;

#[cfg(feature = "tui")]
mod tui;
use crate::TrainingInterrupter;
#[cfg(feature = "tui")]
pub use tui::TuiDashboardRenderer as SelectedDashboardRenderer;

/// The TUI renderer, or a simple stub if the tui feature is not enabled.
#[allow(unused_variables)]
pub(crate) fn default_renderer(interuptor: TrainingInterrupter) -> SelectedDashboardRenderer {
    #[cfg(feature = "tui")]
    return SelectedDashboardRenderer::new(interuptor);

    #[cfg(not(feature = "tui"))]
    return SelectedDashboardRenderer::new();
}
