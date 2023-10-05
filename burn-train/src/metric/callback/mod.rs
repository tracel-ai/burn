mod base;

pub use base::*;

#[cfg(not(feature = "tui"))]
mod cli;
#[cfg(not(feature = "tui"))]
pub use cli::CliMetricsRenderer as SelectedMetricsRenderer;

#[cfg(feature = "tui")]
mod tui;
use crate::TrainingInterrupter;
#[cfg(feature = "tui")]
pub use tui::TuiMetricsRenderer as SelectedMetricsRenderer;

/// The TUI renderer, or a simple stub if the tui feature is not enabled.
#[allow(unused_variables)]
pub(crate) fn default_renderer(
    interuptor: TrainingInterrupter,
    checkpoint: Option<usize>,
) -> SelectedMetricsRenderer {
    #[cfg(feature = "tui")]
    return SelectedMetricsRenderer::new(interuptor, checkpoint);

    #[cfg(not(feature = "tui"))]
    return SelectedMetricsRenderer::new();
}
