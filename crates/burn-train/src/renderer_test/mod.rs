#[cfg(feature = "tui")]
use std::io::IsTerminal;

mod base;
pub use base::*;

mod cli;

#[cfg(feature = "tui")]
mod tui;
use crate::TrainingInterrupter;

/// Return the default metrics renderer.
///
/// This can be either:
///   - `TuiMetricsRenderer`, when the `tui` feature is enabled and `stdout` is
///     a terminal, or
///   - `CliMetricsRenderer`, when the `tui` feature is not enabled, or `stdout`
///     is not a terminal.
#[allow(unused_variables)]
pub(crate) fn default_renderer(
    interuptor: TrainingInterrupter,
    checkpoint: Option<usize>,
) -> Box<dyn MetricsRenderer> {
    #[cfg(feature = "tui")]
    if std::io::stdout().is_terminal() {
        return Box::new(tui::TuiMetricsRenderer::new(interuptor, checkpoint));
    }

    Box::new(cli::CliMetricsRenderer::new())
}
