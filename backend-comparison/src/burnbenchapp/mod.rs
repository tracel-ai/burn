mod base;
pub use base::*;
mod model;

#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
use tui::TuiApplication as App;

#[cfg(not(feature = "tui"))]
mod term;
#[cfg(not(feature = "tui"))]
use term::TermApplication as App;
