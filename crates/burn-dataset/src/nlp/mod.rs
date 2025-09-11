#[cfg(feature = "builtin-sources")]
mod ag_news;
#[cfg(feature = "builtin-sources")]
mod fdu_clf;
mod text_forlder;

#[cfg(feature = "builtin-sources")]
pub use ag_news::*;
#[cfg(feature = "builtin-sources")]
pub use fdu_clf::*;
pub use text_forlder::*;
