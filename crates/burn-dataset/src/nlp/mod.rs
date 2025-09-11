#[cfg(feature = "builtin-sources")]
mod ag_news;
#[cfg(feature = "builtin-sources")]
mod fdu_clf;
mod text_folder;

#[cfg(feature = "builtin-sources")]
pub use ag_news::*;
#[cfg(feature = "builtin-sources")]
pub use fdu_clf::*;
pub use text_folder::*;
