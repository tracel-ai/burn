#[macro_use]
extern crate derive_new;

pub mod data;
pub mod inference;
pub mod model;
pub mod training;

pub use data::{AgNewsDataset, DbPediaDataset, TextClassificationDataset};
