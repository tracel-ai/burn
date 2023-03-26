#[macro_use]
extern crate derive_new;

mod data;
mod model;

pub mod inference;
pub mod training;

pub use data::{AgNewsDataset, DbPediaDataset, TextClassificationDataset};
