//! Gaba Burn Vector utilities
//!
//! Provides a CPU search engine (Rayon-parallel) and simple types for vector stores.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub title: String,
    pub source_path: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: VectorMetadata,
}

pub mod cpu_search;

pub use cpu_search::CpuSearchEngine;
