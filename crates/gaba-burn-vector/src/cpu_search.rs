use rayon::prelude::*;

use crate::{SearchResult, VectorMetadata};

pub struct CpuSearchEngine {
    min_score_threshold: f32,
}

impl CpuSearchEngine {
    pub fn new() -> Self {
        Self { min_score_threshold: 0.1 }
    }

    pub fn with_threshold(threshold: f32) -> Self {
        Self { min_score_threshold: threshold }
    }

    /// Parallel search over vectors (id, vector, metadata)
    pub fn search_parallel(
        &self,
        query_vector: &[f32],
        vectors: &[(String, Vec<f32>, VectorMetadata)],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<_> = vectors
            .par_iter()
            .filter_map(|(id, vec, metadata)| {
                let score = cosine_similarity(query_vector, vec);
                if score >= self.min_score_threshold {
                    Some(SearchResult { id: id.clone(), score, metadata: metadata.clone() })
                } else {
                    None
                }
            })
            .collect();

        results.par_sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Simple reranker that boosts results containing query terms in metadata
    pub fn rerank(&self, results: &mut [SearchResult], query: &str) {
        for result in results.iter_mut() {
            let content_lower = result.metadata.content.to_lowercase();
            let query_lower = query.to_lowercase();

            let mut boost = 0.0;
            for word in query_lower.split_whitespace() {
                if content_lower.contains(word) {
                    boost += 0.05;
                }
            }

            if content_lower.contains(&query_lower) {
                boost += 0.15;
            }

            result.score = (result.score + boost).min(1.0);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}
