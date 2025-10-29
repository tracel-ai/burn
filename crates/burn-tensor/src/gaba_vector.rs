#[cfg(feature = "gaba-vector-cpu")]
pub mod gaba_vector {
    use gaba_burn_vector::CpuSearchEngine;
    use gaba_burn_vector::{SearchResult, VectorMetadata};

    pub struct GabaCpuSearch {
        engine: CpuSearchEngine,
    }

    impl GabaCpuSearch {
        pub fn new(threshold: f32) -> Self {
            Self { engine: CpuSearchEngine::with_threshold(threshold) }
        }

        pub fn search(&self, query: &[f32], vectors: &[(String, Vec<f32>, VectorMetadata)], top_k: usize) -> Vec<SearchResult> {
            self.engine.search_parallel(query, vectors, top_k)
        }

        pub fn rerank(&self, results: &mut [SearchResult], query: &str) {
            self.engine.rerank(results, query)
        }
    }
}
