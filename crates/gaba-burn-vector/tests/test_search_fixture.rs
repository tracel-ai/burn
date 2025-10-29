use gaba_burn_vector::{CpuSearchEngine, VectorMetadata};
use std::fs;

#[test]
fn test_search_with_fixture() {
    let data = fs::read_to_string("tests/fixtures/vectors_200.json").expect("fixture missing");
    let parsed: serde_json::Value = serde_json::from_str(&data).expect("invalid json");
    let mut vectors = Vec::new();
    for item in parsed.as_array().unwrap() {
        let id = item["id"].as_str().unwrap().to_string();
        let content = item["content"].as_str().unwrap().to_string();
    let vec: Vec<f32> = item["vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let meta = VectorMetadata { title: id.clone(), source_path: id.clone(), content };
        vectors.push((id, vec, meta));
    }

    let dim = vectors[0].1.len();
    let query = vec![0.1f32; dim];
    let engine = CpuSearchEngine::new();

    let results = engine.search_parallel(&query, &vectors, 5);
    assert!(results.len() <= 5);
}
