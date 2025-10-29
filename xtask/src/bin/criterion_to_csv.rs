use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

// Very small heuristic converter: scans target/criterion/* for estimates.json files and
// extracts means into a CSV with header `benchmark,median_s`.
fn find_estimate_files(base: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(base) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let est = p.join("new").join("estimates.json");
                if est.exists() {
                    out.push(est);
                } else {
                    // fallback: any json under the directory
                    if let Ok(sub) = fs::read_dir(&p) {
                        for s in sub.flatten() {
                            let q = s.path();
                            if q.extension().map(|s| s == "json").unwrap_or(false) {
                                out.push(q);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn extract_mean(path: &Path) -> Option<(String, f64)> {
    let s = fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;
    // Heuristic: try to find a name and mean inside JSON. Many Criterion outputs store
    // estimates.mean. We'll walk the JSON to find a numeric `mean` field.
    fn walk(value: &Value) -> Option<f64> {
        match value {
            Value::Object(map) => {
                if let Some(Value::Number(n)) = map.get("mean") {
                    return n.as_f64();
                }
                for (_k, v) in map.iter() {
                    if let Some(f) = walk(v) { return Some(f); }
                }
                None
            }
            Value::Array(arr) => {
                for v in arr.iter() {
                    if let Some(f) = walk(v) { return Some(f); }
                }
                None
            }
            _ => None,
        }
    }

    let mean = walk(&v)?;
    // derive a name from the path
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
    Some((name, mean))
}

fn main() -> anyhow::Result<()> {
    let base = Path::new("target/criterion");
    let out_dir = Path::new("target/bench-csv");
    fs::create_dir_all(out_dir)?;
    let out_file = out_dir.join("criterion_converted.csv");
    let mut f = fs::File::create(&out_file)?;
    writeln!(f, "benchmark,median_s")?;

    let files = find_estimate_files(base);
    for file in files {
        if let Some((name, mean)) = extract_mean(&file) {
            writeln!(f, "{} ,{}", name, mean)?;
        }
    }

    println!("Wrote criterion CSV to {}", out_file.display());
    Ok(())
}
