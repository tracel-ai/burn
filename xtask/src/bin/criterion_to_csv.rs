use clap::Parser;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Convert Criterion's JSON outputs into a CSV with strict extraction of ids, medians and CI.
#[derive(Parser, Debug)]
#[command(name = "criterion-to-csv")]
struct Args {
    /// Base directory to search (default: target/criterion)
    #[arg(long)]
    base: Option<String>,

    /// Output CSV file (default: target/bench-csv/criterion_converted.csv)
    #[arg(long)]
    out: Option<String>,
}

// scans base for estimates.json files
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
/// Try to extract id, median, and confidence interval (lower/upper) from the Criterion JSON.
/// This function follows the Criterion JSON conventions and searches for keys `id` and
/// either `median` or `mean`. If found, returns (name, median, lower_ci, upper_ci, id).
fn extract_estimates(path: &Path) -> Option<(String, f64, Option<f64>, Option<f64>, Option<String>)> {
    let s = fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;

    // Walk recursively to find an object containing `median` or `mean`.
    fn walk(value: &Value) -> Option<(&Value, Option<String>)> {
        match value {
            Value::Object(map) => {
                // If this object has 'median' or 'mean', return it
                if map.contains_key("median") || map.contains_key("mean") {
                    // try to extract an id from same object or its parents is not trivial here,
                    // but try to read 'id' if present.
                    let id = map.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                    return Some((value, id));
                }
                for (_k, v) in map.iter() {
                    if let Some((obj, id)) = walk(v) { return Some((obj, id)); }
                }
                None
            }
            Value::Array(arr) => {
                for v in arr.iter() {
                    if let Some(x) = walk(v) { return Some(x); }
                }
                None
            }
            _ => None,
        }
    }

    let (obj, maybe_id) = walk(&v)?;
    if let Value::Object(map) = obj {
        // median preferred over mean
        let median = map
            .get("median")
            .and_then(|n| n.as_f64())
            .or_else(|| map.get("mean").and_then(|n| n.as_f64()))?;

        // try to extract confidence interval under 'median' or 'estimates' structures
        let (lower, upper) = map
            .get("median")
            .and_then(|m| m.get("confidence_interval"))
            .and_then(|ci| {
                let lower = ci.get("lower").and_then(|n| n.as_f64());
                let upper = ci.get("upper").and_then(|n| n.as_f64());
                Some((lower, upper))
            })
            .or_else(|| {
                // alternative layout: confidence_interval at top-level
                map.get("confidence_interval").and_then(|ci| {
                    let lower = ci.get("lower").and_then(|n| n.as_f64());
                    let upper = ci.get("upper").and_then(|n| n.as_f64());
                    Some((lower, upper))
                })
            })
            .unwrap_or((None, None));

        // derive name from file stem
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        return Some((name, median, lower, upper, maybe_id));
    }
    None
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let base_str = args.base.unwrap_or_else(|| "target/criterion".to_string());
    let out_dir_str = args.out.unwrap_or_else(|| "target/bench-csv".to_string());
    let base = Path::new(&base_str);
    let out_dir = Path::new(&out_dir_str);
    fs::create_dir_all(out_dir)?;
    let out_file = out_dir.join("criterion_converted.csv");
    let mut f = fs::File::create(&out_file)?;
    writeln!(f, "benchmark,median_s,median_lower_s,median_upper_s,id")?;

    let files = find_estimate_files(base);
    for file in files {
        if let Some((name, median, lower, upper, id)) = extract_estimates(&file) {
            writeln!(
                f,
                "{} ,{:.12},{},{},{}",
                name,
                median,
                lower.map(|v| format!("{:.12}", v)).unwrap_or_else(|| "".to_string()),
                upper.map(|v| format!("{:.12}", v)).unwrap_or_else(|| "".to_string()),
                id.as_deref().unwrap_or("")
            )?;
        }
    }

    println!("Wrote criterion CSV to {}", out_file.display());
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_extract_estimates_simple() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("bmk").join("new");
        fs::create_dir_all(&p).unwrap();
        let json = r#"{
            "median": 0.123,
            "confidence_interval": { "lower": 0.11, "upper": 0.14 },
            "id": "bench::example"
        }"#;
        let file = p.join("estimates.json");
        fs::write(&file, json).unwrap();

        let res = extract_estimates(&file).unwrap();
        assert_eq!(res.1, 0.123);
        assert_eq!(res.2.unwrap(), 0.11);
        assert_eq!(res.3.unwrap(), 0.14);
        assert_eq!(res.4.unwrap(), "bench::example");
    }
}
