use clap::Parser;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Compare two CSV outputs of benches and fail if regression exceeds threshold.
#[derive(Parser, Debug)]
#[command(name = "bench-compare")]
struct Args {
    /// Baseline CSV path (default: env BENCH_BASELINE_CSV or crates/gaba-native-kernels/benches/baseline.csv)
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Current CSV path (default: env BENCH_CURRENT_CSV or target/bench-csv/gaba-native-kernels.csv)
    #[arg(long)]
    current: Option<PathBuf>,

    /// Allowed regression percent (default: env BENCH_MAX_REGRESSION_PCT or 5.0)
    #[arg(long)]
    max_regression_pct: Option<f64>,

    /// Write JSON summary to this path
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn read_csv(path: &PathBuf) -> anyhow::Result<HashMap<String, f64>> {
    let f = File::open(path)?;
    let rdr = BufReader::new(f);
    let mut map = HashMap::new();
    for (i, line) in rdr.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; } // skip header
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 2 { continue; }
        let name = parts[0].trim().to_string();
        let val: f64 = parts[1].trim().parse()?;
        map.insert(name, val);
    }
    Ok(map)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_csv_simple() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "benchmark,median_s").unwrap();
        writeln!(tmp, "bmk1, 0.123").unwrap();
        writeln!(tmp, "bmk2, 0.456").unwrap();
        let path = tmp.path().to_path_buf();
        let map = read_csv(&path).unwrap();
        assert_eq!(map.get("bmk1").unwrap(), &0.123);
        assert_eq!(map.get("bmk2").unwrap(), &0.456);
    }
}

fn main() -> anyhow::Result<()> {
    // Parse CLI args (with env-var fallbacks for compatibility)
    let args = Args::parse();

    let baseline = args
        .baseline
        .or_else(|| env::var("BENCH_BASELINE_CSV").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("crates/gaba-native-kernels/benches/baseline.csv"));

    let current = args
        .current
        .or_else(|| env::var("BENCH_CURRENT_CSV").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/bench-csv/gaba-native-kernels.csv"));

    let threshold: f64 = args
        .max_regression_pct
        .or_else(|| env::var("BENCH_MAX_REGRESSION_PCT").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(5.0);

    let json_out = args
        .json_out
        .or_else(|| env::var("BENCH_COMPARE_JSON").ok().map(PathBuf::from));

    let baseline_map = read_csv(&baseline)?;
    let current_map = read_csv(&current)?;

    let mut failed = false;
    let mut results = Vec::new();

    for (name, base_val) in &baseline_map {
        if let Some(curr_val) = current_map.get(name) {
            let pct = ((*curr_val - *base_val) / *base_val) * 100.0;
            println!("{}: baseline={}s current={}s change={:.2}%", name, base_val, curr_val, pct);
            let pass = pct <= threshold;
            if !pass {
                println!("REGRESSION: {} changed by >{:.2}% ({:.2}%)", name, threshold, pct);
                failed = true;
            }
            results.push((name.clone(), *base_val, *curr_val, pct, pass));
        } else {
            println!("Missing current result for {}", name);
            failed = true;
            results.push((name.clone(), *base_val, std::f64::NAN, std::f64::NAN, false));
        }
    }

    if let Some(path) = json_out {
        let mut f = File::create(path)?;
        writeln!(f, "{{")?;
        writeln!(f, "  \"results\": [")?;
        for (i, (name, base, cur, pct, pass)) in results.iter().enumerate() {
            writeln!(
                f,
                "    {{\"name\":\"{}\", \"baseline\":{}, \"current\":{}, \"change_pct\":{}, \"pass\":{} }}{}",
                name,
                base,
                cur,
                pct,
                pass,
                if i + 1 == results.len() { "" } else { "," }
            )?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "}}")?;
    }

    if failed {
        println!("Bench compare FAILED");
        std::process::exit(2);
    }

    println!("Bench compare OK");
    Ok(())
}
