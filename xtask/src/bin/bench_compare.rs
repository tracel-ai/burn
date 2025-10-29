use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

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

fn main() -> anyhow::Result<()> {
    // Configurable baseline/current paths and allowed regression pct via env vars
    let baseline = env::var("BENCH_BASELINE_CSV").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("crates/gaba-native-kernels/benches/baseline.csv"));
    let current = env::var("BENCH_CURRENT_CSV").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("target/bench-csv/gaba-native-kernels.csv"));

    let threshold: f64 = env::var("BENCH_MAX_REGRESSION_PCT").ok().and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let json_out = env::var("BENCH_COMPARE_JSON").ok().map(PathBuf::from);

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
        writeln!(f, "{{\"results\":[")?;
        for (i, (name, base, cur, pct, pass)) in results.iter().enumerate() {
            writeln!(f, "  {{\"name\":\"{}\", \"baseline\":{}, \"current\":{}, \"change_pct\":{}, \"pass\":{} }}{}", name, base, cur, pct, pass, if i + 1 == results.len() {""} else {","})?;
        }
    writeln!(f, "]}}")?;
    }

    if failed {
        println!("Bench compare FAILED");
        std::process::exit(2);
    }

    println!("Bench compare OK");
    Ok(())
}
