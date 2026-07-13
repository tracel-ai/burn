use std::path::{Path, PathBuf};

/// Directory this example is compiled from. The `cubecl.toml` sitting there configures
/// the autotune logger, and the log file is written relative to it.
pub fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Path of the autotune log file produced by the shipped `cubecl.toml`.
pub fn log_path() -> PathBuf {
    example_dir().join("autotune.log")
}

/// Root of the persistent autotune cache (`target/autotune`).
pub fn autotune_cache_dir() -> PathBuf {
    example_dir().join("target").join("autotune")
}

/// Wipe the on-disk autotune cache and log. The `runner` process calls this before every run:
/// because each run is a fresh process, its in-memory tuner cache starts empty, and wiping the
/// on-disk cache means nothing is loaded into it — so the same config re-tunes (and re-logs)
/// every single time, which is the whole point of shelling out to a subprocess.
pub fn start_fresh_session() {
    let _ = std::fs::remove_dir_all(autotune_cache_dir());
    let _ = std::fs::remove_file(log_path());
}

/// Float dtypes selectable in the UI. The input dtype is what drives the matmul autotune key
/// (`elem_lhs`/`elem_rhs`/`elem_out`) and therefore whether tensor-core (CMMA) kernels are
/// eligible — e.g. f32 inputs rule them out, f16/bf16 make them candidates. The `runner` maps
/// these names back to `FloatDType`; keeping them as plain strings keeps this crate burn-free
/// so the UI builds without the backend.
pub const DTYPE_NAMES: [&str; 4] = ["f32", "f16", "bf16", "flex32"];

/// Outcome of a single candidate kernel within a tune event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    /// Actually benchmarked (an `Autotune[..] => BenchmarkComputations { .. }` line).
    Benchmarked,
    /// Pruned before benchmarking (`... skipped manually`).
    Skipped,
    /// Ran but produced no usable samples (`All samples are invalid`).
    Invalid,
    /// Anything else the parser didn't recognize.
    Other,
}

/// A single candidate line under a tune event.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub kind: CandidateKind,
    pub text: String,
}

/// A single autotune decision parsed from the log: the kernel that won, the tune key, the
/// planner context (the `- Tuning: [..]` groups — fewer batches here is where bound-driven
/// short-circuits show up), and the categorized per-candidate outcomes.
#[derive(Debug, Clone, Default)]
pub struct TuneEvent {
    pub fastest: String,
    pub key: String,
    pub context: String,
    pub tuning_batches: usize,
    pub candidates: Vec<Candidate>,
    pub short_circuit: bool,
}

impl TuneEvent {
    pub fn count(&self, kind: CandidateKind) -> usize {
        self.candidates.iter().filter(|c| c.kind == kind).count()
    }
}

const SHORT_CIRCUIT_HINTS: [&str; 4] = ["bound", "short", "throughput", "overhead"];

/// Parse a full-level autotune log into a list of tune events.
///
/// Each event starts at a `Fastest result <name>-<key>. Context: <context>` header line.
/// The context value (`- Tuning: [..]`) continues on following `- `-prefixed lines; every
/// other line until the next header is a candidate outcome.
pub fn parse_log(text: &str) -> Vec<TuneEvent> {
    let mut events = Vec::new();
    let mut current: Option<TuneEvent> = None;

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Fastest result ") {
            if let Some(event) = current.take() {
                events.push(finalize(event));
            }
            current = Some(parse_header(rest));
        } else if let Some(event) = current.as_mut() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(context) = line.strip_prefix("- ") {
                if !event.context.is_empty() {
                    event.context.push('\n');
                }
                event.context.push_str(line);
                if context.starts_with("Tuning:") {
                    event.tuning_batches += 1;
                }
            } else {
                event.candidates.push(classify(line));
            }
        }
    }

    if let Some(event) = current.take() {
        events.push(finalize(event));
    }

    events
}

fn parse_header(rest: &str) -> TuneEvent {
    let (name_key, context) = match rest.split_once(". Context:") {
        Some((left, right)) => (left, right.trim().to_string()),
        None => (rest, String::new()),
    };

    // `name_key` is `<kernel name>-<key>`. The kernel name has no dashes, so the first dash
    // splits the winning kernel from the (long) key display.
    let (fastest, key) = match name_key.split_once('-') {
        Some((name, key)) => (name.trim().to_string(), key.trim().to_string()),
        None => (name_key.trim().to_string(), String::new()),
    };

    TuneEvent {
        fastest,
        key,
        context,
        ..Default::default()
    }
}

fn finalize(mut event: TuneEvent) -> TuneEvent {
    let context = event.context.to_lowercase();
    event.short_circuit = SHORT_CIRCUIT_HINTS
        .iter()
        .any(|hint| context.contains(hint));
    event
}

fn classify(line: &str) -> Candidate {
    let lower = line.to_lowercase();
    let kind = if line.starts_with("Autotune[") || line.contains("BenchmarkComputations") {
        CandidateKind::Benchmarked
    } else if lower.contains("skipped") {
        CandidateKind::Skipped
    } else if lower.contains("invalid") {
        CandidateKind::Invalid
    } else {
        CandidateKind::Other
    };
    Candidate {
        kind,
        text: line.to_string(),
    }
}

/// Read and parse the log file, returning an empty list if it does not exist yet.
pub fn load_events(path: &Path) -> Vec<TuneEvent> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_log(&text),
        Err(_) => Vec::new(),
    }
}

pub mod ansi;

mod app;
pub use app::AutotuneObservabilityApp;
