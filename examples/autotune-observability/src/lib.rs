use std::path::{Path, PathBuf};

/// Directory this example is compiled from. The `cubecl.toml` sitting there configures
/// the autotune logger, and the log file is written relative to it.
pub fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Write a per-run `cubecl.toml` inside `run_dir` pointing the autotune log and the cache it
/// generates into that directory. The runner runs with `run_dir` as its working directory, so
/// cubecl discovers this config (it wins over the crate's fallback `cubecl.toml`). Each run
/// therefore gets its own empty cache — so the same config re-tunes (and re-logs) every time —
/// and both the log and the cache are archived side by side for later reference.
pub fn write_run_config(run_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(run_dir)?;
    let log = run_dir.join("autotune.log");
    let cache = run_dir.join("cache");
    let config = format!(
        "[autotune.logger]\nlevel = \"full\"\nfile = {log:?}\n\n[autotune.cache]\nfile = {cache:?}\n"
    );
    std::fs::write(run_dir.join("cubecl.toml"), config)
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
/// planner context (the `- Tuning: [..]` groups), the throughput-bound lines emitted while
/// tuning (where the short-circuit decision is logged), and the per-candidate outcomes.
#[derive(Debug, Clone, Default)]
pub struct TuneEvent {
    pub fastest: String,
    pub key: String,
    pub context: String,
    pub tuning_batches: usize,
    pub bounds: Vec<String>,
    pub candidate_progress: Vec<f32>,
    pub candidates: Vec<Candidate>,
    pub short_circuit: Option<f32>,
}

impl TuneEvent {
    pub fn count(&self, kind: CandidateKind) -> usize {
        self.candidates.iter().filter(|c| c.kind == kind).count()
    }
}

/// Parse a full-level autotune log into a list of tune events.
///
/// The throughput-bound lines (`Calculated bounds`, `Autotune candidate … achieved …%`,
/// `Short circuiting …`) are logged *while* tuning, so they precede that autotune's
/// `Fastest result` header. They are buffered and attached to the event they lead into. After
/// the header, the context continues on `- `-prefixed lines and everything else is a candidate.
pub fn parse_log(text: &str) -> Vec<TuneEvent> {
    let mut events = Vec::new();
    let mut current: Option<TuneEvent> = None;
    let mut pending_bounds: Vec<String> = Vec::new();
    let mut pending_candidate_progress: Vec<f32> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("Fastest result ") {
            if let Some(event) = current.take() {
                events.push(finalize(event));
            }
            let mut event = parse_header(rest);
            event.bounds = std::mem::take(&mut pending_bounds);
            event.candidate_progress = std::mem::take(&mut pending_candidate_progress);
            current = Some(event);
        } else if is_bounds_line(line) {
            // Belongs to the *next* autotune, whose header hasn't been seen yet.
            pending_bounds.push(line.to_string());
            if let Some(progress) = parse_candidate_progress(line) {
                pending_candidate_progress.push(progress);
            }
        } else if let Some(event) = current.as_mut() {
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

    if let Some(mut event) = current.take() {
        // Trailing bounds with no following header still describe the last autotune's context.
        event.bounds.extend(pending_bounds);
        event.candidate_progress.extend(pending_candidate_progress);
        events.push(finalize(event));
    }

    events
}

/// Whether a line is one of the throughput-bound / short-circuit log messages.
fn is_bounds_line(line: &str) -> bool {
    line.starts_with("Calculated bounds")
        || line.starts_with("Autotune candidate")
        || line.starts_with("Short circuiting")
}

fn parse_candidate_progress(line: &str) -> Option<f32> {
    if !line.starts_with("Autotune candidate") {
        return None;
    }

    line.split_whitespace()
        .find(|word| word.ends_with('%'))
        .and_then(|word| word.trim_end_matches('%').parse::<f32>().ok())
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
    event.short_circuit = if event
        .bounds
        .iter()
        .any(|line| line.starts_with("Short circuiting"))
    {
        event
            .bounds
            .iter()
            .filter(|line| line.starts_with("Autotune candidate"))
            .last()
            .and_then(|line| line.split_whitespace().find(|word| word.ends_with('%')))
            .and_then(|s| s.trim_end_matches('%').parse::<f32>().ok())
    } else {
        None
    };

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

/// Directory holding archived runs, one sub-directory per run.
pub fn runs_dir() -> PathBuf {
    example_dir().join("runs")
}

/// Archived run directories, newest first (their names are millisecond-timestamp-prefixed).
pub fn list_runs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(runs_dir())
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs.reverse();
    dirs
}

/// Path to an archived run's log file.
pub fn run_log_path(run_dir: &Path) -> PathBuf {
    run_dir.join("autotune.log")
}

pub mod ansi;

pub mod run_support;
mod ui_components;

mod app;
pub use app::AutotuneObservabilityApp;
pub use run_support::ProblemKind;

pub mod tui;
