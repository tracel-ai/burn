use super::progressbar::RunnerProgressBar;
use std::io::{self, BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// Processor for standard output of cargo process
pub trait OutputProcessor {
    fn process_line(&self, line: &str);
    fn finish(&self);
}

/// A processor that does nothing except printing the output lines as is.
#[derive(Default)]
pub struct VerboseProcessor;

impl OutputProcessor for VerboseProcessor {
    fn process_line(&self, line: &str) {
        println!("{}", line);
    }
    fn finish(&self) {}
}

/// A processor that just send the output into oblivion.
#[derive(Default)]
pub struct SinkProcessor;

impl OutputProcessor for SinkProcessor {
    fn process_line(&self, _line: &str) {}
    fn finish(&self) {}
}

/// A processor for a nice and compact outtput experience using a progress bar
pub struct NiceProcessor {
    bench: String,
    backend: String,
    pb: Arc<Mutex<RunnerProgressBar>>,
}

pub(crate) enum NiceProcessorState {
    Compiling,
    Running,
}

impl NiceProcessor {
    pub fn new(bench: String, backend: String, pb: Arc<Mutex<RunnerProgressBar>>) -> Self {
        Self { bench, backend, pb }
    }

    pub fn format_pb_message(&self, state: NiceProcessorState) -> String {
        match state {
            NiceProcessorState::Compiling => {
                format!("🔨 Compiling: {} 🠆  {}", self.bench, self.backend)
            }
            NiceProcessorState::Running => {
                format!("🔥 Running: {} 🠆  {}", self.bench, self.backend)
            }
        }
    }
}

impl OutputProcessor for NiceProcessor {
    fn process_line(&self, line: &str) {
        let pb = self.pb.lock().unwrap();
        if line.contains("Compiling") {
            pb.message(self.format_pb_message(NiceProcessorState::Compiling));
        } else if line.contains("Running") {
            pb.message(self.format_pb_message(NiceProcessorState::Running));
        }
    }

    fn finish(&self) {
        self.pb.lock().unwrap().inc_by_one();
    }
}

/// Benchmark runner using cargo bench.
pub struct CargoRunner<'a> {
    params: &'a [&'a str],
    stdout_processor: Arc<dyn OutputProcessor + Send + Sync + 'static>,
    stderr_processor: Arc<dyn OutputProcessor + Send + Sync + 'static>,
}

impl<'a> CargoRunner<'a> {
    pub fn new(
        params: &'a [&'a str],
        stdout_processor: Arc<dyn OutputProcessor + Send + Sync + 'static>,
        stderr_processor: Arc<dyn OutputProcessor + Send + Sync + 'static>,
    ) -> Self {
        Self {
            params,
            stdout_processor,
            stderr_processor,
        }
    }

    pub fn run(&mut self) -> io::Result<ExitStatus> {
        let mut cargo = Command::new("cargo")
            .env("CARGO_TERM_COLOR", "always")
            .arg("bench")
            .args(self.params)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Cargo command should start successfully");
        // stdout
        let stdout = BufReader::new(cargo.stdout.take().expect("stdout should be captured"));
        let stdout_processor = Arc::clone(&self.stdout_processor);
        let stdout_thread = thread::spawn(move || {
            for line in stdout.lines() {
                let line = line.expect("A line from stdout should be read");
                stdout_processor.process_line(&line);
            }
        });
        // stderr
        let stderr = BufReader::new(cargo.stderr.take().expect("stderr should be captured"));
        let stderr_processor = Arc::clone(&self.stderr_processor);
        let stderr_thread = thread::spawn(move || {
            for line in stderr.lines() {
                let line = line.expect("A line from stderr should be read");
                stderr_processor.process_line(&line);
            }
        });
        // wait for process completion
        stdout_thread
            .join()
            .expect("The stderr thread should not panic");
        self.stdout_processor.finish();
        stderr_thread
            .join()
            .expect("The stderr thread should not panic");
        self.stderr_processor.finish();
        cargo.wait().map_err(|e| e.into())
    }
}
