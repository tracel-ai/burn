use super::progressbar::RunnerProgressBar;
use std::io::{self, BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// Processor for standard output of cargo process
pub trait OutputProcessor: Send + Sync + 'static {
    /// Process a line
    fn process_line(&self, line: &str);
    /// To be called to indicate progress has been made
    fn progress(&self);
    /// To be called went the processor has finished processing
    fn finish(&self);
}

/// A processor that does nothing except printing the output lines as is.
#[derive(Default)]
pub struct VerboseProcessor;

impl OutputProcessor for VerboseProcessor {
    fn process_line(&self, line: &str) {
        println!("{}", line);
    }
    fn progress(&self) {}
    fn finish(&self) {}
}

/// A processor that just send the output into oblivion.
#[derive(Default)]
pub struct SinkProcessor;

impl OutputProcessor for SinkProcessor {
    fn process_line(&self, _line: &str) {}
    fn progress(&self) {}
    fn finish(&self) {}
}

/// A processor for a nice and compact output experience using a progress bar
pub struct NiceProcessor {
    bench: String,
    backend: String,
    pb: Arc<Mutex<RunnerProgressBar>>,
}

pub(crate) enum NiceProcessorState {
    Default,
    Compiling,
    Running,
    Uploading,
}

impl NiceProcessor {
    pub fn new(bench: String, backend: String, pb: Arc<Mutex<RunnerProgressBar>>) -> Self {
        Self { bench, backend, pb }
    }

    pub fn format_pb_message(&self, state: NiceProcessorState) -> String {
        match state {
            NiceProcessorState::Default | NiceProcessorState::Compiling => {
                format!("🔨 {} ▶ {}", self.bench, self.backend)
            }
            NiceProcessorState::Running => {
                format!("🔥 {} ▶ {}", self.bench, self.backend)
            }
            NiceProcessorState::Uploading => {
                format!("💾 {} ▶ {}", self.bench, self.backend)
            }
        }
    }
}

impl OutputProcessor for NiceProcessor {
    fn process_line(&self, line: &str) {
        let pb = self.pb.lock().unwrap();
        let state = if line.contains("Compiling") {
            pb.stop_spinner();
            NiceProcessorState::Compiling
        } else if line.contains("Running") {
            pb.stop_spinner();
            NiceProcessorState::Running
        } else if line.contains("Sharing") {
            pb.start_spinner();
            NiceProcessorState::Uploading
        } else {
            NiceProcessorState::Default
        };
        pb.message(self.format_pb_message(state));
    }

    fn progress(&self) {
        self.pb.lock().unwrap().advance_spinner();
    }

    fn finish(&self) {
        self.pb.lock().unwrap().inc_by_one();
    }
}

/// Benchmark runner using cargo bench.
pub struct CargoRunner<'a> {
    params: &'a [&'a str],
    processor: Arc<dyn OutputProcessor>,
}

impl<'a> CargoRunner<'a> {
    pub fn new(params: &'a [&'a str], processor: Arc<dyn OutputProcessor>) -> Self {
        Self { params, processor }
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
        let stdout_processor = Arc::clone(&self.processor);
        let stdout_thread = thread::spawn(move || {
            for line in stdout.lines() {
                let line = line.expect("A line from stdout should be read");
                stdout_processor.process_line(&line);
                stdout_processor.progress();
            }
        });
        // stderr
        let stderr = BufReader::new(cargo.stderr.take().expect("stderr should be captured"));
        let stderr_processor = Arc::clone(&self.processor);
        let stderr_thread = thread::spawn(move || {
            for line in stderr.lines() {
                let line = line.expect("A line from stderr should be read");
                stderr_processor.process_line(&line);
                stderr_processor.progress();
            }
        });
        // wait for process completion
        stdout_thread
            .join()
            .expect("The stderr thread should not panic");
        stderr_thread
            .join()
            .expect("The stderr thread should not panic");
        self.processor.finish();
        cargo.wait()
    }
}
