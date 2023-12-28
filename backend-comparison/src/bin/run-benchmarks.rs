use clap::Parser;
use std::io::{self, Stdout, Write};
use std::process::{Child, Command, Stdio};

#[cfg(feature = "tui")]
use ratatui::{
    backend::CrosstermBackend,
    widgets::{Block, Borders, Gauge},
    Terminal,
};

#[cfg(feature = "tui")]
use crossterm::{
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};

// CLI  arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Comma-separated list of backends to include
    #[clap(long, value_name = "BACKENDS")]
    backends: Option<String>,

    /// Comma-separated list of benches to run
    #[clap(long, value_name = "BENCHES")]
    benches: Option<String>,
}

fn execute_cargo_bench(backend: &str, bench: &str) -> io::Result<()> {
    let mut cargo = Command::new("cargo");
    cargo.arg("bench").arg("--bench").arg(bench);
    if !backend.is_empty() {
        cargo.args(&["--features", backend]);
    }
    let output = cargo.output().expect("cargo bench executed successfully");
    // println!("{}", output.status);
    // io::stdout().write_all(&output.stdout).unwrap();
    // io::stderr().write_all(&output.stderr).unwrap();
    Ok(())
}

fn execute_bench(bench: &String, backend: &String, terminal_ui: bool) {
    let mut command = Command::new("cargo");
    command.arg("bench").arg("--bench").arg(bench);

    if !backend.is_empty() {
        command.args(&["--features", backend]);
    }

    if !terminal_ui {
        println!(
            "Running: cargo bench --bench {} --features {}",
            bench, backend
        );
    }
    // Execute the command
    let status = command.status();
    if !terminal_ui {
        match status {
            Ok(status) if status.success() => {
                println!("Benchmark {} with backend {} complete.", bench, backend)
            }
            Ok(status) => println!(
                "Error! Benchmark {} with backend {} exited with code {}",
                bench, backend, status
            ),
            Err(e) => println!(
                "Error! Failed to run benchmark {} with backend {}: {}",
                bench, backend, e
            ),
        }
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let backends = args
        .backends
        .unwrap_or_default()
        .split(',')
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let benches = args
        .benches
        .unwrap_or_default()
        .split(',')
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    #[cfg(feature = "tui")]
    {
        let mut terminal = setup_terminal()?;
        let total_combinations = backends.len() * benches.len();
        run_tui(&mut terminal, &backends, &benches, total_combinations)?;
        teardown_terminal(terminal)?;
    }

    #[cfg(not(feature = "tui"))]
    {
        // Iterate over each combination of backend and bench
        for backend in backends.iter() {
            for bench in benches.iter() {
                execute_cargo_bench(backend, bench);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "tui")]
fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    terminal::enable_raw_mode()?;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

#[cfg(feature = "tui")]
fn run_tui(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    backends: &[String],
    benches: &[String],
    total_combinations: usize,
) -> io::Result<()> {
    let mut progress = 0;
    for backend in backends {
        for bench in benches {
            execute_cargo_bench(backend, bench);
            progress += 1;
            // Update progress bar for each combination
            terminal.draw(|f| {
                let size = f.size();
                let percent = progress as f64 / total_combinations as f64 * 100.0;
                let gauge = Gauge::default()
                    .block(Block::default().title("Progress").borders(Borders::ALL))
                    .percent(percent as u16);
                f.render_widget(gauge, size);
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "tui")]
fn teardown_terminal(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    terminal::disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
