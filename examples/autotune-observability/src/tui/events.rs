use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode};
use ratatui::{Terminal, backend::Backend};

use crate::DTYPE_NAMES;
use crate::run_support::RunMsg;
use crate::run_support::{BACKENDS, ProblemKind};

use super::app::App;
use super::ui::ui;

pub fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()>
where
    io::Error: From<<B as Backend>::Error>,
{
    let tick_rate = Duration::from_millis(50);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => {
                        if let Some(selected) = app.run_list_state.selected() {
                            if selected < app.runs.len().saturating_sub(1) {
                                app.run_list_state.select(Some(selected + 1));
                            }
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if let Some(selected) = app.run_list_state.selected() {
                            if selected > 0 {
                                app.run_list_state.select(Some(selected - 1));
                            }
                        }
                    }
                    KeyCode::Char('b') => app.backend_idx = (app.backend_idx + 1) % BACKENDS.len(),
                    KeyCode::Char('p') => {
                        app.problem_idx = (app.problem_idx + 1) % ProblemKind::ALL.len()
                    }
                    KeyCode::Char('i') => {
                        app.in_dtype_idx = (app.in_dtype_idx + 1) % DTYPE_NAMES.len()
                    }
                    KeyCode::Char('o') => {
                        app.out_dtype_idx = (app.out_dtype_idx + 1) % DTYPE_NAMES.len()
                    }
                    KeyCode::Enter | KeyCode::Char('r') => app.run_selected_problem(),
                    KeyCode::Char('D') => {
                        if let Some(selected) = app.run_list_state.selected() {
                            if let Some(run) = app.runs.get(selected) {
                                let _ = std::fs::remove_dir_all(&run.dir);
                                app.rescan_runs(None);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }

        if let Some(rx) = &app.run_rx {
            let mut done = None;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    RunMsg::Line(l) => app.output_lines.push(l),
                    RunMsg::Done { ok } => done = Some(ok),
                }
            }
            if let Some(ok) = done {
                app.run_rx = None;
                if ok {
                    let id = app.pending_run.take();
                    app.rescan_runs(id.as_deref());
                }
            }
        }
    }
}
