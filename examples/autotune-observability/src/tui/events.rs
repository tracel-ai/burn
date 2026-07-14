use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode};
use ratatui::{Terminal, backend::Backend};

use crate::DTYPE_NAMES;
use crate::run_support::RunMsg;
use crate::run_support::{BACKENDS, ProblemKind};

use super::app::{App, RemoteField};
use super::ui::ui;

pub(crate) fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()>
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
                if app.show_help {
                    // Any key dismisses the help overlay.
                    app.show_help = false;
                } else if app.editing_book_name {
                    let idx = app.active_book_index();
                    match key.code {
                        KeyCode::Enter | KeyCode::Esc => {
                            app.editing_book_name = false;
                            app.run_books.save();
                        }
                        KeyCode::Backspace => {
                            app.run_books.books[idx].name.pop();
                        }
                        KeyCode::Char(c) => {
                            app.run_books.books[idx].name.push(c);
                        }
                        _ => {}
                    }
                } else if let Some(field) = app.remote_edit {
                    match key.code {
                        KeyCode::Enter | KeyCode::Esc => {
                            app.remote_edit = None;
                            app.remote.save();
                        }
                        KeyCode::Backspace => {
                            app.remote_field_mut(field).pop();
                        }
                        KeyCode::Char(c) => {
                            app.remote_field_mut(field).push(c);
                        }
                        _ => {}
                    }
                } else if app.input_mode {
                    match key.code {
                        KeyCode::Enter | KeyCode::Esc => app.input_mode = false,
                        KeyCode::Backspace => {
                            app.shape_dims[app.active_dim_idx].pop();
                        }
                        KeyCode::Char(c) if c.is_ascii_digit() => {
                            app.shape_dims[app.active_dim_idx].push(c);
                        }
                        KeyCode::Left => {
                            if app.active_dim_idx > 0 {
                                app.active_dim_idx -= 1;
                            }
                        }
                        KeyCode::Right => {
                            if app.active_dim_idx + 1 < app.shape_dims.len() {
                                app.active_dim_idx += 1;
                            }
                        }
                        KeyCode::Tab => {
                            app.active_dim_idx = (app.active_dim_idx + 1) % app.shape_dims.len();
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Char('?') => app.show_help = true,
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::PageDown => {
                            app.events_scroll = app.events_scroll.saturating_add(10)
                        }
                        KeyCode::PageUp => app.events_scroll = app.events_scroll.saturating_sub(10),
                        KeyCode::Home => app.events_scroll = 0,
                        KeyCode::End => app.events_scroll = u16::MAX,
                        KeyCode::Down | KeyCode::Char('j') => {
                            if let Some(selected) = app.run_list_state.selected() {
                                if selected < app.runs.len().saturating_sub(1) {
                                    app.run_list_state.select(Some(selected + 1));
                                    app.events_scroll = 0;
                                }
                            }
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            if let Some(selected) = app.run_list_state.selected() {
                                if selected > 0 {
                                    app.run_list_state.select(Some(selected - 1));
                                    app.events_scroll = 0;
                                }
                            }
                        }
                        KeyCode::Char('b') => {
                            app.backend_idx = (app.backend_idx + 1) % BACKENDS.len()
                        }
                        KeyCode::Char('p') => {
                            app.problem_idx = (app.problem_idx + 1) % ProblemKind::ALL.len();
                            app.shape_dims = ProblemKind::ALL[app.problem_idx]
                                .default_shape()
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect();
                            app.active_dim_idx = 0;
                        }
                        KeyCode::Char('i') => {
                            app.in_dtype_idx = (app.in_dtype_idx + 1) % DTYPE_NAMES.len()
                        }
                        KeyCode::Char('o') => {
                            app.out_dtype_idx = (app.out_dtype_idx + 1) % DTYPE_NAMES.len()
                        }
                        KeyCode::Char('s') => app.input_mode = true,
                        KeyCode::Char('c') => app.cancel_run(),
                        KeyCode::Char('m') => {
                            app.remote.enabled = !app.remote.enabled;
                            app.remote.save();
                        }
                        KeyCode::Char('h') => app.remote_edit = Some(RemoteField::Host),
                        KeyCode::Char('g') => app.remote_edit = Some(RemoteField::Base),
                        KeyCode::Char('w') => app.remote_edit = Some(RemoteField::Password),
                        KeyCode::Char('f') if app.remote.enabled => {
                            app.force_sync = !app.force_sync
                        }
                        KeyCode::Char('t') => {
                            app.disable_throughput_cache = !app.disable_throughput_cache
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
                        // --- Run books ---
                        KeyCode::Char('[') => app.select_prev_book(),
                        KeyCode::Char(']') => app.select_next_book(),
                        KeyCode::Char('N') => app.new_book(),
                        KeyCode::Char('X') => app.delete_book(),
                        KeyCode::Char('e') => app.editing_book_name = true,
                        KeyCode::Char('B') => app.cycle_book_backend(),
                        KeyCode::Char('a') => app.book_add_current(),
                        KeyCode::Char('J') => app.book_entry_next(),
                        KeyCode::Char('K') => app.book_entry_prev(),
                        KeyCode::Char('x') => app.book_delete_selected_entry(),
                        KeyCode::Char('V') => app.cycle_selected_entry_backend(),
                        KeyCode::Char('R') => app.book_run_selected_entry(),
                        KeyCode::Char('A') => app.book_run_all(),
                        _ => {}
                    }
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
                    RunMsg::Progress(p) => app.status = p,
                    RunMsg::Done { ok } => done = Some(ok),
                }
            }
            if let Some(ok) = done {
                app.run_rx = None;
                if ok {
                    let id = app.pending_run.take();
                    app.rescan_runs(id.as_deref());
                }
                app.start_next_queued();
            }
        }
    }
}
