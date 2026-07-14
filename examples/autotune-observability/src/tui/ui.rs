use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
};

use crate::DTYPE_NAMES;
use crate::ansi::{AnsiStyle, parse_ansi};
use crate::run_support::{BACKENDS, ProblemKind};

use super::app::{App, RemoteField};

fn into_ratatui_style(ansi: AnsiStyle) -> Style {
    let mut style = Style::default();
    if ansi.bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    if let Some(c) = ansi.color {
        let color = match (c, ansi.bright) {
            (0, false) => Color::Black,
            (1, false) => Color::Red,
            (2, false) => Color::Green,
            (3, false) => Color::Yellow,
            (4, false) => Color::Blue,
            (5, false) => Color::Magenta,
            (6, false) => Color::Cyan,
            (7, false) => Color::White,
            (0, true) => Color::DarkGray,
            (1, true) => Color::LightRed,
            (2, true) => Color::LightGreen,
            (3, true) => Color::LightYellow,
            (4, true) => Color::LightBlue,
            (5, true) => Color::LightMagenta,
            (6, true) => Color::LightCyan,
            (7, true) => Color::Gray,
            _ => Color::Reset,
        };
        style = style.fg(color);
    }
    style
}

fn render_book(f: &mut Frame, app: &mut App, area: Rect) {
    let idx = app.active_book_index();
    let total = app.run_books.books.len();
    let (name, backend, items): (String, String, Vec<ListItem>) = {
        let book = &app.run_books.books[idx];
        let items = book
            .specs
            .iter()
            .map(|spec| {
                let label = if spec.name.trim().is_empty() {
                    String::new()
                } else {
                    format!("{}: ", spec.name)
                };
                let override_backend = match &spec.backend {
                    Some(backend) => format!("  [{backend}]"),
                    None => String::new(),
                };
                ListItem::new(format!(
                    "{label}{} {}→{} {}{override_backend}",
                    spec.problem.name(),
                    spec.input,
                    spec.output,
                    spec.shapes_string(),
                ))
            })
            .collect();
        (book.name.clone(), book.backend.clone(), items)
    };

    let display_name = if name.trim().is_empty() { "(unnamed)" } else { name.as_str() };
    let title = if app.editing_book_name {
        format!(" Rename book: {name}█ (Enter/Esc) ")
    } else {
        format!(" Book {}/{}: {display_name} [{backend}] ", idx + 1, total)
    };

    let book_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)].as_ref())
        .split(area);

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");
    f.render_stateful_widget(list, book_chunks[0], &mut app.book_entry_state);

    let hint = Paragraph::new(Line::from(vec![Span::styled(
        " a:add R:run A:run-all x:del J/K:move V:ovr · N/X/e/B book · [/]:switch",
        Style::default().fg(Color::DarkGray),
    )]));
    f.render_widget(hint, book_chunks[1]);
}

pub(crate) fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)].as_ref())
        .split(f.area());

    let runs: Vec<ListItem> = app
        .runs
        .iter()
        .map(|r| {
            let label = r.custom_name.as_deref().unwrap_or(&r.name);
            ListItem::new(label.to_string())
        })
        .collect();

    let runs_list = List::new(runs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Runs (Del: 'D') "),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)].as_ref())
        .split(chunks[0]);

    f.render_stateful_widget(runs_list, left_chunks[0], &mut app.run_list_state);
    render_book(f, app, left_chunks[1]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(10)].as_ref())
        .split(chunks[1]);

    let backend_name = BACKENDS[app.backend_idx].0;
    let problem_name = ProblemKind::ALL[app.problem_idx].label();
    let in_dtype = DTYPE_NAMES[app.in_dtype_idx];
    let out_dtype = DTYPE_NAMES[app.out_dtype_idx];

    let controls_text = if let Some(field) = app.remote_edit {
        let (name, value) = match field {
            RemoteField::Host => ("Host (user@host or ssh alias)", app.remote.host.clone()),
            RemoteField::Base => (
                "Base dir (blank = remote temp)",
                app.remote.base_dir.clone(),
            ),
            RemoteField::Password => (
                "Password (blank = key)",
                "*".repeat(app.remote.password.len()),
            ),
        };
        vec![
            Line::from(vec![
                Span::styled(
                    format!(" Edit {name}: "),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("[ {value}█ ]"), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled(" [Enter] Save ", Style::default().fg(Color::Green)),
                Span::styled(" | [Esc] Cancel", Style::default().fg(Color::Red)),
            ]),
        ]
    } else if app.input_mode {
        let mut shape_spans = vec![Span::styled(
            " Enter Shape (Tab/Left/Right): ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )];
        for (i, dim) in app.shape_dims.iter().enumerate() {
            if i > 0 {
                shape_spans.push(Span::raw(" x "));
            }
            if i == app.active_dim_idx {
                shape_spans.push(Span::styled("[ ", Style::default().fg(Color::Yellow)));
                shape_spans.push(Span::raw(dim));
                shape_spans.push(Span::styled("█ ]", Style::default().fg(Color::Yellow)));
            } else {
                shape_spans.push(Span::raw(dim));
            }
        }
        vec![
            Line::from(shape_spans),
            Line::from(vec![
                Span::styled(" [Enter] Save ", Style::default().fg(Color::Green)),
                Span::styled(" | [Esc] Cancel", Style::default().fg(Color::Red)),
            ]),
        ]
    } else {
        let remote_line = if app.remote.enabled {
            Line::from(vec![
                Span::styled(" [m] Remote: ", Style::default().fg(Color::Cyan)),
                Span::styled("ON", Style::default().fg(Color::Green)),
                Span::styled("  [h]ost: ", Style::default().fg(Color::Cyan)),
                Span::raw(if app.remote.host.is_empty() {
                    "<unset>"
                } else {
                    app.remote.host.as_str()
                }),
                Span::styled("  [g]base remote dir: ", Style::default().fg(Color::Cyan)),
                Span::raw(if app.remote.base_dir.is_empty() {
                    "<temp>"
                } else {
                    app.remote.base_dir.as_str()
                }),
                Span::styled("  [w]password", Style::default().fg(Color::Cyan)),
            ])
        } else {
            Line::from(vec![
                Span::styled(" [m] Remote: ", Style::default().fg(Color::Cyan)),
                Span::styled("OFF (local)", Style::default().fg(Color::DarkGray)),
            ])
        };
        let toggles_line = if app.remote.enabled {
            Line::from(vec![
                Span::styled(" [f] force-sync: ", Style::default().fg(Color::Cyan)),
                Span::raw(if app.force_sync { "ON" } else { "off" }),
                Span::styled(" | [t] re-bench peak: ", Style::default().fg(Color::Cyan)),
                Span::raw(if app.disable_throughput_cache {
                    "ON"
                } else {
                    "off"
                }),
            ])
        } else {
            Line::from(vec![
                Span::styled(" [t] re-bench peak: ", Style::default().fg(Color::Cyan)),
                Span::raw(if app.disable_throughput_cache {
                    "ON"
                } else {
                    "off"
                }),
            ])
        };
        vec![
            Line::from(vec![
                Span::styled(" [b] Backend: ", Style::default().fg(Color::Cyan)),
                Span::raw(backend_name),
                Span::styled(" | [p] Problem: ", Style::default().fg(Color::Cyan)),
                Span::raw(problem_name),
                Span::styled(" | [s] Shape: ", Style::default().fg(Color::Cyan)),
                Span::raw(app.shape_dims.join("x")),
            ]),
            Line::from(vec![
                Span::styled(" [i] In DType: ", Style::default().fg(Color::Cyan)),
                Span::raw(in_dtype),
                Span::styled(" | [o] Out DType: ", Style::default().fg(Color::Cyan)),
                Span::raw(out_dtype),
            ]),
            remote_line,
            toggles_line,
            Line::from(vec![
                Span::styled(" [Enter/r] Run ", Style::default().fg(Color::Green)),
                if app.run_rx.is_some() {
                    Span::styled(" | [c] Cancel", Style::default().fg(Color::Yellow))
                } else {
                    Span::styled(" | [q/Esc] Quit", Style::default().fg(Color::Red))
                },
                Span::styled(" | [?] Help", Style::default().fg(Color::Cyan)),
                Span::raw(if app.run_rx.is_some() {
                    "  --> RUNNING..."
                } else {
                    ""
                }),
            ]),
        ]
    };

    let controls_p = Paragraph::new(controls_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Configuration & Hotkeys "),
    );
    f.render_widget(controls_p, right_chunks[0]);

    if app.run_rx.is_some() {
        let mut ansi_state = AnsiStyle::default();
        let lines: Vec<Line> = app
            .output_lines
            .iter()
            .rev()
            .take(50)
            .rev()
            .map(|raw_line| {
                let parsed = parse_ansi(raw_line, &mut ansi_state);
                let spans: Vec<Span> = parsed
                    .into_iter()
                    .map(|(style, text)| Span::styled(text, into_ratatui_style(style)))
                    .collect();
                Line::from(spans)
            })
            .collect();

        let output_p = Paragraph::new(lines)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" Live Output — {} ", app.status)),
            );
        f.render_widget(output_p, right_chunks[1]);
    } else {
        if let Some(selected_idx) = app.run_list_state.selected() {
            if let Some(run) = app.runs.get(selected_idx) {
                let mut lines = Vec::new();
                for (idx, event) in run.events.iter().enumerate() {
                    let benchmarked = event.count(crate::CandidateKind::Benchmarked);
                    let skipped = event.count(crate::CandidateKind::Skipped);
                    let invalid = event.count(crate::CandidateKind::Invalid);

                    let title = if let Some(sc) = event.short_circuit {
                        Span::styled(
                            format!(
                                "▶ Event #{idx} | Fastest: {} ⚡ short-circuit: {:.2}%",
                                event.fastest, sc
                            ),
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(ratatui::style::Modifier::BOLD),
                        )
                    } else {
                        Span::styled(
                            format!("▶ Event #{idx} | Fastest: {}", event.fastest),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(ratatui::style::Modifier::BOLD),
                        )
                    };
                    lines.push(Line::from(title));
                    lines.push(Line::from(vec![
                        Span::raw("    • "),
                        Span::raw(format!("Batches: {} | Candidates: ", event.tuning_batches)),
                        Span::styled(
                            format!("{} benchmarked", benchmarked),
                            Style::default().fg(Color::Green),
                        ),
                        Span::raw(", "),
                        Span::styled(
                            format!("{} skipped", skipped),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::raw(", "),
                        Span::styled(
                            format!("{} invalid", invalid),
                            Style::default().fg(Color::Red),
                        ),
                    ]));

                    if !event.candidate_progress.is_empty() {
                        lines.push(Line::from(vec![Span::raw("    • Progress to limit:")]));
                        for (i, p) in event.candidate_progress.iter().copied().enumerate() {
                            let bar_len = ((p / 100.0) * 20.0).clamp(0.0, 20.0) as usize;
                            let bar_str = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
                            let color = if p >= 100.0 {
                                Color::Yellow
                            } else {
                                Color::Green
                            };
                            lines.push(Line::from(vec![
                                Span::raw(format!("      Candidate #{}: {:5.1}% [", i + 1, p)),
                                Span::styled(bar_str, Style::default().fg(color)),
                                Span::raw("]"),
                                if p >= 100.0 {
                                    Span::styled(" ⚡", Style::default().fg(Color::Yellow))
                                } else {
                                    Span::raw("")
                                },
                            ]));
                        }
                    }

                    if !event.bounds.is_empty() {
                        lines.push(Line::from(vec![Span::raw("    • Throughput bounds:")]));
                        for bound in &event.bounds {
                            let color = if bound.starts_with("Short circuiting") {
                                Color::Yellow
                            } else {
                                Color::Reset
                            };
                            lines.push(Line::from(vec![
                                Span::raw("      "),
                                Span::styled(bound, Style::default().fg(color)),
                            ]));
                        }
                    }

                    if !event.context.is_empty() {
                        lines.push(Line::from(vec![Span::raw(
                            "    • Planner context (tuning groups):",
                        )]));
                        for line in event.context.lines() {
                            lines.push(Line::from(vec![Span::raw("      "), Span::raw(line)]));
                        }
                    }

                    if !event.key.is_empty() {
                        lines.push(Line::from(vec![Span::raw("    • Key:")]));
                        lines.push(Line::from(vec![Span::raw("      "), Span::raw(&event.key)]));
                    }

                    if !event.candidates.is_empty() {
                        lines.push(Line::from(vec![Span::raw(format!(
                            "    • Candidates ({}):",
                            event.candidates.len()
                        ))]));
                        for candidate in &event.candidates {
                            let color = match candidate.kind {
                                crate::CandidateKind::Benchmarked => Color::Green,
                                crate::CandidateKind::Skipped => Color::DarkGray,
                                crate::CandidateKind::Invalid => Color::Red,
                                crate::CandidateKind::Other => Color::Reset,
                            };
                            lines.push(Line::from(vec![
                                Span::raw("      "),
                                Span::styled(&candidate.text, Style::default().fg(color)),
                            ]));
                        }
                    }

                    lines.push(Line::from(""));
                }
                if lines.is_empty() {
                    lines.push(Line::from("No events found in this log."));
                }
                let p = Paragraph::new(lines)
                    .scroll((app.events_scroll, 0))
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" {} Events (PgUp/PgDown to scroll) ", run.name)),
                    );
                f.render_widget(p, right_chunks[1]);
            }
        } else {
            let p = Paragraph::new("Select a run from the left panel.")
                .block(Block::default().borders(Borders::ALL).title(" Events "));
            f.render_widget(p, right_chunks[1]);
        }
    }

    if app.show_help {
        render_help(f);
    }
}

/// A centered rectangle `percent_x` × `percent_y` of `area`, for overlay popups.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn render_help(f: &mut Frame) {
    let key = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let heading = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    let mut lines: Vec<Line> = Vec::new();
    let section = |lines: &mut Vec<Line>, title: &str, rows: &[(&str, &str)]| {
        lines.push(Line::from(Span::styled(title.to_string(), heading)));
        for (k, desc) in rows {
            lines.push(Line::from(vec![
                Span::styled(format!("  {k:<10}"), key),
                Span::raw(desc.to_string()),
            ]));
        }
        lines.push(Line::from(""));
    };

    section(
        &mut lines,
        "Configuration",
        &[
            ("b", "cycle backend"),
            ("p", "cycle problem (matmul/attention/flash/reduce)"),
            ("i / o", "cycle input / output dtype"),
            ("s", "edit shape"),
            ("t", "toggle re-benchmark peak throughput"),
        ],
    );
    section(
        &mut lines,
        "Remote (SSH)",
        &[
            ("m", "toggle running on remote"),
            ("h / g / w", "edit host / base dir / password"),
            ("f", "toggle force-sync (remote only)"),
        ],
    );
    section(
        &mut lines,
        "Run & archived runs",
        &[
            ("Enter / r", "run the current configuration"),
            ("c", "cancel the running job"),
            ("↑/k ↓/j", "select an archived run"),
            ("D", "delete the selected run"),
            ("PgUp/PgDn", "scroll events (Home/End: top/bottom)"),
        ],
    );
    section(
        &mut lines,
        "Run books",
        &[
            ("[ / ]", "previous / next book"),
            ("N / X", "new / delete book"),
            ("e", "rename book"),
            ("B", "cycle book default backend"),
            ("a", "add current config as an entry"),
            ("J / K", "move entry selection down / up"),
            ("x", "delete selected entry"),
            ("V", "cycle selected entry's backend override"),
            ("R", "run selected entry"),
            ("A", "run all entries in the book"),
        ],
    );
    section(
        &mut lines,
        "General",
        &[("?", "toggle this help"), ("q / Esc", "quit")],
    );

    let area = centered_rect(70, 90, f.area());
    let popup = Paragraph::new(lines)
        .wrap(ratatui::widgets::Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Keybindings — press any key to close "),
        );
    f.render_widget(Clear, area);
    f.render_widget(popup, area);
}
