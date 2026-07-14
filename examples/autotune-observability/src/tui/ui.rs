use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::DTYPE_NAMES;
use crate::ansi::{AnsiStyle, parse_ansi};
use crate::run_support::{BACKENDS, ProblemKind};

use super::app::App;

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

    f.render_stateful_widget(runs_list, chunks[0], &mut app.run_list_state);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(10)].as_ref())
        .split(chunks[1]);

    let backend_name = BACKENDS[app.backend_idx].0;
    let problem_name = ProblemKind::ALL[app.problem_idx].label();
    let in_dtype = DTYPE_NAMES[app.in_dtype_idx];
    let out_dtype = DTYPE_NAMES[app.out_dtype_idx];

    let controls_text = if app.input_mode {
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
        vec![
            Line::from(vec![
                Span::styled(" [B] Backend: ", Style::default().fg(Color::Cyan)),
                Span::raw(backend_name),
                Span::styled(" | [P] Problem: ", Style::default().fg(Color::Cyan)),
                Span::raw(problem_name),
                Span::styled(" | [S] Shape: ", Style::default().fg(Color::Cyan)),
                Span::raw(app.shape_dims.join("x")),
            ]),
            Line::from(vec![
                Span::styled(" [I] In DType: ", Style::default().fg(Color::Cyan)),
                Span::raw(in_dtype),
                Span::styled(" | [O] Out DType: ", Style::default().fg(Color::Cyan)),
                Span::raw(out_dtype),
            ]),
            Line::from(vec![
                Span::styled(" [Enter/R] Run ", Style::default().fg(Color::Green)),
                if app.run_rx.is_some() {
                    Span::styled(" | [C] Cancel", Style::default().fg(Color::Yellow))
                } else {
                    Span::styled(" | [Q/Esc] Quit", Style::default().fg(Color::Red))
                },
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
                    .title(" Live Output "),
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
}
