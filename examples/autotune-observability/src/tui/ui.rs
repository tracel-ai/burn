use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::DTYPE_NAMES;
use crate::run_support::{BACKENDS, ProblemKind};

use super::app::App;

pub fn ui(f: &mut Frame, app: &mut App) {
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

    let controls_text = vec![
        Line::from(vec![
            Span::styled(" [B] Backend: ", Style::default().fg(Color::Cyan)),
            Span::raw(backend_name),
            Span::styled(" | [P] Problem: ", Style::default().fg(Color::Cyan)),
            Span::raw(problem_name),
        ]),
        Line::from(vec![
            Span::styled(" [I] In DType: ", Style::default().fg(Color::Cyan)),
            Span::raw(in_dtype),
            Span::styled(" | [O] Out DType: ", Style::default().fg(Color::Cyan)),
            Span::raw(out_dtype),
        ]),
        Line::from(vec![
            Span::styled(" [Enter/R] Run ", Style::default().fg(Color::Green)),
            Span::styled(" | [Q/Esc] Quit", Style::default().fg(Color::Red)),
            Span::raw(if app.run_rx.is_some() {
                "  --> RUNNING..."
            } else {
                ""
            }),
        ]),
    ];

    let controls_p = Paragraph::new(controls_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Configuration & Hotkeys "),
    );
    f.render_widget(controls_p, right_chunks[0]);

    if app.run_rx.is_some() {
        let text: String = app
            .output_lines
            .iter()
            .rev()
            .take(50)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let output_p = Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Live Output "),
        );
        f.render_widget(output_p, right_chunks[1]);
    } else {
        if let Some(selected_idx) = app.run_list_state.selected() {
            if let Some(run) = app.runs.get(selected_idx) {
                let mut lines = Vec::new();
                for event in &run.events {
                    lines.push(Line::from(vec![Span::styled(
                        format!("Fastest: {} - {}", event.fastest, event.key),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )]));
                    if let Some(sc) = event.short_circuit {
                        lines.push(Line::from(format!("Short-circuit: {}%", sc)));
                    } else {
                        lines.push(Line::from(format!(
                            "Tuning batches: {}",
                            event.tuning_batches
                        )));
                    }
                    lines.push(Line::from(""));
                }
                if lines.is_empty() {
                    lines.push(Line::from("No events found in this log."));
                }
                let p = Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" {} Events ", run.name)),
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
