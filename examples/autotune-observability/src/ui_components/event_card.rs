use crate::{CandidateKind, TuneEvent};

use super::graph::limit_graph;
use super::palette::{GRAY, GREEN, ORANGE, RED};

pub(crate) fn event_view(ui: &mut egui::Ui, idx: usize, event: &TuneEvent) {
    let benchmarked = event.count(CandidateKind::Benchmarked);
    let skipped = event.count(CandidateKind::Skipped);
    let invalid = event.count(CandidateKind::Invalid);

    let title = if let Some(short_circuit) = event.short_circuit {
        egui::RichText::new(format!(
            "#{idx}  {}  ⚡ short-circuit - achieved {:.2}% of limit",
            event.fastest, short_circuit
        ))
        .strong()
        .color(ORANGE)
    } else {
        egui::RichText::new(format!("#{idx}  {}", event.fastest)).strong()
    };

    egui::CollapsingHeader::new(title)
        .id_salt(idx)
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{} tuning batch(es)", event.tuning_batches));
                ui.separator();
                ui.colored_label(GREEN, format!("{benchmarked} benchmarked"));
                ui.colored_label(GRAY, format!("{skipped} skipped"));
                ui.colored_label(RED, format!("{invalid} invalid"));
            });

            if !event.candidate_progress.is_empty() {
                ui.add_space(6.0);
                limit_graph(ui, event);
            }

            if !event.bounds.is_empty() {
                ui.collapsing("throughput bounds", |ui| {
                    for line in &event.bounds {
                        let color = if line.starts_with("Short circuiting") {
                            ORANGE
                        } else {
                            ui.style().visuals.text_color()
                        };
                        ui.colored_label(color, egui::RichText::new(line).monospace());
                    }
                });
            }

            if !event.context.is_empty() {
                ui.collapsing("planner context (tuning groups)", |ui| {
                    ui.monospace(&event.context);
                });
            }
            if !event.key.is_empty() {
                ui.collapsing("key", |ui| {
                    ui.monospace(&event.key);
                });
            }
            if !event.candidates.is_empty() {
                ui.collapsing(format!("candidates ({})", event.candidates.len()), |ui| {
                    for candidate in &event.candidates {
                        let color = match candidate.kind {
                            CandidateKind::Benchmarked => GREEN,
                            CandidateKind::Skipped => GRAY,
                            CandidateKind::Invalid => RED,
                            CandidateKind::Other => ui.style().visuals.text_color(),
                        };
                        ui.colored_label(color, egui::RichText::new(&candidate.text).monospace());
                    }
                });
            }
        });
}