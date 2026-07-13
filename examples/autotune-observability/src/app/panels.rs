use eframe::egui;

use super::AutotuneObservabilityApp;
use crate::run_support::{BACKENDS, MatmulShape, ProblemKind};
use crate::ui_components::{dtype_field, event_view, problem_field, size_fields};

impl AutotuneObservabilityApp {
    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("controls").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Backend");
                egui::ComboBox::from_id_salt("backend")
                    .selected_text(BACKENDS[self.selected].0)
                    .show_ui(ui, |ui| {
                        for (i, (label, _, _)) in BACKENDS.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, i, *label);
                        }
                    });
                ui.separator();
                dtype_field(ui, "in", "in_dtype", &mut self.input_dtype);
                dtype_field(ui, "out", "out_dtype", &mut self.output_dtype);
                ui.separator();
                problem_field(ui, &mut self.problem);
            });

            ui.horizontal(|ui| {
                let (shape_label, field_labels) = match self.problem {
                    ProblemKind::Matmul => ("Matmul shapes (m×k · k×n)", ["m", "k", "n"]),
                    ProblemKind::Attention => (
                        "Attention shapes (batch×seq×head)",
                        ["batch", "seq", "head"],
                    ),
                };
                ui.label(shape_label);
                let can_remove = self.shapes.len() > 1;
                let mut remove = None;
                for (index, shape) in self.shapes.iter_mut().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            size_fields(ui, shape, field_labels);
                            if can_remove && ui.small_button("×").clicked() {
                                remove = Some(index);
                            }
                        });
                    });
                }
                if let Some(index) = remove {
                    self.shapes.remove(index);
                }
                if ui.button("+ add").clicked() {
                    self.shapes.push(MatmulShape {
                        m: 512,
                        k: 512,
                        n: 512,
                    });
                }
            });

            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.running(), |ui| {
                    if ui
                        .button(format!("Run {}", self.problem.label().to_lowercase()))
                        .clicked()
                    {
                        self.run_selected_problem();
                    }
                });
                if self.running() {
                    ui.spinner();
                }
                if ui.button("Rescan runs").clicked() {
                    self.rescan_runs(None);
                }
                ui.checkbox(&mut self.only_short_circuits, "Only short-circuits");
                let (events, shorted) = self
                    .runs
                    .iter()
                    .filter(|r| r.selected)
                    .flat_map(|r| r.events.iter())
                    .fold((0usize, 0usize), |(e, s), ev| {
                        (e + 1, s + ev.short_circuit.is_some() as usize)
                    });
                ui.label(format!(
                    "{} shown event(s) · {shorted} short-circuited",
                    events
                ));
            });

            ui.horizontal(|ui| {
                ui.label(match self.problem {
                    ProblemKind::Matmul => {
                        "Hold Shift while dragging a dimension to change m, k, and n together."
                    }
                    ProblemKind::Attention => {
                        "Attention shape fields map to batch, sequence length, and head dimension."
                    }
                });
            });
            ui.add_space(4.0);
        });
    }

    pub(super) fn render_output_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("output")
            .resizable(true)
            .default_size(150.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Runner output");
                    if ui.button("clear").clicked() {
                        self.output = Default::default();
                        self.ansi_style = Default::default();
                    }
                    ui.label(&self.status);
                });
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.label(self.output.clone());
                    });
            });
    }

    pub(super) fn render_runs_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("runs")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.strong(format!("Runs ({})", self.runs.len()));
                ui.separator();

                let mut delete = None;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, run) in self.runs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.small_button("🗑").clicked() {
                                delete = Some(i);
                            }
                            ui.checkbox(&mut run.selected, &run.name);
                        });
                    }
                });
                if let Some(i) = delete {
                    let dir = self.runs[i].dir.clone();
                    self.delete_run(&dir);
                }
            });
    }

    pub(super) fn render_events_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            if !self.runs.iter().any(|r| r.selected) {
                ui.label("No run selected. Run a workload, or tick a run on the left.");
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                for run in self.runs.iter().filter(|r| r.selected) {
                    let shorted = run
                        .events
                        .iter()
                        .filter(|e| e.short_circuit.is_some())
                        .count();
                    let header = format!(
                        "{}  —  {} events, {shorted} short-circuited",
                        run.name,
                        run.events.len()
                    );
                    egui::CollapsingHeader::new(header)
                        .id_salt(&run.name)
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.push_id(&run.name, |ui| {
                                let mut shown = 0;
                                for (idx, event) in run.events.iter().enumerate() {
                                    if self.only_short_circuits && event.short_circuit.is_none() {
                                        continue;
                                    }
                                    event_view(ui, idx, event);
                                    ui.separator();
                                    shown += 1;
                                }
                                if shown == 0 {
                                    ui.weak("(no matching events)");
                                }
                            });
                        });
                }
            });
        });
    }
}
