use eframe::egui;

use super::AutotuneObservabilityApp;
use crate::run_support::{BACKENDS, ProblemKind, RunSpec};
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
                let old_problem = self.problem;
                problem_field(ui, &mut self.problem);
                if old_problem != self.problem {
                    self.shapes = vec![self.problem.default_shape()];
                }
            });

            ui.horizontal(|ui| {
                let shape_label = match self.problem {
                    ProblemKind::Matmul => "Matmul shapes (m×k · k×n)",
                    ProblemKind::Attention => "Attention shapes (batch×seq×head)",
                    ProblemKind::FlashAttention => "Flash Attention shapes (batch×seq×head)",
                    ProblemKind::Reduce => "Reduce shapes (batch×dim1×dim2)",
                };
                let field_labels = self.problem.shape_labels();
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
                    self.shapes.push(self.problem.default_shape());
                }
            });

            ui.horizontal(|ui| {
                let mut changed = ui
                    .checkbox(&mut self.remote.enabled, "Run on remote (SSH)")
                    .changed();
                if self.remote.enabled {
                    ui.label("host");
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut self.remote.host)
                                .desired_width(150.0)
                                .hint_text("user@host or ssh alias"),
                        )
                        .changed();
                    ui.label("base");
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut self.remote.base_dir)
                                .desired_width(130.0)
                                .hint_text("blank = remote temp"),
                        )
                        .changed();
                    ui.label("pass");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.remote.password)
                            .password(true)
                            .desired_width(110.0)
                            .hint_text("blank = key"),
                    );
                    if ui
                        .add_enabled(!self.testing(), egui::Button::new("Test"))
                        .clicked()
                    {
                        self.test_remote_connection();
                    }
                    if self.testing() {
                        ui.spinner();
                    }
                    ui.checkbox(&mut self.force_sync, "force sync")
                        .on_hover_text("Bypass the sync cache and re-check every file this run");
                }
                if changed {
                    self.remote.save();
                }
            });

            ui.horizontal(|ui| {
                let selected_runs = self.runs.iter().filter(|r| r.selected).count();
                ui.add_enabled_ui(!self.running(), |ui| {
                    let (_, backend_name, _) = BACKENDS[self.selected];
                    let needs_build = !self.built_backends.iter().any(|b| b == backend_name);
                    let button_text = if self.remote.enabled {
                        format!("Run {} on remote", self.problem.label().to_lowercase())
                    } else if needs_build {
                        format!("Build & Run {}", self.problem.label().to_lowercase())
                    } else {
                        format!("Run {}", self.problem.label().to_lowercase())
                    };
                    if ui.button(button_text).clicked() {
                        self.run_selected_problem();
                    }
                });
                if self.running() {
                    ui.spinner();
                    if ui.button("Cancel").clicked() {
                        self.cancel_run();
                    }
                }
                ui.checkbox(&mut self.disable_throughput_cache, "Re-benchmark peak")
                    .on_hover_text(
                        "Disable cubecl's throughput cache so the peak bound is measured every run",
                    );
                ui.checkbox(&mut self.disable_short_circuit, "Disable short circuits")
                    .on_hover_text(
                        "Disable short circuits in autotune",
                    );
                let rerun_label = if selected_runs <= 1 {
                    String::from("Rerun Selected")
                } else {
                    format!("Rerun Selected ({selected_runs})")
                };
                let rerun_response = ui
                    .add_enabled(
                        !self.running() && selected_runs > 0,
                        egui::Button::new(rerun_label),
                    )
                    .on_hover_text(
                        "Shift-click to preview and override backend/input/output dtypes before rerunning",
                    );
                if rerun_response.clicked() {
                    if ui.input(|inp| inp.modifiers.shift) {
                        self.open_rerun_selected_popup();
                    } else {
                        self.rerun_selected_runs();
                    }
                }
                if ui.button("Rescan runs").clicked() {
                    self.rescan_backends();
                    self.rescan_runs(None);
                }
                if selected_runs >= 2 {
                    if ui
                        .button(if self.comparison_mode {
                            "Exit Comparison"
                        } else {
                            "Compare Selected"
                        })
                        .clicked()
                    {
                        self.comparison_mode = !self.comparison_mode;
                    }
                } else {
                    self.comparison_mode = false;
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
                    ProblemKind::Attention | ProblemKind::FlashAttention => {
                        "Attention shape fields map to batch, sequence length, and head dimension."
                    }
                    ProblemKind::Reduce => {
                        "Reduce shape fields map to batch, dim1, and dim2 (reducing over dim2)."
                    }
                });
            });
            ui.add_space(4.0);
        });
    }

    pub(super) fn render_rerun_popup(&mut self, ctx: &egui::Context) {
        let Some(requests) = self.rerun_popup.as_mut() else {
            return;
        };

        let mut open = true;
        let mut start = false;
        let mut cancel = false;
        egui::Window::new("Customize reruns")
            .collapsible(false)
            .resizable(true)
            .default_width(760.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Shift-clicked reruns let you preview the saved runs and override backend/input/output dtypes before replaying them.");
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        egui::Grid::new("rerun_popup_grid")
                            .striped(true)
                            .num_columns(6)
                            .show(ui, |ui| {
                                ui.strong("Run");
                                ui.strong("Problem");
                                ui.strong("Shapes");
                                ui.strong("Backend");
                                ui.strong("Input");
                                ui.strong("Output");
                                ui.end_row();

                                for (index, request) in requests.iter_mut().enumerate() {
                                    let label = request.source_label.as_deref().unwrap_or("(saved run)");
                                    let shapes = request
                                        .shapes
                                        .iter()
                                        .map(|shape| {
                                            shape
                                                .iter()
                                                .map(|dim| dim.to_string())
                                                .collect::<Vec<_>>()
                                                .join("x")
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ");

                                    ui.label(label);
                                    ui.label(request.problem.label());
                                    ui.label(shapes);

                                    egui::ComboBox::from_id_salt(("rerun_backend", index))
                                        .selected_text(&request.backend)
                                        .show_ui(ui, |ui| {
                                            for (label, backend, _) in BACKENDS.iter() {
                                                ui.selectable_value(
                                                    &mut request.backend,
                                                    (*backend).to_string(),
                                                    *label,
                                                );
                                            }
                                        });
                                    egui::ComboBox::from_id_salt(("rerun_input", index))
                                        .selected_text(&request.input)
                                        .show_ui(ui, |ui| {
                                            for dtype in crate::DTYPE_NAMES.iter() {
                                                ui.selectable_value(
                                                    &mut request.input,
                                                    (*dtype).to_string(),
                                                    *dtype,
                                                );
                                            }
                                        });
                                    egui::ComboBox::from_id_salt(("rerun_output", index))
                                        .selected_text(&request.output)
                                        .show_ui(ui, |ui| {
                                            for dtype in crate::DTYPE_NAMES.iter() {
                                                ui.selectable_value(
                                                    &mut request.output,
                                                    (*dtype).to_string(),
                                                    *dtype,
                                                );
                                            }
                                        });
                                    ui.end_row();
                                }
                            });
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Start reruns").clicked() {
                        start = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if cancel {
            open = false;
        }

        if start {
            if let Some(requests) = self.rerun_popup.take() {
                self.start_rerun_requests(requests);
            }
        } else if !open {
            self.rerun_popup = None;
        }
    }

    pub(super) fn render_run_book_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("run_book")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                let running = self.running();
                ui.add_space(4.0);

                // Keep the selection valid before anything indexes with it.
                self.selected_book = self.selected_book.min(self.run_books.books.len() - 1);
                let book_idx = self.selected_book;

                let mut save = false;
                let mut select_book = None;
                let mut new_book = false;
                let mut delete_book = false;
                let mut add_current = false;
                let mut run_all = false;
                let mut run_one = None;
                let mut load = None;
                let mut duplicate = None;
                let mut delete = None;
                let mut toggle_edit = None;

                // --- Book selector ---
                let book_names: Vec<String> =
                    self.run_books.books.iter().map(|book| book.name.clone()).collect();
                ui.horizontal(|ui| {
                    ui.strong("Book");
                    let selected_text = book_names
                        .get(book_idx)
                        .map(|name| Self::book_label(name))
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("run_book_select")
                        .selected_text(selected_text)
                        .show_ui(ui, |ui| {
                            for (i, name) in book_names.iter().enumerate() {
                                if ui
                                    .selectable_label(i == book_idx, Self::book_label(name))
                                    .clicked()
                                {
                                    select_book = Some(i);
                                }
                            }
                        });
                    if ui.small_button("+").on_hover_text("New book").clicked() {
                        new_book = true;
                    }
                    if ui
                        .add_enabled(self.run_books.books.len() > 1, egui::Button::new("🗑").small())
                        .on_hover_text("Delete this book")
                        .clicked()
                    {
                        delete_book = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("name");
                    let book = &mut self.run_books.books[book_idx];
                    save |= ui
                        .add(
                            egui::TextEdit::singleline(&mut book.name)
                                .desired_width(f32::INFINITY)
                                .hint_text("book name"),
                        )
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("backend");
                    let book = &mut self.run_books.books[book_idx];
                    egui::ComboBox::from_id_salt("run_book_backend")
                        .selected_text(book.backend.clone())
                        .show_ui(ui, |ui| {
                            for (label, backend, _) in BACKENDS.iter() {
                                save |= ui
                                    .selectable_value(
                                        &mut book.backend,
                                        (*backend).to_string(),
                                        *label,
                                    )
                                    .changed();
                            }
                        })
                        .response
                        .on_hover_text("Default backend for every entry in this book");
                });

                let book_backend = self.run_books.books[book_idx].backend.clone();
                let spec_count = self.run_books.books[book_idx].specs.len();
                ui.horizontal(|ui| {
                    if ui
                        .button("+ Add current")
                        .on_hover_text(
                            "Save the current backend, problem, dtypes and shapes as an entry",
                        )
                        .clicked()
                    {
                        add_current = true;
                    }
                    if ui
                        .add_enabled(
                            !running && spec_count > 0,
                            egui::Button::new(format!("▶ Run all ({spec_count})")),
                        )
                        .on_hover_text("Launch every entry, one after another")
                        .clicked()
                    {
                        run_all = true;
                    }
                });
                ui.separator();

                let editing = self.run_book_editing;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let specs = &mut self.run_books.books[book_idx].specs;
                        if specs.is_empty() {
                            ui.weak(
                                "Empty. Configure the controls above, then click \"+ Add current\".",
                            );
                        }
                        for (i, spec) in specs.iter_mut().enumerate() {
                            let is_editing = editing == Some(i);
                            ui.group(|ui| {
                                ui.push_id(i, |ui| {
                                    ui.horizontal(|ui| {
                                        save |= ui
                                            .add(
                                                egui::TextEdit::singleline(&mut spec.name)
                                                    .desired_width(f32::INFINITY)
                                                    .hint_text("name (optional)"),
                                            )
                                            .changed();
                                    });

                                    if is_editing {
                                        save |= edit_spec_fields(ui, spec, &book_backend);
                                    } else {
                                        let backend = match &spec.backend {
                                            Some(backend) => format!("{backend}*"),
                                            None => book_backend.clone(),
                                        };
                                        ui.label(format!(
                                            "{} · {} · {} → {}",
                                            spec.problem.label(),
                                            backend,
                                            spec.input,
                                            spec.output
                                        ));
                                        ui.label(
                                            egui::RichText::new(spec.shapes_string())
                                                .monospace()
                                                .small(),
                                        );
                                    }

                                    ui.horizontal(|ui| {
                                        if ui
                                            .add_enabled(
                                                !running,
                                                egui::Button::new("▶ Run").small(),
                                            )
                                            .clicked()
                                        {
                                            run_one = Some(i);
                                        }
                                        let edit_label = if is_editing { "✓ Done" } else { "✏ Edit" };
                                        if ui.small_button(edit_label).clicked() {
                                            toggle_edit = Some(i);
                                        }
                                        if ui
                                            .small_button("Load")
                                            .on_hover_text("Copy into the controls above")
                                            .clicked()
                                        {
                                            load = Some(i);
                                        }
                                        if ui.small_button("Dup").clicked() {
                                            duplicate = Some(i);
                                        }
                                        if ui.small_button("🗑").on_hover_text("Delete").clicked() {
                                            delete = Some(i);
                                        }
                                    });
                                });
                            });
                        }
                    });

                if save {
                    self.save_run_books();
                }
                if let Some(i) = select_book {
                    self.select_book(i);
                }
                if new_book {
                    self.new_run_book();
                }
                if delete_book {
                    self.delete_active_book();
                }
                if add_current {
                    self.add_current_to_run_book();
                }
                if run_all {
                    self.run_book_run_all();
                }
                if let Some(i) = run_one {
                    self.run_book_run_one(i);
                }
                if let Some(i) = load {
                    self.load_spec_into_controls(i);
                }
                if let Some(i) = duplicate {
                    self.duplicate_run_book_spec(i);
                }
                if let Some(i) = toggle_edit {
                    self.run_book_editing =
                        if self.run_book_editing == Some(i) { None } else { Some(i) };
                }
                if let Some(i) = delete {
                    self.delete_run_book_spec(i);
                }
            });
    }

    fn book_label(name: &str) -> &str {
        if name.trim().is_empty() {
            "(unnamed)"
        } else {
            name
        }
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
                let mut finish_rename = None;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (i, run) in self.runs.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                if ui.small_button("🗑").clicked() {
                                    delete = Some(i);
                                }
                                let is_renaming = self
                                    .rename_buffer
                                    .as_ref()
                                    .map_or(false, |(ri, _)| *ri == i);
                                if is_renaming {
                                    let (_, buffer) = self.rename_buffer.as_mut().unwrap();
                                    let response = ui.text_edit_singleline(buffer);
                                    if response.lost_focus()
                                        && ui.input(|inp| inp.key_pressed(egui::Key::Enter))
                                    {
                                        finish_rename = Some((i, buffer.clone()));
                                    } else if response.lost_focus() {
                                        finish_rename = Some((i, buffer.clone()));
                                    }
                                } else {
                                    let label = run.custom_name.as_deref().unwrap_or(&run.name);
                                    ui.checkbox(&mut run.selected, "");
                                    if ui.small_button("✏").on_hover_text("Rename run").clicked()
                                    {
                                        self.rename_buffer = Some((i, label.to_string()));
                                    }
                                    // Truncate the (long) run name so the panel can shrink below it;
                                    // the full name is on hover, and clicking it toggles selection.
                                    let name = ui
                                        .add(
                                            egui::Label::new(label)
                                                .truncate()
                                                .sense(egui::Sense::click()),
                                        )
                                        .on_hover_text(label);
                                    if name.clicked() {
                                        run.selected = !run.selected;
                                    }
                                }
                            });
                        }
                    });
                if let Some((i, name)) = finish_rename {
                    self.rename_run(i, name);
                    self.rename_buffer = None;
                }
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

            if self.comparison_mode {
                self.render_comparison(ui);
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

    fn render_comparison(&mut self, ui: &mut egui::Ui) {
        let selected: Vec<_> = self.runs.iter().filter(|r| r.selected).collect();
        ui.heading(format!("Comparing {} runs", selected.len()));
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let first_run = selected.first().unwrap();
            for event in &first_run.events {
                ui.group(|ui| {
                    ui.strong(format!("{} - {}", event.fastest, event.key));
                    for run in &selected {
                        if let Some(matching) = run.events.iter().find(|e| e.key == event.key) {
                            ui.horizontal(|ui| {
                                ui.label(run.custom_name.as_deref().unwrap_or(&run.name));
                                ui.label(format!("Fastest: {}", matching.fastest));
                                if let Some(sc) = matching.short_circuit {
                                    ui.label(format!("Short circuit: {}%", sc));
                                } else {
                                    ui.label(format!(
                                        "Tuning batches: {}",
                                        matching.tuning_batches
                                    ));
                                }
                            });
                        } else {
                            ui.horizontal(|ui| {
                                ui.label(run.custom_name.as_deref().unwrap_or(&run.name));
                                ui.weak("No matching event");
                            });
                        }
                    }
                });
            }
        });
    }
}

/// Inline editors for a run book entry's problem, backend override, dtypes and shapes. Returns
/// whether any field changed, so the caller can persist the book.
fn edit_spec_fields(ui: &mut egui::Ui, spec: &mut RunSpec, book_backend: &str) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("problem");
        egui::ComboBox::from_id_salt("edit_problem")
            .selected_text(spec.problem.label())
            .show_ui(ui, |ui| {
                for problem in ProblemKind::ALL {
                    changed |= ui
                        .selectable_value(&mut spec.problem, problem, problem.label())
                        .changed();
                }
            });
        ui.label("on");
        let backend_text = match &spec.backend {
            Some(backend) => backend.clone(),
            None => format!("book ({book_backend})"),
        };
        egui::ComboBox::from_id_salt("edit_backend")
            .selected_text(backend_text)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_label(spec.backend.is_none(), format!("book ({book_backend})"))
                    .clicked()
                    .then(|| spec.backend = None)
                    .is_some();
                for (label, backend, _) in BACKENDS.iter() {
                    if ui
                        .selectable_label(
                            spec.backend.as_deref() == Some(*backend),
                            *label,
                        )
                        .clicked()
                    {
                        spec.backend = Some((*backend).to_string());
                        changed = true;
                    }
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("in");
        egui::ComboBox::from_id_salt("edit_in")
            .selected_text(&spec.input)
            .show_ui(ui, |ui| {
                for dtype in crate::DTYPE_NAMES.iter() {
                    changed |= ui
                        .selectable_value(&mut spec.input, (*dtype).to_string(), *dtype)
                        .changed();
                }
            });
        ui.label("out");
        egui::ComboBox::from_id_salt("edit_out")
            .selected_text(&spec.output)
            .show_ui(ui, |ui| {
                for dtype in crate::DTYPE_NAMES.iter() {
                    changed |= ui
                        .selectable_value(&mut spec.output, (*dtype).to_string(), *dtype)
                        .changed();
                }
            });
    });

    let labels = spec.problem.shape_labels();
    let can_remove = spec.shapes.len() > 1;
    let mut remove_shape = None;
    for (j, shape) in spec.shapes.iter_mut().enumerate() {
        ui.push_id(j, |ui| {
            ui.horizontal(|ui| {
                changed |= size_fields(ui, shape, labels);
                if can_remove && ui.small_button("×").on_hover_text("Remove shape").clicked() {
                    remove_shape = Some(j);
                }
            });
        });
    }
    if let Some(j) = remove_shape {
        spec.shapes.remove(j);
        changed = true;
    }
    if ui.button("+ add shape").clicked() {
        spec.shapes.push(spec.problem.default_shape());
        changed = true;
    }

    changed
}
