use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui;
use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontId};

use crate::ansi::{self, AnsiStyle};
use crate::{
    CandidateKind, DTYPE_NAMES, TuneEvent, example_dir, list_runs, load_events, run_log_path,
    runs_dir,
};

/// Selectable backends: (dropdown label, `--backend` value, cargo `--features` value). wgpu is
/// the baseline; the others need their feature (and toolchain), and wgpu can't do tensor cores.
const BACKENDS: [(&str, &str, &str); 5] = [
    ("wgpu", "wgpu", "backend"),
    ("cuda", "cuda", "cuda"),
    ("vulkan", "vulkan", "vulkan"),
    ("metal", "metal", "metal"),
    ("cpu", "cpu", "cpu"),
];

/// A message streamed from the worker thread running the `runner` subprocess.
enum RunMsg {
    Line(String),
    Done { ok: bool },
}

/// One archived run: its directory, parsed events, and whether it is currently shown.
struct RunView {
    name: String,
    dir: PathBuf,
    events: Vec<TuneEvent>,
    selected: bool,
}

#[derive(Clone, Copy)]
struct MatmulShape {
    m: usize,
    k: usize,
    n: usize,
}

pub struct AutotuneObservabilityApp {
    runs: Vec<RunView>,
    selected: usize,
    input_dtype: usize,
    output_dtype: usize,
    matmuls: Vec<MatmulShape>,
    only_short_circuits: bool,
    status: String,
    output: LayoutJob,
    ansi_style: AnsiStyle,
    text_color: Color32,
    run_rx: Option<Receiver<RunMsg>>,
    /// Directory name of the in-flight run, selected once it finishes successfully.
    pending_run: Option<String>,
}

impl Default for AutotuneObservabilityApp {
    fn default() -> Self {
        let mut app = Self {
            runs: Vec::new(),
            selected: 0,
            input_dtype: 0,
            output_dtype: 0,
            matmuls: vec![MatmulShape {
                m: 512,
                k: 512,
                n: 512,
            }],
            only_short_circuits: false,
            status: String::from("Ready."),
            output: LayoutJob::default(),
            ansi_style: AnsiStyle::default(),
            text_color: Color32::GRAY,
            run_rx: None,
            pending_run: None,
        };
        app.rescan_runs(None);
        app
    }
}

impl AutotuneObservabilityApp {
    fn running(&self) -> bool {
        self.run_rx.is_some()
    }

    /// Rebuild the run list from disk, preserving which runs are shown (by name) and optionally
    /// forcing `select` to be shown.
    fn rescan_runs(&mut self, select: Option<&str>) {
        let shown: Vec<String> = self
            .runs
            .iter()
            .filter(|r| r.selected)
            .map(|r| r.name.clone())
            .collect();

        self.runs = list_runs()
            .into_iter()
            .map(|dir| {
                let name = dir
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let events = load_events(&run_log_path(&dir));
                let selected = Some(name.as_str()) == select || shown.contains(&name);
                RunView {
                    name,
                    dir,
                    events,
                    selected,
                }
            })
            .collect();

        // Default to showing the newest run when nothing is selected.
        if !self.runs.iter().any(|r| r.selected) {
            if let Some(first) = self.runs.first_mut() {
                first.selected = true;
            }
        }
    }

    /// Launch the runner through `cargo run` on a worker thread. Cargo builds it on demand (in
    /// the UI's profile, with the selected backend's feature), and stdout/stderr — build
    /// progress and the run itself — stream back line by line. A fresh process per run means an
    /// empty in-memory autotune cache, so the same config re-tunes every time.
    fn run_matmul(&mut self) {
        let (_, backend, feature) = BACKENDS[self.selected];
        let input = DTYPE_NAMES[self.input_dtype];
        let output = DTYPE_NAMES[self.output_dtype];

        let mut cargo_args: Vec<String> = vec!["run".into()];
        if is_release() {
            cargo_args.push("--release".into());
        }
        cargo_args.extend(["--bin", "runner", "--features", feature].map(String::from));
        cargo_args.push("--".into());

        let shape_names: Vec<String> = self
            .matmuls
            .iter()
            .map(|shape| format!("{}x{}x{}", shape.m, shape.k, shape.n))
            .collect();
        let stamp = now_millis();
        let id = format!(
            "{stamp}-{backend}-{}-{input}-{output}",
            shape_names.join("_")
        );
        let run_dir = runs_dir().join(&id);

        cargo_args
            .extend(["--backend", backend, "--input", input, "--output", output].map(String::from));
        for shape in &self.matmuls {
            cargo_args.extend([
                "--matmul".to_string(),
                format!("{}x{}x{}", shape.m, shape.k, shape.n),
            ]);
        }
        cargo_args.extend([
            "--run-dir".to_string(),
            run_dir.to_string_lossy().into_owned(),
        ]);

        self.pending_run = Some(id);

        self.ansi_style = AnsiStyle::default();
        self.push_line(&format!("$ cargo {}", cargo_args.join(" ")));
        self.status = String::from("Running… (first run of a backend compiles it)");

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || stream_command(cargo_args, tx));
        self.run_rx = Some(rx);
    }

    /// Append one line to the console output, colouring its ANSI spans.
    fn push_line(&mut self, line: &str) {
        for (style, text) in ansi::parse_ansi(line, &mut self.ansi_style) {
            let color = ansi_color(style, self.text_color);
            self.output.append(
                &text,
                0.0,
                TextFormat {
                    font_id: FontId::monospace(12.0),
                    color,
                    ..Default::default()
                },
            );
        }
        self.output.append("\n", 0.0, TextFormat::default());
    }

    fn poll_run(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.run_rx else { return };
        let mut lines = Vec::new();
        let mut finished = None;
        loop {
            match rx.try_recv() {
                Ok(RunMsg::Line(line)) => lines.push(line),
                Ok(RunMsg::Done { ok }) => finished = Some(ok),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = finished.or(Some(false));
                    break;
                }
            }
        }
        for line in lines {
            self.push_line(&line);
        }

        match finished {
            Some(true) => {
                self.run_rx = None;
                let id = self.pending_run.take();
                self.rescan_runs(id.as_deref());
                self.status = match id {
                    Some(id) => format!("Run {id} archived."),
                    None => String::from("Run finished."),
                };
            }
            Some(false) => {
                self.run_rx = None;
                self.pending_run = None;
                self.status = String::from("Run failed — see output panel.");
            }
            None => ctx.request_repaint(),
        }
    }

    fn delete_run(&mut self, dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
        self.rescan_runs(None);
        self.status = String::from("Deleted run.");
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Map an ANSI style to a concrete colour. The palette is chosen to stay legible on both light
/// and dark themes (no pure black/white); the default colour follows the current theme.
fn ansi_color(style: AnsiStyle, default_color: Color32) -> Color32 {
    const NORMAL: [Color32; 8] = [
        Color32::from_rgb(0x88, 0x88, 0x88), // "black" → gray so it stays visible
        Color32::from_rgb(0xD9, 0x53, 0x4F), // red
        Color32::from_rgb(0x3C, 0xB3, 0x71), // green
        Color32::from_rgb(0xC7, 0xA0, 0x08), // yellow
        Color32::from_rgb(0x4C, 0x8D, 0xD1), // blue
        Color32::from_rgb(0xB8, 0x6B, 0xD1), // magenta
        Color32::from_rgb(0x2F, 0xA8, 0xA8), // cyan
        Color32::from_rgb(0xC8, 0xC8, 0xC8), // white
    ];
    const BRIGHT: [Color32; 8] = [
        Color32::from_rgb(0xAA, 0xAA, 0xAA),
        Color32::from_rgb(0xF0, 0x6A, 0x66),
        Color32::from_rgb(0x5A, 0xD6, 0x8E),
        Color32::from_rgb(0xE6, 0xC2, 0x2E),
        Color32::from_rgb(0x6F, 0xAE, 0xF0),
        Color32::from_rgb(0xD3, 0x8B, 0xF0),
        Color32::from_rgb(0x4F, 0xC8, 0xC8),
        Color32::from_rgb(0xF0, 0xF0, 0xF0),
    ];
    match style.color {
        None => default_color,
        Some(i) => {
            let table = if style.bright { &BRIGHT } else { &NORMAL };
            table[(i % 8) as usize]
        }
    }
}

/// Run `cargo <args>` in the example dir, streaming interleaved stdout+stderr as [`RunMsg::Line`]
/// and finishing with [`RunMsg::Done`].
fn stream_command(args: Vec<String>, tx: Sender<RunMsg>) {
    let mut child = match Command::new("cargo")
        .current_dir(example_dir())
        // Force colour: cargo and rustc emit ANSI codes we parse for the console pane.
        .env("CARGO_TERM_COLOR", "always")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            let _ = tx.send(RunMsg::Line(format!("failed to launch cargo: {err}")));
            let _ = tx.send(RunMsg::Done { ok: false });
            return;
        }
    };

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let tx_out = tx.clone();
    let tx_err = tx.clone();
    let out_reader = thread::spawn(move || pipe_lines(stdout, &tx_out));
    let err_reader = thread::spawn(move || pipe_lines(stderr, &tx_err));

    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
    let _ = out_reader.join();
    let _ = err_reader.join();
    let _ = tx.send(RunMsg::Done { ok });
}

fn pipe_lines(reader: impl std::io::Read, tx: &Sender<RunMsg>) {
    for line in BufReader::new(reader).lines().map_while(Result::ok) {
        if tx.send(RunMsg::Line(line)).is_err() {
            break;
        }
    }
}

impl eframe::App for AutotuneObservabilityApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.text_color = ui.visuals().text_color();
        self.poll_run(&ctx);

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
            });

            ui.horizontal(|ui| {
                ui.label("Matmuls (m×k · k×n)");
                let can_remove = self.matmuls.len() > 1;
                let mut remove = None;
                for (index, shape) in self.matmuls.iter_mut().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            size_fields(ui, shape);
                            if can_remove && ui.small_button("×").clicked() {
                                remove = Some(index);
                            }
                        });
                    });
                }
                if let Some(index) = remove {
                    self.matmuls.remove(index);
                }
                if ui.button("+ add").clicked() {
                    self.matmuls.push(MatmulShape {
                        m: 512,
                        k: 512,
                        n: 512,
                    });
                }
            });

            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.running(), |ui| {
                    if ui.button("Run matmul").clicked() {
                        self.run_matmul();
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
                ui.label("Hold Shift while dragging a dimension to change m, k, and n together.");
            });
            ui.add_space(4.0);
        });

        egui::Panel::bottom("output")
            .resizable(true)
            .default_size(150.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Runner output");
                    if ui.button("clear").clicked() {
                        self.output = LayoutJob::default();
                        self.ansi_style = AnsiStyle::default();
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

        egui::CentralPanel::default().show(ui, |ui| {
            if !self.runs.iter().any(|r| r.selected) {
                ui.label("No run selected. Run a matmul, or tick a run on the left.");
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

/// Whether the UI itself is a release build, so the runner is compiled with the same profile
/// (and cargo reuses the UI's own build artifacts instead of a second full compile).
fn is_release() -> bool {
    std::env::current_exe()
        .ok()
        .map(|p| p.components().any(|c| c.as_os_str() == "release"))
        .unwrap_or(false)
}

fn size_fields(ui: &mut egui::Ui, shape: &mut MatmulShape) {
    let original = [
        shape.m.ilog2().min(14),
        shape.k.ilog2().min(14),
        shape.n.ilog2().min(14),
    ];
    let mut exponents = original;
    let mut changed = None;

    for (index, (label, exponent)) in ["m", "k", "n"]
        .into_iter()
        .zip(exponents.iter_mut())
        .enumerate()
    {
        ui.label(label);
        if ui
            .add(
                egui::DragValue::new(exponent)
                    .range(0..=14)
                    .speed(0.15)
                    .custom_formatter(|exponent, _| (1usize << exponent as u32).to_string()),
            )
            .changed()
        {
            changed = Some(index);
        }
    }

    if let Some(changed) = changed {
        let delta = exponents[changed] as isize - original[changed] as isize;
        if ui.input(|input| input.modifiers.shift) {
            exponents = original.map(|exponent| (exponent as isize + delta).clamp(0, 14) as u32);
        }
        shape.m = 1usize << exponents[0];
        shape.k = 1usize << exponents[1];
        shape.n = 1usize << exponents[2];
    }
}

fn dtype_field(ui: &mut egui::Ui, label: &str, id: &str, selected: &mut usize) {
    ui.label(label);
    egui::ComboBox::from_id_salt(id)
        .selected_text(DTYPE_NAMES[*selected])
        .show_ui(ui, |ui| {
            for (i, name) in DTYPE_NAMES.iter().enumerate() {
                ui.selectable_value(selected, i, *name);
            }
        });
}

fn event_view(ui: &mut egui::Ui, idx: usize, event: &TuneEvent) {
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

fn limit_graph(ui: &mut egui::Ui, event: &TuneEvent) {
    let samples = &event.candidate_progress;
    if samples.is_empty() {
        return;
    }

    let desired_height = 92.0;
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), desired_height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();
    let frame = rect.shrink(2.0);

    painter.rect_filled(frame, egui::CornerRadius::same(6), visuals.faint_bg_color);

    let plot_rect = frame.shrink2(egui::vec2(10.0, 12.0));
    let max_sample = samples.iter().copied().fold(100.0_f32, f32::max).max(110.0);
    let scale = plot_rect.height() / max_sample.max(1.0);
    let threshold_y = plot_rect.bottom() - (100.0 * scale);

    painter.line_segment(
        [
            egui::pos2(plot_rect.left(), threshold_y),
            egui::pos2(plot_rect.right(), threshold_y),
        ],
        egui::Stroke::new(1.0, ORANGE),
    );
    painter.text(
        egui::pos2(plot_rect.left(), threshold_y - 1.0),
        egui::Align2::LEFT_BOTTOM,
        "100% limit",
        FontId::monospace(9.0),
        ORANGE,
    );

    let slot_width = plot_rect.width() / samples.len() as f32;
    for (index, percent) in samples.iter().copied().enumerate() {
        let slot_center = plot_rect.left() + slot_width * (index as f32 + 0.5);
        let bar_width = (slot_width * 0.62).clamp(8.0, 28.0);
        let bar_height = (percent * scale).min(plot_rect.height());
        let top = plot_rect.bottom() - bar_height;
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(slot_center - bar_width / 2.0, top),
            egui::pos2(slot_center + bar_width / 2.0, plot_rect.bottom()),
        );
        let fill = if percent > 100.0 { ORANGE } else { GREEN };
        painter.rect_filled(
            bar_rect,
            egui::CornerRadius::same(4),
            fill.gamma_multiply(0.88),
        );
        painter.rect_stroke(
            bar_rect,
            egui::CornerRadius::same(4),
            egui::Stroke::new(1.0, fill),
            egui::StrokeKind::Outside,
        );
        painter.text(
            egui::pos2(slot_center, (top - 2.0).max(plot_rect.top())),
            egui::Align2::CENTER_BOTTOM,
            format!("{percent:.0}%"),
            FontId::monospace(9.0),
            visuals.text_color(),
        );
        painter.text(
            egui::pos2(slot_center, plot_rect.bottom() + 2.0),
            egui::Align2::CENTER_TOP,
            format!("#{}", index + 1),
            FontId::monospace(9.0),
            GRAY,
        );
    }

    if event.short_circuit.is_some() {
        painter.text(
            egui::pos2(plot_rect.right(), plot_rect.top() - 1.0),
            egui::Align2::RIGHT_BOTTOM,
            "short-circuit crossed 100%",
            FontId::monospace(9.0),
            ORANGE,
        );
    }
}

const ORANGE: egui::Color32 = egui::Color32::from_rgb(0xE0, 0x8A, 0x1E);
const GREEN: egui::Color32 = egui::Color32::from_rgb(0x3C, 0xB3, 0x71);
const RED: egui::Color32 = egui::Color32::from_rgb(0xD9, 0x53, 0x4F);
const GRAY: egui::Color32 = egui::Color32::from_rgb(0x99, 0x99, 0x99);
