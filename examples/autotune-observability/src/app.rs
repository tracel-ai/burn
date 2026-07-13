use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui;
use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontId};

use crate::ansi::{self, AnsiStyle};
use crate::{CandidateKind, DTYPE_NAMES, TuneEvent, example_dir, load_events, log_path};

/// Selectable backends: (dropdown label, `--backend` value, cargo `--features` value). wgpu is
/// the baseline; the others need their feature (and toolchain), and wgpu can't do tensor cores.
const BACKENDS: [(&str, &str, &str); 5] = [
    ("wgpu", "wgpu", "backend"),
    ("cuda (tensor cores)", "cuda", "cuda"),
    ("vulkan", "vulkan", "vulkan"),
    ("metal", "metal", "metal"),
    ("cpu", "cpu", "cpu"),
];

/// A message streamed from the worker thread running the `runner` subprocess.
enum RunMsg {
    Line(String),
    Done { ok: bool },
}

pub struct AutotuneObservabilityApp {
    log_path: PathBuf,
    events: Vec<TuneEvent>,
    selected: usize,
    input_dtype: usize,
    output_dtype: usize,
    m: usize,
    k: usize,
    n: usize,
    only_short_circuits: bool,
    status: String,
    output: LayoutJob,
    ansi_style: AnsiStyle,
    text_color: Color32,
    run_rx: Option<Receiver<RunMsg>>,
}

impl Default for AutotuneObservabilityApp {
    fn default() -> Self {
        let log_path = log_path();
        let events = load_events(&log_path);
        Self {
            log_path,
            events,
            selected: 0,
            input_dtype: 0,
            output_dtype: 0,
            m: 512,
            k: 512,
            n: 512,
            only_short_circuits: false,
            status: String::from("Ready."),
            output: LayoutJob::default(),
            ansi_style: AnsiStyle::default(),
            text_color: Color32::GRAY,
            run_rx: None,
        }
    }
}

impl AutotuneObservabilityApp {
    fn running(&self) -> bool {
        self.run_rx.is_some()
    }

    fn reload(&mut self) {
        self.events = load_events(&self.log_path);
        self.status = format!("Loaded {} tune events.", self.events.len());
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
        let (m, k, n) = (self.m.to_string(), self.k.to_string(), self.n.to_string());
        cargo_args.extend(
            [
                "--backend",
                backend,
                "--m",
                m.as_str(),
                "--k",
                k.as_str(),
                "--n",
                n.as_str(),
                "--input",
                input,
                "--output",
                output,
            ]
            .map(String::from),
        );

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
            Some(ok) => {
                self.run_rx = None;
                self.reload();
                self.status = if ok {
                    format!("Run finished — {} tune events.", self.events.len())
                } else {
                    String::from("Run failed — see output panel.")
                };
            }
            None => ctx.request_repaint(),
        }
    }

    fn clear_log(&mut self) {
        let _ = std::fs::remove_file(log_path());
        self.events.clear();
        self.status = String::from("Cleared the log view.");
    }
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
        .env("CUBECL_LOG_AUTOTUNE", "full")
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
                ui.label("Matmul (m×k · k×n)");
                size_field(ui, "m", &mut self.m);
                size_field(ui, "k", &mut self.k);
                size_field(ui, "n", &mut self.n);

                ui.add_enabled_ui(!self.running(), |ui| {
                    if ui.button("Run matmul").clicked() {
                        self.run_matmul();
                    }
                });
                if self.running() {
                    ui.spinner();
                }
                if ui.button("Reload log").clicked() {
                    self.reload();
                }
                if ui.button("Clear log").clicked() {
                    self.clear_log();
                }
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.only_short_circuits, "Only short-circuits");
                let count = self.events.iter().filter(|e| e.short_circuit).count();
                ui.label(format!(
                    "{} events · {count} short-circuited",
                    self.events.len()
                ));
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

        egui::CentralPanel::default().show(ui, |ui| {
            if self.events.is_empty() {
                ui.label("No tune events yet. Pick a backend/dtype/size and Run matmul.");
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, event) in self.events.iter().enumerate() {
                    if self.only_short_circuits && !event.short_circuit {
                        continue;
                    }
                    event_view(ui, idx, event);
                    ui.separator();
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

fn size_field(ui: &mut egui::Ui, label: &str, value: &mut usize) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).range(1..=16384).speed(8.0));
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

    let title = if event.short_circuit {
        egui::RichText::new(format!("#{idx}  {}  ⚡ short-circuit", event.fastest))
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

const ORANGE: egui::Color32 = egui::Color32::from_rgb(0xE0, 0x8A, 0x1E);
const GREEN: egui::Color32 = egui::Color32::from_rgb(0x3C, 0xB3, 0x71);
const RED: egui::Color32 = egui::Color32::from_rgb(0xD9, 0x53, 0x4F);
const GRAY: egui::Color32 = egui::Color32::from_rgb(0x99, 0x99, 0x99);
