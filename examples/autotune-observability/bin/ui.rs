use autotune_observability::AutotuneObservabilityApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Autotune Observability",
        native_options,
        Box::new(|_cc| Ok(Box::new(AutotuneObservabilityApp::default()))),
    )
}
