use crate::DTYPE_NAMES;
use crate::run_support::ProblemKind;

pub(crate) fn size_fields(ui: &mut egui::Ui, shape: &mut Vec<usize>, labels: &[&'static str]) {
    let original: Vec<u32> = shape.iter().map(|s| s.ilog2().min(14)).collect();
    let mut exponents = original.clone();
    let mut changed = None;

    for (index, (label, exponent)) in labels.iter().zip(exponents.iter_mut()).enumerate() {
        ui.label(*label);
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
            exponents = original.iter().map(|exponent| (*exponent as isize + delta).clamp(0, 14) as u32).collect();
        }
        for (i, exponent) in exponents.iter().enumerate() {
            shape[i] = 1usize << exponent;
        }
    }
}

pub(crate) fn dtype_field(ui: &mut egui::Ui, label: &str, id: &str, selected: &mut usize) {
    ui.label(label);
    egui::ComboBox::from_id_salt(id)
        .selected_text(DTYPE_NAMES[*selected])
        .show_ui(ui, |ui| {
            for (i, name) in DTYPE_NAMES.iter().enumerate() {
                ui.selectable_value(selected, i, *name);
            }
        });
}

pub(crate) fn problem_field(ui: &mut egui::Ui, selected: &mut ProblemKind) {
    ui.label("Problem");
    egui::ComboBox::from_id_salt("problem")
        .selected_text(selected.label())
        .show_ui(ui, |ui| {
            for problem in ProblemKind::ALL {
                ui.selectable_value(selected, problem, problem.label());
            }
        });
}
