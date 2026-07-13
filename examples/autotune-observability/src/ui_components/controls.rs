use crate::run_support::MatmulShape;
use crate::DTYPE_NAMES;

pub(crate) fn size_fields(ui: &mut egui::Ui, shape: &mut MatmulShape) {
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