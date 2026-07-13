use egui::FontId;

use crate::TuneEvent;

use super::palette::{GRAY, GREEN, ORANGE};

pub(crate) fn limit_graph(ui: &mut egui::Ui, event: &TuneEvent) {
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