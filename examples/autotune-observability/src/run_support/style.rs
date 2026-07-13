use egui::Color32;

use crate::ansi::AnsiStyle;

/// Map an ANSI style to a concrete colour. The palette is chosen to stay legible on both light
/// and dark themes (no pure black/white); the default colour follows the current theme.
pub(crate) fn ansi_color(style: AnsiStyle, default_color: Color32) -> Color32 {
    const NORMAL: [Color32; 8] = [
        Color32::from_rgb(0x88, 0x88, 0x88),
        Color32::from_rgb(0xD9, 0x53, 0x4F),
        Color32::from_rgb(0x3C, 0xB3, 0x71),
        Color32::from_rgb(0xC7, 0xA0, 0x08),
        Color32::from_rgb(0x4C, 0x8D, 0xD1),
        Color32::from_rgb(0xB8, 0x6B, 0xD1),
        Color32::from_rgb(0x2F, 0xA8, 0xA8),
        Color32::from_rgb(0xC8, 0xC8, 0xC8),
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
