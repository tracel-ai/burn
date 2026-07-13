//! Minimal ANSI SGR (color) parser for the console output pane.
//!
//! Cargo and rustc emit standard 8/16-colour escape codes when told to (`CARGO_TERM_COLOR`),
//! which is what makes their output readable — green `Compiling`, red errors, and so on. This
//! turns a line of such text into styled spans; the UI maps [`AnsiStyle`] to concrete colours.
//! It is deliberately egui-free so it can be unit-tested on its own.

/// Colour/emphasis state carried across a line (and across lines, since SGR state persists).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnsiStyle {
    /// Base colour index 0..=7 (black, red, green, yellow, blue, magenta, cyan, white), or
    /// `None` for the terminal default.
    pub color: Option<u8>,
    /// Bright ("intense") variant of `color`.
    pub bright: bool,
    /// Bold text.
    pub bold: bool,
}

/// Split one line into styled spans, updating `style` as SGR codes are encountered.
pub fn parse_ansi(line: &str, style: &mut AnsiStyle) -> Vec<(AnsiStyle, String)> {
    let mut spans = Vec::new();
    let mut text = String::new();
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            text.push(c);
            continue;
        }
        // Control Sequence Introducer: ESC '['. Anything else we drop the ESC and move on.
        if chars.peek() != Some(&'[') {
            continue;
        }
        chars.next();

        let mut params = String::new();
        let mut final_byte = None;
        for nc in chars.by_ref() {
            if ('@'..='~').contains(&nc) {
                final_byte = Some(nc);
                break;
            }
            params.push(nc);
        }

        // Only 'm' (SGR) affects colour; ignore other sequences (cursor moves, etc.).
        if final_byte == Some('m') {
            if !text.is_empty() {
                spans.push((*style, std::mem::take(&mut text)));
            }
            apply_sgr(&params, style);
        }
    }

    if !text.is_empty() {
        spans.push((*style, text));
    }
    spans
}

fn apply_sgr(params: &str, style: &mut AnsiStyle) {
    let codes: Vec<u16> = if params.is_empty() {
        vec![0]
    } else {
        params.split(';').map(|p| p.parse().unwrap_or(0)).collect()
    };

    for code in codes {
        match code {
            0 => *style = AnsiStyle::default(),
            1 => style.bold = true,
            22 => style.bold = false,
            30..=37 => {
                style.color = Some((code - 30) as u8);
                style.bright = false;
            }
            90..=97 => {
                style.color = Some((code - 90) as u8);
                style.bright = true;
            }
            39 => style.color = None,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_colored_spans() {
        let mut style = AnsiStyle::default();
        let spans = parse_ansi("\u{1b}[0m\u{1b}[1m\u{1b}[32mCompiling\u{1b}[0m burn", &mut style);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].0.color, Some(2));
        assert!(spans[0].0.bold);
        assert_eq!(spans[0].1, "Compiling");
        assert_eq!(spans[1].0.color, None);
        assert_eq!(spans[1].1, " burn");
        // Trailing reset leaves the carried style at default for the next line.
        assert_eq!(style, AnsiStyle::default());
    }

    #[test]
    fn plain_text_is_one_span() {
        let mut style = AnsiStyle::default();
        let spans = parse_ansi("just text", &mut style);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].1, "just text");
        assert_eq!(spans[0].0, AnsiStyle::default());
    }

    #[test]
    fn ignores_non_sgr_sequences() {
        let mut style = AnsiStyle::default();
        let spans = parse_ansi("a\u{1b}[2Kb", &mut style);
        let joined: String = spans.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(joined, "ab");
    }
}
