#![no_std]
#![warn(missing_docs)]

//! Shape-independent equation parsing shared by Burn's einsum interfaces.
//!
//! Named labels use the range `0..52`: uppercase letters first, followed by
//! lowercase letters. [`ELLIPSIS`] represents the dimensions matched by `...`.
//! Tensor ranks and dimension sizes are validated by the executor.

extern crate alloc;

use alloc::{vec, vec::Vec};
use core::fmt;

/// The label representing an ellipsis (`...`).
pub const ELLIPSIS: u8 = 52;

/// A parsed einsum equation, independent of operand shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Equation {
    /// Labels for each operand, including repeated labels and ellipses.
    pub inputs: Vec<Vec<u8>>,
    /// Output labels in their requested order.
    ///
    /// Implicit output starts with [`ELLIPSIS`], followed by the named labels
    /// occurring exactly once in all inputs, in alphabetical order.
    pub output: Vec<u8>,
}

impl Equation {
    /// Parse an equation without inspecting tensor shapes.
    pub fn parse(equation: &str) -> Result<Self, ParseError> {
        parse(equation)
    }

    /// Return an operand's logical rank when its subscript has no ellipsis.
    ///
    /// Empty subscripts have logical rank zero. Returns `None` for an operand
    /// containing an ellipsis or an index outside the input list.
    pub fn input_rank(&self, index: usize) -> Option<usize> {
        let input = self.inputs.get(index)?;
        (!input.contains(&ELLIPSIS)).then_some(input.len())
    }

    /// Return the logical output rank when it is independent of operand ranks.
    ///
    /// An output ellipsis contributes zero dimensions when no input contains
    /// an ellipsis. An omitted output ellipsis reduces its dimensions and does
    /// not prevent determining the output rank.
    pub fn output_rank(&self) -> Option<usize> {
        if self.output.contains(&ELLIPSIS)
            && self.inputs.iter().any(|input| input.contains(&ELLIPSIS))
        {
            None
        } else {
            Some(
                self.output
                    .iter()
                    .filter(|&&label| label != ELLIPSIS)
                    .count(),
            )
        }
    }
}

/// An invalid einsum equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A character other than an ASCII label, space, or valid separator.
    InvalidCharacter {
        /// The character's byte offset in the equation.
        index: usize,
        /// The unexpected character.
        character: char,
    },
    /// A dot that is not part of three consecutive dots.
    InvalidEllipsis {
        /// The dot's byte offset in the equation.
        index: usize,
    },
    /// More than one ellipsis in one input or the output.
    RepeatedEllipsis {
        /// The repeated ellipsis's byte offset in the equation.
        index: usize,
    },
    /// An output label that does not occur in any input.
    UnknownOutputLabel {
        /// The unknown label.
        label: char,
    },
    /// An output label occurring more than once.
    RepeatedOutputLabel {
        /// The repeated label.
        label: char,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCharacter { index, character } => write!(
                f,
                "einsum(): invalid character {character:?} at byte {index}; subscripts must be in [a-zA-Z]"
            ),
            Self::InvalidEllipsis { index } => write!(
                f,
                "einsum(): dot at byte {index} is not part of an ellipsis (...)"
            ),
            Self::RepeatedEllipsis { index } => write!(
                f,
                "einsum(): more than one ellipsis in the same subscript at byte {index}"
            ),
            Self::UnknownOutputLabel { label } => write!(
                f,
                "einsum(): output subscript {label} does not appear in any input operand"
            ),
            Self::RepeatedOutputLabel { label } => write!(
                f,
                "einsum(): output subscript {label} appears more than once in the output"
            ),
        }
    }
}

impl core::error::Error for ParseError {}

/// Parse an einsum equation using PyTorch-compatible string syntax.
///
/// Only ASCII spaces are ignored. Ellipses (`...`) and the output arrow (`->`)
/// must remain contiguous. An empty equation describes one scalar operand;
/// commas may separate empty scalar subscripts. Repeated input labels describe
/// diagonals, while output labels must be unique and occur in an input.
///
/// Without an explicit arrow, the output contains the ellipsis followed by
/// labels occurring exactly once across all inputs, sorted `A-Z`, then `a-z`.
pub fn parse(equation: &str) -> Result<Equation, ParseError> {
    let (input, explicit_output) = match equation.split_once("->") {
        Some((input, output)) => (input, Some(output)),
        None => (equation, None),
    };

    let mut inputs = Vec::new();
    let mut counts = [0usize; ELLIPSIS as usize];
    let mut offset = 0;
    for subscript in input.split(',') {
        let labels = parse_subscript(subscript, offset)?;
        for &label in &labels {
            if label != ELLIPSIS {
                counts[label as usize] += 1;
            }
        }
        inputs.push(labels);
        offset += subscript.len() + 1;
    }

    let output = if let Some(output) = explicit_output {
        let labels = parse_subscript(output, input.len() + 2)?;
        let mut seen = [false; ELLIPSIS as usize];
        for &label in &labels {
            if label == ELLIPSIS {
                continue;
            }
            let character = label_to_char(label);
            if counts[label as usize] == 0 {
                return Err(ParseError::UnknownOutputLabel { label: character });
            }
            if seen[label as usize] {
                return Err(ParseError::RepeatedOutputLabel { label: character });
            }
            seen[label as usize] = true;
        }
        labels
    } else {
        let mut labels = vec![ELLIPSIS];
        for label in 0..ELLIPSIS {
            if counts[label as usize] == 1 {
                labels.push(label);
            }
        }
        labels
    };

    Ok(Equation { inputs, output })
}

fn parse_subscript(subscript: &str, offset: usize) -> Result<Vec<u8>, ParseError> {
    let bytes = subscript.as_bytes();
    let mut labels = Vec::new();
    let mut ellipsis_seen = false;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b' ' => index += 1,
            b'.' => {
                if ellipsis_seen {
                    return Err(ParseError::RepeatedEllipsis {
                        index: offset + index,
                    });
                }
                if !subscript[index..].starts_with("...") {
                    return Err(ParseError::InvalidEllipsis {
                        index: offset + index,
                    });
                }
                labels.push(ELLIPSIS);
                ellipsis_seen = true;
                index += 3;
            }
            byte @ b'A'..=b'Z' => {
                labels.push(byte - b'A');
                index += 1;
            }
            byte @ b'a'..=b'z' => {
                labels.push(byte - b'a' + 26);
                index += 1;
            }
            _ => {
                return Err(ParseError::InvalidCharacter {
                    index: offset + index,
                    character: subscript[index..].chars().next().unwrap(),
                });
            }
        }
    }
    Ok(labels)
}

fn label_to_char(label: u8) -> char {
    if label < 26 {
        (b'A' + label) as char
    } else {
        (b'a' + label - 26) as char
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(labels: &str) -> Vec<u8> {
        parse_subscript(labels, 0).unwrap()
    }

    #[test]
    fn explicit_output_preserves_order_and_input_diagonals() {
        let equation = parse("ii,jk->ki").unwrap();
        assert_eq!(equation.inputs, vec![named("ii"), named("jk")]);
        assert_eq!(equation.output, named("ki"));
    }

    #[test]
    fn implicit_output_counts_occurrences_and_sorts_uppercase_first() {
        let equation = parse("ziiA,bj,j").unwrap();
        let mut output = vec![ELLIPSIS];
        output.extend(named("Abz"));
        assert_eq!(equation.output, output);
        assert_eq!(equation.output_rank(), Some(3));
    }

    #[test]
    fn scalars_and_empty_subscripts_are_valid() {
        let scalar = parse("").unwrap();
        assert_eq!(scalar.inputs, vec![vec![]]);
        assert_eq!(scalar.output, vec![ELLIPSIS]);
        assert_eq!(scalar.input_rank(0), Some(0));
        assert_eq!(scalar.output_rank(), Some(0));
        let product = parse(",i,->").unwrap();
        assert_eq!(product.inputs, vec![vec![], named("i"), vec![]]);
        assert!(product.output.is_empty());
        assert_eq!(parse("->").unwrap().inputs, vec![vec![]]);
    }

    #[test]
    fn ascii_spaces_are_ignored_between_tokens() {
        assert_eq!(parse(" i j , ... j -> i ... "), parse("ij,...j->i..."));
    }

    #[test]
    fn ellipses_can_be_retained_reordered_or_reduced() {
        let retained = parse("...ij,jk->i...k").unwrap();
        assert_eq!(retained.inputs[0], vec![ELLIPSIS, 34, 35]);
        assert_eq!(retained.output, vec![34, ELLIPSIS, 36]);
        assert_eq!(retained.input_rank(0), None);
        assert_eq!(retained.input_rank(1), Some(2));
        assert_eq!(retained.input_rank(2), None);
        assert_eq!(retained.output_rank(), None);
        assert_eq!(parse("...i->i").unwrap().output_rank(), Some(1));
        assert_eq!(parse("i->...i").unwrap().output_rank(), Some(1));
        assert_eq!(parse("...i").unwrap().output_rank(), None);
    }

    #[test]
    fn explicit_output_requires_unique_labels_present_in_inputs() {
        assert_eq!(
            parse("ij->k"),
            Err(ParseError::UnknownOutputLabel { label: 'k' })
        );
        assert_eq!(
            parse("ij->ii"),
            Err(ParseError::RepeatedOutputLabel { label: 'i' })
        );
    }

    #[test]
    fn malformed_or_repeated_ellipses_are_rejected_on_both_sides() {
        for equation in [".", "..", ". . .", "i->.", "i->..", "i->. . ."] {
            assert!(matches!(
                parse(equation),
                Err(ParseError::InvalidEllipsis { .. })
            ));
        }
        for equation in ["....", "...i...", "...->......", "i->...i..."] {
            assert!(matches!(
                parse(equation),
                Err(ParseError::RepeatedEllipsis { .. })
            ));
        }
    }

    #[test]
    fn invalid_labels_separators_and_non_ascii_whitespace_are_rejected() {
        for equation in [
            "i1", "i_", "α", "i\tj", "i\nj", "i\u{a0}j", "i-j", "i>j", "i- >i", "i->i->i", "i->i,i",
        ] {
            assert!(matches!(
                parse(equation),
                Err(ParseError::InvalidCharacter { .. })
            ));
        }
    }

    #[test]
    fn errors_report_offsets_in_the_full_equation() {
        assert_eq!(
            parse("i,j->k?"),
            Err(ParseError::InvalidCharacter {
                index: 6,
                character: '?',
            })
        );
        assert_eq!(
            parse("i,é"),
            Err(ParseError::InvalidCharacter {
                index: 2,
                character: 'é',
            })
        );
    }
}
