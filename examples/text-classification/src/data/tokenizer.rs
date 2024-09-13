// This module defines a trait `Tokenizer` that represents a common interface for all tokenizer
// types used in the text classification library. A specific implementation of this trait,
// `BertCasedTokenizer`, uses the BERT cased tokenization strategy provided by the `tokenizers` library.

// This trait represents the common interface for all tokenizer types.
// The `Send + Sync` bounds are necessary for allowing these operations
// to work across thread boundaries.
#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    /// Converts a text string into a sequence of tokens.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize;

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize;

    /// Gets the string representation of the padding token.
    /// The default implementation uses `decode` on the padding token.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

/// Struct represents a specific tokenizer using the BERT cased tokenization strategy.
pub struct BertCasedTokenizer {
    // The underlying tokenizer from the `tokenizers` library.
    tokenizer: tokenizers::Tokenizer,
}

// Default implementation for creating a new BertCasedTokenizer.
// This uses a pretrained BERT cased tokenizer model.
impl Default for BertCasedTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

// Implementation of the Tokenizer trait for BertCasedTokenizer.
impl Tokenizer for BertCasedTokenizer {
    // Convert a text string into a sequence of tokens using the BERT cased tokenization strategy.
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    /// Gets the size of the BERT cased tokenizer's vocabulary.
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}
