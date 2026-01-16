#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn start_token(&self) -> usize;
    fn end_token(&self) -> usize;
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
    fn start_token_value(&self) -> String {
        self.decode(&[self.start_token()])
    }
    fn end_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
}

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        // CI can run multiple tests/examples in parallel, and HuggingFace cache uses a file lock.
        // This occasionally causes flaky failures (LockAcquisition on *.lock). Retry with backoff.
        let mut last_err: Option<String> = None;
        let mut tokenizer = (0..8)
            .find_map(|attempt| {
                match tokenizers::Tokenizer::from_pretrained("gpt2", None) {
                    Ok(tok) => Some(tok),
                    Err(err) => {
                        let msg = err.to_string();
                        last_err = Some(msg.clone());

                        // Only retry on lock acquisition (best-effort flake fix).
                        if msg.contains("LockAcquisition") || msg.contains(".lock") {
                            std::thread::sleep(std::time::Duration::from_millis(50 * (attempt + 1) as u64));
                            None
                        } else {
                            // Non-lock failures should fail fast (network, model not found, etc.).
                            panic!("Failed to load gpt2 tokenizer: {msg}");
                        }
                    }
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "Failed to load gpt2 tokenizer after retries (last error: {})",
                    last_err.unwrap_or_else(|| "unknown".to_string())
                )
            });
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("[START]", true),
            tokenizers::AddedToken::from("[END]", true),
            tokenizers::AddedToken::from("[PAD]", true),
        ]);

        Self { tokenizer }
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        let text = match special_tokens {
            true => "[START]".to_owned() + value + "[END]",
            false => value.to_string(),
        };
        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }

    fn start_token(&self) -> usize {
        self.tokenizer.token_to_id("[START]").unwrap() as usize
    }

    fn end_token(&self) -> usize {
        self.tokenizer.token_to_id("[END]").unwrap() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = Gpt2Tokenizer::default();
        let text = "A sentence";

        let tokens = tokenizer.encode(text, false);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_add_start_end_token() {
        let tokenizer = Gpt2Tokenizer::default();
        let text = "A sentence";

        let tokens_without = tokenizer.encode(text, false);
        let tokens_with = tokenizer.encode(text, true);

        assert_eq!(tokens_with.len() - 2, tokens_without.len());
    }
}
