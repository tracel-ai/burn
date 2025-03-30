pub trait Tokenizer {
    /// Load the tokenizer from the provided path.
    fn new(tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode a string into a list of token identifiers.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32>;

    /// Decode a list of token identifiers into a string.
    fn decode(&self, tokens: Vec<u32>) -> Result<String, String>;

    /// Beginning of sentence token.
    fn bos(&self) -> Result<String, String> {
        self.decode(vec![self.bos_id()])
    }

    /// Beginning of sentence token identifier.
    fn bos_id(&self) -> u32;

    /// End of sentence token.
    fn eos(&self) -> Result<String, String> {
        self.decode(vec![self.eos_id()])
    }

    /// End of sentence token identifier.
    fn eos_id(&self) -> u32;

    /// Stop token identifiers.
    fn stop_ids(&self) -> Vec<u32>;
}
