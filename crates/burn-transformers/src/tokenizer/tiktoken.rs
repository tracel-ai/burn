use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "<|end_of_text|>";
const EOT_TOKEN: &str = "<|eot_id|>";
const EOM_TOKEN: &str = "<|eom_id|>";

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;
const SPECIAL_TOKENS: [&str; 11] = [
    BOS_TOKEN,
    EOS_TOKEN,
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    EOM_TOKEN, // end of message
    EOT_TOKEN, // end of turn
    "<|python_tag|>",
];
const PATTERN: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

pub struct Tiktoken {
    bpe: CoreBPE,
    bos_token_id: u32,
    eos_token_id: u32,
    eot_token_id: u32,
    eom_token_id: u32,
}

impl Tokenizer for Tiktoken {
    /// Load the [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    fn new(tiktoken_bpe_file: &str) -> Result<Self, String> {
        let file = File::open(tiktoken_bpe_file).map_err(|e| e.to_string())?;
        let mut mergeable_ranks: HashMap<Vec<u8>, u32> = HashMap::default();

        for line in BufReader::new(file).lines().flatten() {
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<u32>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }
        let num_base_tokens = mergeable_ranks.len();

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (0..NUM_RESERVED_SPECIAL_TOKENS - SPECIAL_TOKENS.len())
                .into_iter()
                .map(|i| format!("<|reserved_special_token_{}|>", i + 2))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, (i + num_base_tokens) as u32))
            .collect::<HashMap<String, u32>>();

        let bos_token_id = special_tokens[BOS_TOKEN];
        let eos_token_id = special_tokens[EOS_TOKEN];
        let eot_token_id = special_tokens[EOT_TOKEN];
        let eom_token_id = special_tokens[EOM_TOKEN];

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        let tokens = self.bpe.encode_with_special_tokens(text);

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .map(|t| t as u32)
            .collect()
    }

    fn decode(&self, tokens: Vec<u32>) -> Result<String, String> {
        self.bpe
            .decode(tokens.into_iter().collect())
            .map_err(|e| e.to_string())
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id as u32
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id as u32
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![
            self.eos_id(),
            self.eom_token_id as u32,
            self.eot_token_id as u32,
        ]
    }
}
