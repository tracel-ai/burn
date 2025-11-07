# Chapter 8: Building LLM Inference in Burn

Training a large language model (LLM) is a significant undertaking, but once you have a trained model, the next step is to use it for **inference**—generating new text. This chapter breaks down how to build an inference pipeline for a generative LLM in Burn, based on the architecture found in the `text-generation` example.

## The Model: A Transformer Encoder

The model in the `text-generation` example (`examples/text-generation/src/model.rs`) is a classic transformer-based architecture, which is the foundation for many modern LLMs.

```rust
// examples/text-generation/src/model.rs

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    // ... other fields
}
```
This chapter will focus on the inference process. For a deeper dive into the `TransformerEncoder` itself, see Chapter 9.

## The Inference Process: Autoregressive Generation

The inference process for a generative model is **autoregressive**, meaning we generate one token at a time, feed it back into the model, and then generate the next one.

### Optimizing Inference with a KV Cache

A naive implementation of autoregressive inference would re-process the entire sequence of tokens at every step. This is very inefficient. A crucial optimization is the **Key-Value (KV) Cache**.

Inside the self-attention mechanism, the "Key" and "Value" vectors are computed for each token. In a generative context, these Key and Value vectors for past tokens do not change. The KV cache stores these vectors so they don't have to be recomputed at every step. At each new step, we only need to compute the K and V vectors for the *newest* token and append them to the cache.

#### KV Cache Diagram
```
Step 1 (Prompt): "The cat" -> tokens [5, 8]
  - Input: [5, 8]
  - Compute K, V for token 5 -> Store in Cache: K_cache=[K5], V_cache=[V5]
  - Compute K, V for token 8 -> Store in Cache: K_cache=[K5, K8], V_cache=[V5, V8]
  - Predict next token: "sat" (token 12)

Step 2: "The cat sat" -> tokens [5, 8, 12]
  - Input: [12] (only the new token)
  - K_cache, V_cache are provided to the model.
  - Compute K, V for token 12 -> Store in Cache: K_cache=[K5, K8, K12], V_cache=[V5, V8, V12]
  - Predict next token: "on" (token 20)

... and so on.
```
This makes inference much faster as the input sequence grows. Burn's transformer implementation includes a `forward_autoregressive_inference` method that utilizes such a cache.

## Runnable Example: A Complete Inference Pipeline

This example shows a complete, runnable inference pipeline, including a `top-k` sampling strategy.

```rust
use burn::prelude::*;
use burn::tensor::random::Distribution;
use burn::tensor::{Data, Element, Int};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::nn::attention::generate_autoregressive_mask;

// Simplified model for demonstration
#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
}
// Assume config and init are defined...

impl<B: Backend> TextGenerationModel<B> {
    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // This is a simplified forward pass for clarity.
        // A real implementation would use the KV cache via `forward_autoregressive_inference`.
        let [batch_size, seq_length] = tokens.dims();
        let device = &self.devices()[0];

        let positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let token_emb = self.embedding_token.forward(tokens);
        let pos_emb = self.embedding_pos.forward(positions);
        let embedding = token_emb + pos_emb;

        let mask = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let input = TransformerEncoderInput::new(embedding).mask_attn(mask);

        let output = self.transformer.forward(input);
        self.output.forward(output)
    }
}

// Sampling function (Top-k)
fn sample_top_k<B: Backend>(logits: Tensor<B, 1>, k: usize, temperature: f32) -> u32 {
    let [vocab_size] = logits.dims();
    let logits = logits / temperature;

    // Get the top k logits and their indices
    let (top_k_logits, top_k_indices) = logits.clone().sort_with_indices(0, false);
    let top_k_logits = top_k_logits.slice([0..k]);
    let top_k_indices = top_k_indices.slice([0..k]);

    // Apply softmax to the top k logits to get probabilities
    let probs = top_k_logits.exp() / top_k_logits.exp().sum();

    // Sample from the top k probabilities
    let sampled_index_in_top_k = probs.multinomial(1).into_scalar().elem::<i64>() as usize;

    // Get the original token index
    let next_token_index = top_k_indices.slice([sampled_index_in_top_k..sampled_index_in_top_k + 1]);
    next_token_index.into_scalar().elem::<i64>() as u32
}

pub fn generate<B: Backend>(
    model: &TextGenerationModel<B>,
    // In a real app, tokenizer would be more complex
    tokenizer_vocab: &Vec<String>,
    prompt: &str,
    max_length: usize,
) -> String {
    let device = &model.devices()[0];
    // Dummy tokenization
    let mut tokens: Vec<u32> = prompt.split_whitespace().map(|s| tokenizer_vocab.iter().position(|v| v == s).unwrap() as u32).collect();

    for _ in 0..max_length {
        let token_tensor = Tensor::<B, 2, Int>::from_data(
            Data::new(tokens.iter().map(|&t| t as i64).collect(), Shape::new([1, tokens.len()])),
            device,
        );

        let logits = model.forward_inference(token_tensor);
        let [_, seq_length, vocab_size] = logits.dims();
        let logits_last = logits.slice([0..1, (seq_length - 1)..seq_length]).reshape([vocab_size]);

        let next_token = sample_top_k(logits_last, 5, 0.8);

        if next_token as usize >= tokenizer_vocab.len() || tokenizer_vocab[next_token as usize] == "<eos>" {
            break;
        }
        tokens.push(next_token);
    }

    // Dummy detokenization
    tokens.iter().map(|&t| tokenizer_vocab[t as usize].clone()).collect::<Vec<String>>().join(" ")
}
```

---

## Exercises

1.  **Implement Top-p (Nucleus) Sampling**:
    a.  Write a new sampling function `sample_top_p`.
    b.  This will be more complex. You'll need to:
        i.  Sort the logits in descending order.
        ii.  Calculate the cumulative sum of their probabilities (after softmax).
        iii. Find the indices of the logits that are part of the "nucleus" (their cumulative probability is <= `p`).
        iv. Mask out all other logits, re-normalize the probabilities, and sample.
2.  **Add a Temperature Parameter**: Modify your `sample_top_k` function to include a `temperature` parameter (a `f32`). Before applying the softmax, divide the logits by the temperature. What effect does a temperature > 1.0 have? What about a temperature < 1.0?
3.  **Thought Experiment**: The current inference code re-processes the entire sequence of tokens in every step. This is inefficient. How could you use the "Key-Value (KV) Cache" of a transformer to make this process much faster? (Hint: You would only need to process the *newest* token at each step).
4.  **Batch Generation**: The current `generate` function works on a single prompt (`batch_size = 1`). How would you modify it to handle a batch of multiple prompts simultaneously? What challenges would you face, especially given that different prompts in the batch might finish generating at different times?
