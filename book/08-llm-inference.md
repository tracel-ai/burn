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

The key components are:

*   **`embedding_token`**: An `Embedding` layer that converts input tokens (represented as integers) into dense vectors.
*   **`embedding_pos`**: An `Embedding` layer that provides positional information, so the model knows the order of the tokens.
*   **`transformer`**: A `TransformerEncoder`, which is the core of the model. It uses self-attention to process the sequence of token embeddings and build a contextual understanding.
*   **`output`**: A `Linear` layer that takes the output of the transformer and projects it back to the size of the vocabulary, producing a probability distribution over the next possible tokens.

## The Inference Process: A Step-by-Step Guide

While the example code focuses on the `forward_training` method, the inference process for a generative model is slightly different. It's an **autoregressive** process, meaning we generate one token at a time, feed it back into the model, and then generate the next one.

Here's how an inference loop would work:

1.  **Initialization**: Start with a "prompt," which is a sequence of initial tokens (e.g., the beginning of a sentence).
2.  **Tokenization**: Convert the prompt text into a sequence of integer token IDs using a tokenizer.
3.  **Model Forward Pass**:
    a.  Pass the current sequence of tokens through the model's `forward` method. This involves getting the token and positional embeddings, passing them through the transformer, and finally through the output `Linear` layer.
    b.  The output of the model will be a tensor of shape `[batch_size, seq_length, vocab_size]`, where the last vector in the sequence (`output.select(1, seq_length - 1)`) represents the logits (raw probabilities) for the *next* token.
4.  **Sampling**: We need to choose the next token from this probability distribution. There are several ways to do this, a process called **sampling**:
    *   **Greedy Sampling**: Simply choose the token with the highest probability (the `argmax` of the logits). This is simple but can lead to repetitive text.
    *   **Top-k Sampling**: Randomly sample from the `k` most likely next tokens. This adds some randomness and can produce more interesting text.
    *   **Nucleus Sampling (Top-p)**: Randomly sample from the smallest set of tokens whose cumulative probability is greater than some threshold `p`.
5.  **Append and Repeat**: Append the chosen token ID to your sequence of tokens.
6.  **Loop**: Repeat steps 3-5 until a special "end of sequence" token is generated or you reach a maximum desired length.
7.  **Detokenization**: Convert the final sequence of token IDs back into human-readable text.

### An ASCII Diagram of the Inference Loop

```
+-------------------------------------------------+
|               Start with a Prompt               |
| e.g., "The quick brown fox"                     |
+-------------------------------------------------+
                 | (Tokenize)
                 V
+-------------------------------------------------+
|      Initial Token Sequence: [12, 4, 33, 19]    |
+-------------------------------------------------+
                 |
  .------------> | (1. Forward Pass through Model)
  |              V
  | +-------------------------------------------+
  | |      Model predicts logits for next token |
  | +-------------------------------------------+
  |              | (2. Sample from logits)
  |              V
  | +-------------------------------------------+
  | |     Choose next token, e.g., token 8      |
  | +-------------------------------------------+
  |              | (3. Append to sequence)
  |              V
  | +-------------------------------------------+
  | |   New Sequence: [12, 4, 33, 19, 8]        |
  | +-------------------------------------------+
  |              | (Loop until stop condition)
  `--------------'
                 |
                 V
+-------------------------------------------------+
|      Final Tokens: [12, 4, 33, 19, 8, ...]      |
+-------------------------------------------------+
                 | (Detokenize)
                 V
+-------------------------------------------------+
|        Generated Text: "The quick brown fox jumps..." |
+-------------------------------------------------+
```

### Code Example: Implementing the Inference Loop

Let's translate the diagram into a concrete Rust function. This function will take a model, a tokenizer, a prompt, and generate a sequence of new tokens.

```rust
use burn::prelude::*;
use burn::tensor::random::Distribution;
use burn::tensor::{Data, Element, Int};

// A simplified TextGenerationModel for demonstration
#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    // ... fields as defined before
}

impl<B: Backend> TextGenerationModel<B> {
    // A simplified forward pass for inference
    pub fn forward_inference(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_length] = tokens.dims();
        let device = &self.devices()[0];

        let positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let token_emb = self.embedding_token.forward(tokens);
        let pos_emb = self.embedding_pos.forward(positions);
        let embedding = (token_emb + pos_emb) / 2;

        let mask = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let input = TransformerEncoderInput::new(embedding).mask_attn(mask);

        let output = self.transformer.forward(input);
        self.output.forward(output)
    }
}

// Sampling function (Greedy)
fn sample_greedy<B: Backend>(logits: Tensor<B, 1>) -> u32 {
    let next_token = logits.argmax(0).into_scalar();
    // Assuming the element type can be converted to u32
    next_token.elem::<i64>() as u32
}

pub fn generate<B: Backend>(
    model: &TextGenerationModel<B>,
    tokenizer: &MyTokenizer, // Assuming a tokenizer struct
    prompt: &str,
    max_length: usize,
) -> String {
    let device = &model.devices()[0];
    let mut tokens = tokenizer.encode(prompt);

    for _ in 0..max_length {
        let token_tensor = Tensor::<B, 2, Int>::from_data(
            Data::new(tokens.clone(), Shape::new([1, tokens.len()])),
            device,
        );

        // Get the logits for the very last token in the sequence
        let logits = model.forward_inference(token_tensor);
        let [_, seq_length, vocab_size] = logits.dims();
        let logits_last = logits.slice([0..1, (seq_length - 1)..seq_length]);
        let logits_last = logits_last.reshape([vocab_size]);

        // Sample the next token
        let next_token = sample_greedy(logits_last);

        // Stop if an end-of-sequence token is generated
        if next_token == tokenizer.eos_token_id() {
            break;
        }

        tokens.push(next_token);
    }

    tokenizer.decode(&tokens)
}

```
This example shows the fundamental logic. In a real application, you would implement more sophisticated sampling strategies (like top-k or nucleus sampling) and handle batching for efficiency. However, the core autoregressive loop remains the same, making it a powerful and versatile technique for text generation.

---

## Exercises

1.  **Implement Top-k Sampling**:
    a.  Write a new sampling function called `sample_top_k<B: Backend>(logits: Tensor<B, 1>, k: usize) -> u32`.
    b.  Inside the function, you will need to:
        i.  Find the top `k` highest logit values and their corresponding indices. (Hint: `Tensor::sort_with_indices` will be useful here).
        ii. "Mask" out all other logits, setting them to a very low value (like `f32::NEG_INFINITY`).
        iii. Apply a softmax function to the masked logits to turn them into probabilities.
        iv. Sample from this new probability distribution. (Hint: `Tensor::random_choice` can be used for this).
    c.  Replace the `sample_greedy` call in the `generate` function with your new `sample_top_k` function.
2.  **Add a Temperature Parameter**: Modify your `sample_top_k` function to include a `temperature` parameter (a `f32`). Before applying the softmax, divide the logits by the temperature. What effect does a temperature > 1.0 have? What about a temperature < 1.0?
3.  **Thought Experiment**: The current inference code re-processes the entire sequence of tokens in every step. This is inefficient. How could you use the "Key-Value (KV) Cache" of a transformer to make this process much faster? (Hint: You would only need to process the *newest* token at each step).
