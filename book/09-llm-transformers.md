# Chapter 9: Working with LLMs and Transformers

At the heart of virtually all modern Large Language Models (LLMs) is the **Transformer** architecture. Originally introduced in the paper "Attention Is All You Need," it has become the de facto standard for sequence modeling tasks. This final chapter provides a deep dive into how the Transformer architecture is implemented in Burn, focusing on the `TransformerEncoder` module.

## The Transformer in Burn: A Composite `Module`

The `TransformerEncoder` (found in `crates/burn-nn/src/modules/transformer/encoder.rs`) is a perfect example of Burn's philosophy of building complex models by composing simpler `Module`s.

A `TransformerEncoder` is not a monolithic block; it is a stack of identical layers.

```rust
// crates/burn-nn/src/modules/transformer/encoder.rs

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    pub layers: Vec<TransformerEncoderLayer<B>>,
    // ... other config fields
}
```

The `TransformerEncoder` itself is a `Module` that contains a `Vec` of `TransformerEncoderLayer`s. The `#[derive(Module)]` macro ensures that all the parameters within this `Vec` of layers are correctly registered, tracked, and serialized.

## The `TransformerEncoderLayer`: The Core Building Block

Each `TransformerEncoderLayer` performs the core logic of the transformer. It is, itself, a `Module` composed of smaller, specialized modules.

```rust
// crates/burn-nn/src/modules/transformer/encoder.rs

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    mha: MultiHeadAttention<B>,
    pwff: PositionWiseFeedForward<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    dropout: Dropout,
    norm_first: bool,
}
```

### Deeper Dive: The Multi-Head Attention Module

The key innovation of the Transformer is the `MultiHeadAttention` module.

#### Multi-Head Attention Data Flow Diagram
```
Input (e.g., [batch, seq_len, d_model])
   |
   +------------------+------------------+
   |                  |                  |
   V                  V                  V
+---------+      +---------+      +---------+
| Linear (Q)|      | Linear (K)|      | Linear (V)|
+---------+      +---------+      +---------+
   | (Query)          | (Key)            | (Value)
   |                  |                  |
   | Split into Heads | Split into Heads | Split into Heads
   |                  |                  |
   `-------. .--------'                  |
           | |                           |
           V V                           |
  +--------------------+                 |
  | Scaled Dot-Product |                 |
  |      Attention     |                 |
  | (Matmul, Scale,    |                 |
  |  Mask, Softmax)    |                 |
  +--------------------+                 |
           |                             |
           `--------------. .------------'
                          | |
                          V V
                   +---------------+
                   |   Matmul (Z)  |
                   +---------------+
                          |
                          V
                 Concatenate Heads
                          |
                          V
                   +-------------+
                   | Linear (Out)|
                   +-------------+
                          |
                          V
                        Output
```
This diagram shows how the input is projected into Queries, Keys, and Values, then split across multiple "heads." Each head performs a scaled dot-product attention operation independently. The results from all heads are then concatenated and passed through a final linear layer. This allows the model to jointly attend to information from different representation subspaces at different positions.

### Code Example: Configuration and Masking

Building and using a `TransformerEncoder` involves two key steps: configuring it correctly and providing the necessary attention masks.

#### Configuration

You configure a `TransformerEncoder` using `TransformerEncoderConfig`. This allows you to set all the important hyperparameters.

```rust
use burn::prelude::*;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig};

fn create_transformer<B: Backend>(device: &B::Device) -> TransformerEncoder<B> {
    let config = TransformerEncoderConfig::new(
        /* d_model */ 256,
        /* d_ff */    1024,
        /* n_heads */ 8,
        /* n_layers */ 6,
    )
    .with_dropout(0.1)
    .with_norm_first(true); // Use Pre-LayerNorm

    config.init(device)
}
```

#### Masking

Masking is a critical concept in transformers. It tells the self-attention mechanism which tokens it should be allowed to "look at."

*   **Padding Mask (`mask_pad`)**: When you process a batch of sequences of different lengths, you pad the shorter sequences with a special "padding" token. The padding mask is a boolean tensor that tells the attention mechanism to ignore these padding tokens, so they don't influence the result.

*   **Attention Mask (`mask_attn`)**: This is a more general mask. For a generative LLM, you need an **autoregressive** or **causal** mask. This is a triangular mask that prevents a token at a given position from attending to any subsequent tokens. This is crucial because during inference, the model should only have access to the tokens it has already generated.

Burn provides a helper function to generate this mask:

```rust
use burn::nn::attention::generate_autoregressive_mask;
use burn::prelude::*;
use burn::nn::transformer::TransformerEncoderInput;

fn run_forward_pass_with_masks<B: Backend>() {
    let device = Default::default();
    let batch_size = 2;
    let seq_length = 5;
    let d_model = 16;
    let pad_token_id = 0;

    // Create a dummy transformer
    let transformer = TransformerEncoderConfig::new(d_model, 64, 4, 2).init::<B>(&device);

    // Create a dummy input tensor
    let input_tensor = Tensor::<B, 3>::random([batch_size, seq_length, d_model], Distribution::Default, &device);

    // 1. Create a Padding Mask
    // Let's say the second sequence in the batch is shorter
    let token_ids = Tensor::<B, 2, Int>::from_data(
        [[1, 2, 3, 4, 5], [1, 2, 3, pad_token_id, pad_token_id]],
        &device
    );
    // The mask should be `true` for valid tokens and `false` for padding.
    let pad_mask = token_ids.not_equal_elem(pad_token_id);

    // 2. Create an Autoregressive (Causal) Mask
    let causal_mask = generate_autoregressive_mask(batch_size, seq_length, &device);

    // Run the forward pass with the masks
    let input = TransformerEncoderInput::new(input_tensor)
        .mask_pad(pad_mask)
        .mask_attn(causal_mask);

    let output = transformer.forward(input);
    println!("Output shape: {:?}", output.shape());
}
```

This modular and composable design makes Burn's transformer implementation easy to understand, modify, and extend. By combining these powerful `Module` building blocks, you can construct sophisticated LLMs and other sequence-based models with confidence.

---

## Exercises

1.  **Explore `MultiHeadAttention`**:
    a.  Navigate to the `crates/burn-nn/src/modules/attention/mod.rs` file.
    b.  Look at the `MultiHeadAttention` struct. What are the main `Module` fields it contains? (Hint: You should see `Linear` layers for Queries, Keys, Values, and the output).
    c.  How is `d_model` and `n_heads` used to determine the dimensions of these linear layers?
2.  **Implement a Padding Mask**:
    a.  Write a function `create_padding_mask<B: Backend>(token_ids: &Tensor<B, 2, Int>, pad_token_id: i64) -> Tensor<B, 2, Bool>`.
    b.  This function should take a batch of token ID sequences (shape `[batch_size, seq_length]`) and return a boolean mask of the same shape where the value is `true` for every position that is *not* the `pad_token_id` and `false` where it is.
3.  **Thought Experiment**: The `TransformerEncoder` is suitable for models like BERT or GPT. However, for a sequence-to-sequence task (like translation), you would also need a `TransformerDecoder`. What is the key difference between a self-attention layer in an encoder and one in a decoder? (Hint: It involves another type of attention, often called "cross-attention").
