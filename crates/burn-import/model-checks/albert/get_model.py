#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "onnxruntime>=1.18.0",
#   "transformers>=4.44.0",
#   "sentencepiece>=0.2.0",
#   "numpy",
#   "torch",
# ]
# ///

import os
import sys
import onnx
from onnx import shape_inference, version_converter
import numpy as np
from pathlib import Path
import argparse


# Supported ALBERT models configuration
SUPPORTED_MODELS = {
    'albert-base-v2': {
        'hf_name': 'albert/albert-base-v2',
        'display_name': 'ALBERT Base v2',
        'seq_length': 128,
    },
}


def download_and_convert_model(model_name, output_path):
    """Download ALBERT model from HuggingFace and export to ONNX format."""
    from transformers import AlbertModel, AlbertTokenizer
    import torch

    model_config = SUPPORTED_MODELS[model_name]
    display_name = model_config['display_name']
    hf_name = model_config['hf_name']
    seq_length = model_config['seq_length']

    print(f"Downloading {display_name} model from HuggingFace...")
    tokenizer = AlbertTokenizer.from_pretrained(hf_name)
    model = AlbertModel.from_pretrained(hf_name)
    model.eval()

    print("Exporting to ONNX format...")

    # Create dummy inputs
    dummy_text = "This is a sample text for ONNX export."
    inputs = tokenizer(
        dummy_text,
        padding='max_length',
        max_length=seq_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # Export to ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        output_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'},
            'pooler_output': {0: 'batch_size'},
        },
        opset_version=16,
        do_constant_folding=True,
    )

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create ONNX file at {output_path}")


def process_model(input_path, output_path, target_opset=16):
    """Load, upgrade opset, and apply shape inference to model."""
    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    # Check and upgrade opset if needed
    current_opset = model.opset_import[0].version
    if current_opset < target_opset:
        print(f"Upgrading opset from {current_opset} to {target_opset}...")
        model = version_converter.convert_version(model, target_opset)

    # Apply shape inference
    print("Applying shape inference...")
    model = shape_inference.infer_shapes(model)

    # Save processed model
    onnx.save(model, output_path)
    print(f"✓ Processed model saved to: {output_path}")

    return model


def generate_test_data(model_path, output_path, model_name):
    """Generate test input/output data and save as PyTorch tensors."""
    import torch
    import onnxruntime as ort

    print("\nGenerating test data...")

    model_config = SUPPORTED_MODELS[model_name]
    seq_length = model_config['seq_length']

    # Create reproducible test input
    np.random.seed(42)
    batch_size = 1

    # Generate random token IDs (typical vocabulary size is 30000 for ALBERT)
    input_ids = np.random.randint(0, 30000, size=(batch_size, seq_length), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    token_type_ids = np.zeros((batch_size, seq_length), dtype=np.int64)

    print(f"  Input shapes:")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")
    print(f"    token_type_ids: {token_type_ids.shape}")

    # Run inference to get output
    session = ort.InferenceSession(model_path)
    outputs = session.run(
        None,
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
    )

    # Save as PyTorch tensors
    test_data = {
        'input_ids': torch.from_numpy(input_ids),
        'attention_mask': torch.from_numpy(attention_mask),
        'token_type_ids': torch.from_numpy(token_type_ids),
        'last_hidden_state': torch.from_numpy(outputs[0]),
        'pooler_output': torch.from_numpy(outputs[1]),
    }

    torch.save(test_data, output_path)

    print(f"  ✓ Test data saved to: {output_path}")
    print(f"    last_hidden_state shape: {outputs[0].shape}")
    print(f"    pooler_output shape: {outputs[1].shape}")


def main():
    parser = argparse.ArgumentParser(description='ALBERT Model Preparation Tool')
    parser.add_argument('--model', type=str, default='albert-base-v2',
                        choices=list(SUPPORTED_MODELS.keys()),
                        help=f'ALBERT model to download and prepare (default: albert-base-v2). Choices: {", ".join(SUPPORTED_MODELS.keys())}')
    parser.add_argument('--list', action='store_true',
                        help='List all supported models')

    args = parser.parse_args()

    if args.list:
        print("Supported ALBERT models:")
        for model_id, config in SUPPORTED_MODELS.items():
            print(f"  - {model_id:20s} ({config['display_name']})")
        return

    model_name = args.model
    display_name = SUPPORTED_MODELS[model_name]['display_name']

    print("=" * 60)
    print(f"{display_name} Model Preparation Tool")
    print("=" * 60)

    # Setup paths
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    original_path = artifacts_dir / f"{model_name}.onnx"
    processed_path = artifacts_dir / f"{model_name}_opset16.onnx"
    test_data_path = artifacts_dir / f"{model_name}_test_data.pt"

    # Check if we already have everything
    if processed_path.exists() and test_data_path.exists():
        print(f"\n✓ All files already exist for {display_name}:")
        print(f"  Model: {processed_path}")
        print(f"  Test data: {test_data_path}")
        print("\nNothing to do!")
        return

    # Download and convert if needed
    if not original_path.exists() and not processed_path.exists():
        print(f"\nStep 1: Downloading and converting {display_name} model...")
        download_and_convert_model(model_name, original_path)

    # Process model if needed
    if not processed_path.exists():
        print("\nStep 2: Processing model...")
        process_model(original_path, processed_path, target_opset=16)

        # Clean up original if we have the processed version
        if original_path.exists():
            original_path.unlink()

    # Generate test data if needed
    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(processed_path, test_data_path, model_name)

    print("\n" + "=" * 60)
    print(f"✓ {display_name} model preparation completed!")
    print(f"  Model: {processed_path}")
    print(f"  Test data: {test_data_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
