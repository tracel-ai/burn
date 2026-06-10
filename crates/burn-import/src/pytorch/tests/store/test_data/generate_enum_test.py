#!/usr/bin/env python3
"""
Generate PyTorch test data for enum variant path mismatch testing.

This script creates a PyTorch checkpoint that simulates how PyTorch models
export their state dicts WITHOUT enum variant names in the paths.

Example:
- PyTorch path: "feature.weight"
- Burn path:    "feature.BaseConv.weight"  (includes enum variant "BaseConv")

Run with: uv run generate_enum_test.py
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """
    Simple PyTorch model that represents what a Burn enum model would look like
    WITHOUT the enum variant names in the path.

    In Burn, this would be:
    struct ModelWithEnum {
        feature: ConvBlock,  // enum with BaseConv, DwsConv variants
        classifier: Linear,
    }

    But PyTorch exports it as flat paths without the enum variant names.
    """
    def __init__(self):
        super().__init__()
        # This represents the "feature" field which is an enum in Burn
        # PyTorch doesn't have enums, so it's just a Linear layer
        # Path will be: "feature.weight" and "feature.bias"
        self.feature = nn.Linear(3, 64)

        # This represents the "classifier" field
        # Path will be: "classifier.weight" and "classifier.bias"
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.feature(x)
        x = torch.relu(x)
        x = self.classifier(x)
        return x


def generate_enum_variant_mismatch_test():
    """Generate test file demonstrating enum variant path mismatch."""
    model = SimpleModel()

    # Initialize with some deterministic weights for testing
    torch.manual_seed(42)
    for param in model.parameters():
        param.data.normal_(0, 0.1)

    # Save the state dict
    # PyTorch paths: "feature.weight", "feature.bias", "classifier.weight", "classifier.bias"
    # Burn paths:    "feature.BaseConv.weight", "feature.BaseConv.bias", ...
    #                        ^^^^^^^^ enum variant is missing in PyTorch
    torch.save(model.state_dict(), "model_without_enum_variants.pt")

    print("Generated: model_without_enum_variants.pt")
    print("\nPyTorch state dict keys:")
    for key in model.state_dict().keys():
        shape = tuple(model.state_dict()[key].shape)
        print(f"  {key}: {shape}")

    print("\nExpected Burn paths (with enum variant):")
    print("  feature.BaseConv.weight: (3, 64)")
    print("  feature.BaseConv.bias: (64,)")
    print("  classifier.weight: (64, 10)")
    print("  classifier.bias: (10,)")

    print("\n⚠️  Notice: Burn includes 'BaseConv' enum variant, PyTorch doesn't!")


if __name__ == "__main__":
    generate_enum_variant_mismatch_test()
