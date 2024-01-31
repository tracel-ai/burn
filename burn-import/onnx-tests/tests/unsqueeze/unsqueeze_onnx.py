from typing import List, Optional
from numpy.typing import ArrayLike
import onnx
import onnxruntime
import numpy as np
from dataclasses import dataclass


@dataclass
class OnnxOpData:
    inputs: dict[str, ArrayLike]
    output: dict[str, ArrayLike]

    @property
    def input_names(self) -> List[str]:
        return list(self.inputs.keys())

    @property
    def output_names(self) -> List[str]:
        return list(self.output.keys())

    @property
    def input_values(self) -> List[onnx.TensorProto | onnx.ValueInfoProto]:
        values = []
        for k, v in self.inputs.items():
            match type(v):
                case np.ndarray:
                    match v.dtype:
                        case np.floating:
                            values.append(
                                onnx.helper.make_tensor_value_info(
                                    k, onnx.TensorProto.FLOAT, v.shape
                                )
                            )
                        case np.integer:
                            values.append(
                                onnx.helper.make_tensor_value_info(
                                    k, onnx.TensorProto.INT64, v.shape
                                )
                            )
                        case _:
                            raise ValueError(f"Unsupported dtype: {v.dtype}")
                case int:
                    values.append(
                        onnx.helper.make_value_info(k, onnx.ValueInfoProto.INT64)
                    )
                case float:
                    values.append(
                        onnx.helper.make_value_info(k, onnx.ValueInfoProto.FLOAT)
                    )
                case bool:
                    values.append(
                        onnx.helper.make_value_info(k, onnx.ValueInfoProto.BOOL)
                    )
                case _:
                    raise ValueError(f"Unsupported type: {type(v)}")
        return values


def make_onnx_graph(
    op_inputs: List[ArrayLike],
    out: ArrayLike,
    op_name: str,
    test_name: Optional[str] = None,
) -> onnx.GraphProto:
    """Create a graph with a single node for testing.

    Args:
        op_inputs: The input tensor to the node."""
    node = onnx.helper.make_node(
        op_name,
        inputs=["x", "axes"],
        outputs=["y"],
    )
    graph = onnx.helper.make_graph(
        [node],
        "unsqueeze_test",
        inputs=[
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4, 5]),
            onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [2]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, 3, 4, 5, 1]
            )
        ],
    )

    return graph
