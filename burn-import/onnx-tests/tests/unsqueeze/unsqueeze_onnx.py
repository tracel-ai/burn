from pathlib import Path
from typing import List, Optional, Tuple
from numpy.typing import ArrayLike, NDArray
import onnx
import onnxruntime
import numpy as np
from dataclasses import dataclass


def validate_sequence(arr: List | Tuple) -> None:
    el_type = type(arr[0])
    for el in arr:
        if type(el) != el_type:
            raise ValueError(
                f"Expected all elements to be of type {el_type} but got {type(el)}"
            )


# extracting because match statement doesn't work nested
def array_handler(k: str, arr: np.ndarray) -> onnx.TensorProto:
    match arr.dtype:
        case np.float32:
            return onnx.helper.make_tensor_value_info(
                k, onnx.TensorProto.FLOAT, arr.shape
            )
        case np.float64:
            return onnx.helper.make_tensor_value_info(
                k, onnx.TensorProto.DOUBLE, arr.shape
            )

        case np.int32:
            return onnx.helper.make_tensor_value_info(
                k, onnx.TensorProto.INT32, arr.shape
            )
        case np.int64:
            return onnx.helper.make_tensor_value_info(
                k, onnx.TensorProto.INT64, arr.shape
            )

        case _:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")


def make_onnx_types(k: str, v: ArrayLike) -> onnx.TensorProto | onnx.ValueInfoProto:
    print(type(v))
    match v:
        case np.ndarray():
            return array_handler(k, v)  # type: ignore
        case list() | tuple():
            validate_sequence(v)  # type: ignore
            return array_handler(k, np.array(v))
        case int():
            return onnx.helper.make_value_info(k, onnx.ValueInfoProto.INT64)
        case float():
            return onnx.helper.make_value_info(k, onnx.ValueInfoProto.FLOAT)
        case bool():
            return onnx.helper.make_value_info(k, onnx.ValueInfoProto.BOOL)
        case _:
            raise ValueError(f"Unsupported type: {type(v)}")


@dataclass
class OnnxOpData:
    name: str
    inputs: dict[str, ArrayLike]
    output: dict[str, ArrayLike]

    @property
    def input_names(self) -> List[str]:
        return list(self.inputs.keys())

    @property
    def output_names(self) -> List[str]:
        return list(self.output.keys())

    @property
    def graph_inputs(self) -> List[onnx.TensorProto | onnx.ValueInfoProto]:
        return [make_onnx_types(k, v) for k, v in self.inputs.items()]

    @property
    def graph_outputs(self) -> List[onnx.TensorProto | onnx.ValueInfoProto]:
        return [make_onnx_types(k, v) for k, v in self.output.items()]

    def make_onnx_graph(self, op_name: Optional[str] = None) -> onnx.GraphProto:
        """Create a graph with a single node for testing.

        Args:
            op_inputs: The input tensor to the node."""
        node = onnx.helper.make_node(
            op_name or self.name,  # you learn something new every day
            inputs=self.input_names,
            outputs=self.output_names,
        )

        graph: onnx.GraphProto = onnx.helper.make_graph(
            [node],
            f"{self.name}_test",
            self.graph_inputs,
            outputs=self.graph_outputs,
        )

        return graph

    def save_model(self, path: Optional[Path | str] = None):
        model = onnx.helper.make_model(self.make_onnx_graph())
        if path and Path(path).suffix != ".onnx":
            raise ValueError(
                f"Provide path {path} must include the model name and end with .onnx extension"
            )
        onnx.save(model, str(path) if path else f"{self.name.lower()}.onnx")
        print(f"Model saved to {path}")

    def validate_model(self):
        sess = onnxruntime.InferenceSession(f"{self.name.lower()}.onnx")
        sess_inputs = [inp.name for inp in sess.get_inputs()]
        outputs = sess.run(self.output_names, self.inputs)
        for i, k in enumerate(self.output.keys()):
            assert np.allclose(self.output[k], outputs[i])
        print("Output is the same as expected. Test passed.")
        return outputs


if __name__ == "__main__":
    axes = [0, 4]
    x = np.array(np.random.randn(3, 4, 5))
    y = np.expand_dims(x, axis=axes)
    if y.shape != (1, 3, 4, 5, 1):
        raise ValueError(f"Expected shape (1,3,4,5,1) but got {y.shape}")

    data = OnnxOpData(
        name="Unsqueeze",
        inputs={"x": x, "axes": axes},
        output={"output": y},
    )
    print(onnx.OperatorProto)
    # graph = data.make_onnx_graph("Unsqueeze")
    data.save_model()
    result = data.validate_model()
    print(result[0].shape)
    # print(data.output["output"])
    # assert np.allclose(result[0], data.output["output"])
    # print("Test passed")
