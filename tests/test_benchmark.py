"""Ensure that the locomotion gate cannot pass a stationary policy."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from corgi_rl.benchmark import benchmark
from corgi_rl.spec import load_spec, write_json


def test_stationary_policy_fails_motion_gate(tmp_path):
    spec = load_spec()
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["actions"], value=numpy_helper.from_array(
            np.zeros((1, 12), dtype=np.float32)))],
        "fixed-home-baseline", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 45])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    policy = tmp_path / "policy.onnx"
    onnx.save(model, policy)
    write_json(tmp_path / "manifest.json", {**spec.manifest(), "seed": 0})
    result = benchmark(spec, policy=policy, seeds=[7], output=tmp_path / "result")
    checks = result["episodes"][0]["checks"]
    assert checks["no_fall"] and checks["stop_speed"]
    assert not checks["forward_motion"] and not checks["both_turn_directions"]
    assert result["pass_count"] == 0
