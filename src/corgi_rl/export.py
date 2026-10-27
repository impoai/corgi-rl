"""Export a saved MLP actor without starting another training environment."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .spec import write_json


def export_checkpoint(spec, *, checkpoint, output):
    checkpoint, out = Path(checkpoint), Path(output)
    manifest = json.loads(checkpoint.with_name("manifest.json").read_text())
    if manifest["spec_sha256"] != spec.digest:
        raise ValueError("Checkpoint specification differs from the requested robot")
    torch.set_num_threads(1)
    obs = TensorDict({"actor": torch.zeros(1, 45)}, batch_size=[1])
    actor_cfg = copy.deepcopy(manifest["ppo"]["actor"])
    actor_cfg = {k: actor_cfg[k] for k in
                 ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")}
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 12, **actor_cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    graph = actor.as_onnx(verbose=False)
    graph.eval()
    out.mkdir(parents=True, exist_ok=False)
    path = out / "policy.onnx"
    torch.onnx.export(graph, graph.get_dummy_inputs(), str(path),
                      export_params=True, opset_version=18,
                      input_names=graph.input_names, output_names=graph.output_names,
                      dynamic_axes={}, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    errors = []
    rng = np.random.default_rng(90210)
    for _ in range(32):
        values = rng.normal(0, .3, (1, 45)).astype(np.float32)
        values[:, 5] = -1.
        with torch.no_grad():
            expected = actor(TensorDict({"actor": torch.from_numpy(values)}, batch_size=[1])).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: values})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.abs(expected - actual).max()))
    manifest["export_source"] = {"checkpoint": str(checkpoint),
                                 "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 "saved_iteration_index": saved["iter"],
                                 "onnx_max_abs_error": max(errors), "parity_samples": 32}
    write_json(out / "manifest.json", manifest)
    write_json(out / "export.json", manifest["export_source"])
    return manifest["export_source"]
