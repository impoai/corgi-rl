"""Explicit entry points; no implicit long-running training or hardware access."""

import argparse
import json
from pathlib import Path

from .spec import load_spec, RobotSpec, write_json


def main():
    parser = argparse.ArgumentParser(description="Corgi simulation tools (run on remote NVIDIA host)")
    parser.add_argument("--spec", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("model")
    model.add_argument("--output", type=Path, default=Path("models/corgi.xml"))
    training = sub.add_parser("train")
    training.add_argument("--task", choices=("stand", "walk"), default="stand")
    training.add_argument("--output", required=True)
    training.add_argument("--iterations", type=int, default=1000)
    training.add_argument("--num-envs", type=int, default=1024)
    training.add_argument("--steps", type=int, default=24)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--resume", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task", choices=("stand", "walk"), default="stand")
    evaluate.add_argument("--policy", type=Path)
    evaluate.add_argument("--seconds", type=float, default=5.)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--video", type=Path)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--policy", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 111)))
    bench.add_argument("--video-seed", type=int)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.command == "model":
        from .model import mjcf
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(mjcf(spec))
        write_json(args.output.with_suffix(".schema.json"), RobotSpec.model_json_schema())
        print(args.output)
    elif args.command == "train":
        from .train import train
        kwargs = vars(args).copy()
        del kwargs["command"], kwargs["spec"]
        print(json.dumps(train(spec, **kwargs), indent=2))
    elif args.command == "export":
        from .export import export_checkpoint
        print(json.dumps(export_checkpoint(spec, checkpoint=args.checkpoint,
                                           output=args.output), indent=2))
    elif args.command == "benchmark":
        from .benchmark import benchmark
        result = benchmark(spec, policy=args.policy, seeds=args.seeds,
                           output=args.output, video_seed=args.video_seed)
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    else:
        from .native import evaluate
        kwargs = vars(args).copy()
        del kwargs["command"], kwargs["spec"]
        print(json.dumps(evaluate(spec, **kwargs), indent=2))


if __name__ == "__main__":
    main()
