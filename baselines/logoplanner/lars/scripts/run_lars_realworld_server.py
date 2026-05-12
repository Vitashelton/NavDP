#!/usr/bin/env python
"""LARS real-world server wrapper.

Wraps around the original LoGoPlanner real-world server to add LARS
safety adaptation WITHOUT modifying the original server code.

Architecture:
    LeKiwi -> LARS Wrapper (this server, port 19998)
            -> LoGoPlanner Server (port 19999) [unmodified]
            -> LARS Runtime adapts commands
            -> Response sent back to LeKiwi

Usage:
    # Terminal 1: Start the original LoGoPlanner server
    python logoplanner_realworld_server.py --port 19999

    # Terminal 2: Start the LARS wrapper
    python run_lars_realworld_server.py --lars-config configs/lars.yaml \
        --safety-config configs/safety.yaml --upstream-url http://localhost:19999
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from io import BytesIO

import requests
from flask import Flask, request, jsonify
from PIL import Image
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="LARS real-world server wrapper"
    )
    parser.add_argument("--port", type=int, default=19998,
                        help="Port for this LARS wrapper server")
    parser.add_argument("--upstream-url", type=str, default="http://localhost:19999",
                        help="URL of the original LoGoPlanner server")
    parser.add_argument("--lars-config", type=str, default="configs/lars.yaml",
                        help="Path to LARS runtime config")
    parser.add_argument("--safety-config", type=str, default="configs/safety.yaml",
                        help="Path to safety shield config")
    parser.add_argument("--model", type=str, default=None,
                        help="Override risk model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model inference")
    parser.add_argument("--disable-lars", action="store_true",
                        help="Passthrough mode: forward without LARS adaptation")
    return parser.parse_args()


def create_app(args, lars_runtime=None):
    app = Flask(__name__)
    app.config["lars_runtime"] = lars_runtime
    app.config["args"] = args

    # Store upstream session: intrinsic, stop_threshold, batch_size
    app.config["upstream_session"] = {
        "intrinsic": None,
        "stop_threshold": -1.0,
        "batch_size": 1,
    }

    @app.route("/navigator_reset", methods=["POST"])
    def navigator_reset():
        """Forward reset to upstream and initialize LARS."""
        data = request.get_json()
        intrinsic = data.get("intrinsic")
        stop_threshold = data.get("stop_threshold", -1.0)
        batch_size = data.get("batch_size", 1)

        app.config["upstream_session"] = {
            "intrinsic": intrinsic,
            "stop_threshold": stop_threshold,
            "batch_size": batch_size,
        }

        # Forward to upstream
        try:
            resp = requests.post(
                f"{args.upstream_url}/navigator_reset",
                json=data,
                timeout=50,
            )
            resp.raise_for_status()
        except Exception as e:
            return jsonify({"error": f"Upstream reset failed: {e}"}), 502

        # Reset LARS state
        runtime = app.config["lars_runtime"]
        if runtime is not None:
            runtime.reset()

        return jsonify({"algo": "logoplanner+lars", "upstream": resp.json()})

    @app.route("/pointgoal_step", methods=["POST"])
    def pointgoal_step():
        """Forward step to upstream, then apply LARS to returned commands."""
        args_local = app.config["args"]
        runtime = app.config["lars_runtime"]

        # Forward the request to the upstream LoGoPlanner server
        # We need to rebuild the multipart form data
        image_file = request.files.get("image")
        depth_file = request.files.get("depth")
        goal_data_str = request.form.get("goal_data")

        if not all([image_file, depth_file, goal_data_str]):
            return jsonify({"error": "Missing image, depth, or goal_data"}), 400

        # Read file contents before forwarding
        image_data = image_file.read()
        depth_data = depth_file.read()

        # Forward to upstream server
        try:
            upstream_resp = requests.post(
                f"{args_local.upstream_url}/pointgoal_step",
                files={
                    "image": ("color.png", BytesIO(image_data), "image/jpeg"),
                    "depth": ("depth.png", BytesIO(depth_data), "image/png"),
                },
                data={"goal_data": goal_data_str},
                timeout=120,
            )
            upstream_resp.raise_for_status()
            upstream_result = upstream_resp.json()
        except Exception as e:
            return jsonify({"error": f"Upstream step failed: {e}"}), 502

        cmd_list = upstream_result.get("cmd_list", [])

        # If LARS disabled, just pass through
        if args_local.disable_lars or runtime is None:
            return jsonify({"cmd_list": cmd_list, "lars_active": False})

        # --- LARS Adaptation ---
        # Decode depth image from the request for scan generation
        depth_pil = Image.open(BytesIO(depth_data)).convert("I")
        depth_np = np.asarray(depth_pil).astype(np.float32)

        # Clean depth similar to original server
        depth_np[np.isnan(depth_np)] = 0
        depth_np[np.isinf(depth_np)] = 0
        depth_np[depth_np > 10000] = 0
        depth_np = depth_np / 1000.0  # mm -> meters

        # Parse goal heading
        try:
            goal_data = json.loads(goal_data_str)
            gx, gy = goal_data.get("goal_x", [0])[0], goal_data.get("goal_y", [0])[0]
            goal_heading = float(np.arctan2(gy, gx))
        except Exception:
            goal_heading = 0.0

        # Apply LARS to each velocity command in the list
        adapted_cmd_list = []
        for cmd in cmd_list:
            if len(cmd) < 3:
                adapted_cmd_list.append(cmd)
                continue

            raw_cmd = np.array(cmd[:3], dtype=np.float32)
            final_action, info = runtime.step(
                policy_output=raw_cmd,
                depth_image=depth_np,
                goal_heading=goal_heading,
            )
            adapted_cmd_list.append(final_action.tolist())

        return jsonify({
            "cmd_list": adapted_cmd_list,
            "lars_active": True,
            "risk_score": float(info.get("risk_score", 0.0)),
        })

    return app


def main():
    args = parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load LARS config
    lars_config_path = os.path.join(base, args.lars_config)
    if not os.path.exists(lars_config_path):
        lars_config_path = args.lars_config

    lars_runtime = None

    if not args.disable_lars:
        sys.path.insert(0, base)
        from lars.lars_runtime import LARSRuntime

        lars_runtime = LARSRuntime.from_config(lars_config_path)

        if args.safety_config:
            safety_config_path = os.path.join(base, args.safety_config)
            if os.path.exists(safety_config_path):
                lars_runtime.load_safety_config(safety_config_path)

        if args.model:
            lars_runtime.load_model(args.model)

        lars_runtime.residual_adapter.set_device(args.device)
        print(f"[LARS] Runtime initialized with config: {lars_config_path}")
        print(f"[LARS] Device: {args.device}")
        print(f"[LARS] Risk model: {'loaded' if lars_runtime.risk_model else 'none (rule-only mode)'}")
    else:
        print("[LARS] LARS disabled — passthrough mode")

    app = create_app(args, lars_runtime)

    print(f"\n[LARS] Wrapper server starting on port {args.port}")
    print(f"[LARS] Forwarding to upstream: {args.upstream_url}")
    print(f"[LARS] Point LeKiwi at: http://<host>:{args.port}")
    print()

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
