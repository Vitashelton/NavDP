#!/usr/bin/env python
"""Collect LARS logs during real-world deployment or simulation.

This script wraps around the LoGoPlanner real-world server and records
depth images, policy outputs, and LARS-adapted actions for later
dataset construction.

Usage:
    python collect_lars_log.py --server-url http://localhost:19999 \
        --goal-x 5.0 --goal-y 0.0 --output-dir logs/
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from io import BytesIO

import requests
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect LARS log data from LoGoPlanner server"
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:19999",
                        help="LoGoPlanner server URL")
    parser.add_argument("--lars-server-url", type=str, default="http://localhost:19998",
                        help="LARS wrapper server URL (if using lars wrapper)")
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="logs")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--send-fps", type=float, default=7.0)
    parser.add_argument("--use-lars", action="store_true",
                        help="Send to LARS wrapper instead of raw server")
    parser.add_argument("--mock-depth", action="store_true",
                        help="Use mock depth instead of camera")
    parser.add_argument("--intrinsic", type=str,
                        default="[[615.0,0,320],[0,615.0,240],[0,0,1]]",
                        help="Camera intrinsic matrix JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    server_url = args.lars_server_url if args.use_lars else args.server_url

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(
        args.output_dir, f"lars_collect_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    print(f"Collecting LARS logs to {log_path}")
    print(f"Server: {server_url}")

    # Initialize server
    intrinsic = json.loads(args.intrinsic)
    reset_data = {
        "intrinsic": intrinsic,
        "stop_threshold": -1.0,
        "batch_size": 1,
    }
    try:
        resp = requests.post(
            f"{server_url}/navigator_reset", json=reset_data, timeout=50
        )
        resp.raise_for_status()
        print(f"Server initialized: {resp.json()}")
    except Exception as e:
        print(f"Server init failed: {e}")
        sys.exit(1)

    goal_data = {"goal_x": [args.goal_x], "goal_y": [args.goal_y]}
    send_interval = 1.0 / args.send_fps

    with open(log_path, "w") as f:
        for step in range(args.max_steps):
            t_start = time.time()

            if args.mock_depth:
                depth_img = np.random.randint(
                    0, 8000, (480, 640), dtype=np.uint16
                )
                color_img = np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8
                )
            else:
                # In a real deployment, capture from Realsense here
                print("[WARN] No real camera; use --mock-depth for testing")
                depth_img = np.zeros((480, 640), dtype=np.uint16)
                color_img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Send to server
            color_pil = Image.fromarray(color_img)
            depth_pil = Image.fromarray(depth_img, mode="I;16")

            color_buf = BytesIO()
            color_pil.save(color_buf, format="jpeg")
            color_buf.seek(0)

            depth_buf = BytesIO()
            depth_pil.save(depth_buf, format="PNG")
            depth_buf.seek(0)

            try:
                resp = requests.post(
                    f"{server_url}/pointgoal_step",
                    files={
                        "image": ("color.png", color_buf, "image/jpeg"),
                        "depth": ("depth.png", depth_buf, "image/png"),
                    },
                    data={"goal_data": json.dumps(goal_data)},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()

                entry = {
                    "timestamp": time.time(),
                    "step": step,
                    "cmd_list": result.get("cmd_list", []),
                    "depth_min": float(depth_img[depth_img > 0].min())
                    if np.any(depth_img > 0) else 0.0,
                    "depth_max": float(depth_img.max()),
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()

                print(f"Step {step}: received {len(entry['cmd_list'])} commands")

            except Exception as e:
                print(f"Step {step} failed: {e}")
                continue
            finally:
                color_buf.close()
                depth_buf.close()

            elapsed = time.time() - t_start
            time.sleep(max(0, send_interval - elapsed))

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
