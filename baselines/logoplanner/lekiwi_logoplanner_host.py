#!/usr/bin/env python

import threading
import queue
import requests
import json
import time
import argparse
from io import BytesIO

import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image

# Import LeKiwi modules from your original code
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwi

class LeKiwiClient:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.cmd_queue = queue.Queue()
        self.pipeline = None
        self.intrinsic_matrix = None
        self.align = None

        # Debug print timers
        self.last_rgbd_print_time = 0.0
        self.last_cmd_print_time = 0.0
        self.last_safety_debug_time = 0.0

        # Latest depth cache for local safety layer
        self.latest_depth_mm = None
        self.depth_lock = threading.Lock()



    def init_realsense(self):
        """Initialize Realsense pipeline and get intrinsic parameters"""
        print("Initializing Realsense...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams (match server's expected resolution)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        try:
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)  # Align depth to color
        except Exception as e:
            print(f"Realsense initialization failed: {e}")
            return False
        
        # Get intrinsic parameters (required for server initialization)
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get color frame for intrinsics")
            
            intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.intrinsic_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ])
            print(f"Realsense Intrinsics:\n{self.intrinsic_matrix}")
            return True
        except Exception as e:
            print(f"Failed to get Realsense intrinsics: {e}")
            self.pipeline.stop()
            return False
    def _save_safety_roi_debug(self, color_img, depth_img):
        vis = color_img.copy()
        h, w = depth_img.shape

        band_y1 = int(h * 0.34)
        band_y2 = int(h * 0.44)

        left_x1, left_x2 = int(w * 0.10), int(w * 0.35)
        front_x1, front_x2 = int(w * 0.38), int(w * 0.62)
        right_x1, right_x2 = int(w * 0.65), int(w * 0.90)

        rois = {
            "left": (left_x1, band_y1, left_x2, band_y2),
            "front": (front_x1, band_y1, front_x2, band_y2),
            "right": (right_x1, band_y1, right_x2, band_y2),
        }

        for name, (x1, y1, x2, y2) in rois.items():
            roi = depth_img[y1:y2, x1:x2]
            valid = roi[(roi > 250) & (roi < 4000)]

            if len(valid) > 0:
                p20 = int(np.percentile(valid, 20))
                p50 = int(np.percentile(valid, 50))
                close_ratio = float(np.sum((roi > 250) & (roi < 700)) / roi.size)
                text = f"{name}: p20={p20}, p50={p50}, close={close_ratio:.2f}"
            else:
                text = f"{name}: no valid depth"

            color = (0, 255, 255) if name == "front" else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite("band_safety_debug.jpg", vis)

        # close mask only inside the band
        mask_vis = color_img.copy()
        band_mask = np.zeros(depth_img.shape, dtype=bool)
        band_mask[band_y1:band_y2, :] = True
        close_mask = (depth_img > 250) & (depth_img < 700) & band_mask

        overlay = mask_vis.copy()
        overlay[close_mask] = (0, 0, 255)
        blended = cv2.addWeighted(mask_vis, 0.65, overlay, 0.35, 0)
        cv2.imwrite("band_close_mask_debug.jpg", blended)

    def communication_worker(self):
        """Thread: Capture Realsense data, send to server, receive command lists"""
        # First initialize server with navigator_reset
        if not self._init_server():
            self.stop_event.set()
            return

        # Prepare goal data (batch_size compatible)
        goal_data = {
            "goal_x": [self.args.goal_x] * self.args.batch_size,
            "goal_y": [self.args.goal_y] * self.args.batch_size
        }

        send_interval = 1.0 / self.args.send_fps  # Control send frequency
        last_send_time = time.time()

        while not self.stop_event.is_set():
            # Control send frequency
            current_time = time.time()
            if current_time - last_send_time < send_interval:
                time.sleep(0.001)
                continue
            last_send_time = current_time

            # Capture Realsense frames
            color_img, depth_img = self._capture_realsense_frames()
            if color_img is None or depth_img is None:
                continue

            # Convert to PIL images (server-compatible format)
            pil_color = self._convert_color_to_pil(color_img)
            pil_depth = self._convert_depth_to_pil(depth_img)
            frame_capture_time = time.time()

            # Send to server and get commands
            cmd_list = self._send_to_server(pil_color, pil_depth, goal_data)
            print(f"Received {len(cmd_list)} commands from server")
            send_time = time.time()
            print(f"Server communication costs {send_time - frame_capture_time:.3f}s")
            if cmd_list:
                self._update_cmd_queue(cmd_list)

    def _init_server(self):
        """Send initial reset request to server"""
        print(f"Initializing server: {self.args.server_url}/navigator_reset")
        reset_data = {
            "intrinsic": self.intrinsic_matrix.tolist(),
            "stop_threshold": self.args.stop_threshold,
            "batch_size": self.args.batch_size
        }
        
        try:
            response = requests.post(
                f"{self.args.server_url}/navigator_reset",
                json=reset_data,
                timeout=50
            )
            response.raise_for_status()
            print(f"Server initialized: {response.json()}")
            return True
        except Exception as e:
            print(f"Server initialization failed: {e}")
            return False

    def _save_close_mask_debug(self, color_img, depth_img):
        vis = color_img.copy()

        close_mask = (depth_img > 250) & (depth_img < 650)

        overlay = vis.copy()
        overlay[close_mask] = (0, 0, 255)  # red close pixels

        blended = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        cv2.imwrite("close_mask_debug.jpg", blended)
    
    def _capture_realsense_frames(self):
        """Capture and preprocess aligned Realsense frames"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)

            # Align depth to color
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("No Realsense frames received")
                return None, None

            color_img = np.asanyarray(color_frame.get_data())  # BGR
            depth_img = np.asanyarray(depth_frame.get_data())  # uint16, aligned depth
            self._save_safety_roi_debug(color_img, depth_img)
            self._save_close_mask_debug(color_img, depth_img)


            with self.depth_lock:
                self.latest_depth_mm = depth_img.copy()

            valid = depth_img > 0
            valid_ratio = float(np.mean(valid))

            if np.any(valid):
                depth_min = int(depth_img[valid].min())
                depth_mean = int(depth_img[valid].mean())
                depth_max = int(depth_img[valid].max())
            else:
                depth_min = depth_mean = depth_max = 0

            h, w = depth_img.shape
            cy, cx = h // 2, w // 2
            patch = depth_img[cy - 5:cy + 6, cx - 5:cx + 6]
            patch_valid = patch[patch > 0]
            center_depth = int(np.median(patch_valid)) if len(patch_valid) > 0 else 0

            now = time.time()
            if now - self.last_rgbd_print_time > 1.0:
                print(
                    f"[RGBD] depth center={center_depth}mm, "
                    f"valid={valid_ratio:.3f}, min={depth_min}, mean={depth_mean}, max={depth_max}"
                )
                self.last_rgbd_print_time = now


            return color_img, depth_img

        except Exception as e:
            print(f"Frame capture failed: {e}")
            return None, None


    @staticmethod
    def _convert_color_to_pil(color_img):
        """Convert BGR numpy array to RGB PIL Image"""
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)

    @staticmethod
    def _convert_depth_to_pil(depth_img):
        """Convert Z16 depth array to 16-bit PIL Image (server-compatible)"""
        return Image.fromarray(depth_img.astype(np.uint16), mode='I;16')

    def _send_to_server(self, pil_color, pil_depth, goal_data):
        """Send multipart request to server and return command list"""
        # Prepare image buffers
        color_buf = BytesIO()
        pil_color.save(color_buf, format='jpeg')
        color_buf.seek(0)

        depth_buf = BytesIO()
        pil_depth.save(depth_buf, format='PNG')
        depth_buf.seek(0)

        # Prepare request data
        files = {
            'image': ('color.png', color_buf, 'image/jpeg'),
            'depth': ('depth.png', depth_buf, 'image/png')
        }
        data = {'goal_data': json.dumps(goal_data)}

        try:
            response = requests.post(
                f"{self.args.server_url}/pointgoal_step",
                files=files,
                data=data,
                timeout=1000000
            )
            response.raise_for_status()
            result = response.json()
            return result.get('cmd_list', [])
        except Exception as e:
            print(f"Server communication failed: {e}")
            return []
        finally:
            color_buf.close()
            depth_buf.close()

    def _update_cmd_queue(self, cmd_list):
        """Update command queue - keep only first few commands from latest plan"""
        if not cmd_list:
            return

        cmd_list = cmd_list[: self.args.max_plan_steps]

        try:
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    break

            self.cmd_queue.put_nowait(cmd_list)
            print(f"Updated queue with latest {len(cmd_list)} commands")
        except Exception as e:
            print(f"Command queue update failed: {e}")

    def execution_worker(self):
        """Thread: Execute commands from queue on LeKiwi robot - prioritize LATEST commands"""

        robot = None

        if self.args.dry_run:
            print("[DRY RUN] Robot connection disabled. Commands will only be printed.")
        else:
            robot_config = LeKiwiConfig()
            print("[INFO] Original LeKiwi cameras:", robot_config.cameras)

            robot_config.cameras = {}

            print("[INFO] Disabled LeKiwi cameras:", robot_config.cameras)

            robot = LeKiwi(robot_config)

            try:
                print("Connecting to LeKiwi robot...")
                robot.connect()
                print("LeKiwi connected successfully")
            except Exception as e:
                print(f"LeKiwi connection failed: {e}")
                self.stop_event.set()
                return

        last_cmd_time = time.time()
        watchdog_active = False
        current_cmds = []  # Current command list to execute
        cmd_idx = 0        # Current position in command list

        while not self.stop_event.is_set():
            loop_start = time.time()

            # Step 1: Check for NEW command list (highest priority)
            new_cmds = self._get_latest_cmds()
            if new_cmds:
                # Abort old commands - switch to new list immediately
                current_cmds = new_cmds
                cmd_idx = 0  # Reset to start of new command list
                last_cmd_time = time.time()
                watchdog_active = False
                print(f"Switched to new command list (size: {len(current_cmds)})")

            # Step 2: Execute next command in current list (if available)
            if current_cmds and cmd_idx < len(current_cmds):
                cmd = current_cmds[cmd_idx]
                self._execute_cmd(robot, cmd)
                cmd_idx += 1  # Move to next command in list
                last_cmd_time = time.time()
                watchdog_active = False
            else:
                # Step 3: Watchdog - stop robot if no commands for timeout
                if (time.time() - last_cmd_time > self.args.watchdog_timeout / 1000 
                    and not watchdog_active):
                    print("Watchdog timeout - stopping robot")
                    robot.stop_base()
                    watchdog_active = True

            # Control loop frequency (maintain exec_freq)
            elapsed = time.time() - loop_start
            time.sleep(max(1/self.args.exec_freq - elapsed, 0.001))

        if robot is not None:
            print("Stopping robot...")
            robot.stop_base()
            robot.disconnect()
            print("LeKiwi disconnected")


    def _get_latest_cmds(self):
        """Get latest command list (discard intermediate lists if multiple)"""
        latest_cmds = None
        try:
            # Get all commands in queue (only keep the last one)
            while not self.cmd_queue.empty():
                latest_cmds = self.cmd_queue.get_nowait()
            return latest_cmds
        except queue.Empty:
            return None
        
    def _apply_depth_safety(self, x_vel, y_vel, theta_vel):
        """
        Simpler and more robust band-based safety.

        LeKiwi frame:
        x_vel > 0: left
        x_vel < 0: right
        y_vel > 0: forward
        y_vel < 0: backward
        """

        # Only intervene when moving forward
        if y_vel <= 0:
            return x_vel, y_vel, theta_vel

        with self.depth_lock:
            if self.latest_depth_mm is None:
                return x_vel, y_vel, theta_vel
            depth = self.latest_depth_mm.copy()

        h, w = depth.shape

        # -----------------------------
        # Key idea:
        # only use a narrow horizontal band
        # instead of a large rectangle.
        # Tune these two numbers later.
        # -----------------------------
        band_y1 = int(h * 0.34)
        band_y2 = int(h * 0.44)

        # Three sectors in this band
        left_x1, left_x2 = int(w * 0.10), int(w * 0.35)
        front_x1, front_x2 = int(w * 0.38), int(w * 0.62)
        right_x1, right_x2 = int(w * 0.65), int(w * 0.90)

        left_band = depth[band_y1:band_y2, left_x1:left_x2]
        front_band = depth[band_y1:band_y2, front_x1:front_x2]
        right_band = depth[band_y1:band_y2, right_x1:right_x2]

        def band_stats(band):
            valid = band[(band > 250) & (band < 4000)]
            if len(valid) == 0:
                return {
                    "p20": 9999.0,
                    "p50": 9999.0,
                    "close_ratio": 0.0,
                    "valid_ratio": 0.0,
                }

            total = band.size
            return {
                "p20": float(np.percentile(valid, 20)),
                "p50": float(np.percentile(valid, 50)),
                "close_ratio": float(np.sum((band > 250) & (band < 700)) / total),
                "valid_ratio": float(len(valid) / total),
            }

        left = band_stats(left_band)
        front = band_stats(front_band)
        right = band_stats(right_band)

        now = time.time()
        if now - self.last_safety_debug_time > 0.5:
            print(
                f"[BAND SAFETY DEBUG] "
                f"front p20={front['p20']:.0f}, p50={front['p50']:.0f}, close={front['close_ratio']:.3f}; "
                f"left p50={left['p50']:.0f}, right p50={right['p50']:.0f}"
            )
            self.last_safety_debug_time = now

        # Safer thresholds
        hard_close_ratio_th = 0.18
        slow_close_ratio_th = 0.08
        hard_dist_mm = 650.0
        slow_dist_mm = 1100.0

        # Hard stop / sidestep
        if front["close_ratio"] > hard_close_ratio_th and front["p20"] < hard_dist_mm:
            y_vel = 0.0

            side_speed = min(self.args.max_v, 0.08)

            # Choose freer side by comparing median distance
            if left["p50"] > right["p50"]:
                x_vel = side_speed      # move left
                theta_vel = 0.12
                side = "LEFT"
            else:
                x_vel = -side_speed     # move right
                theta_vel = -0.12
                side = "RIGHT"

            print(
                f"[BAND SAFETY] HARD obstacle. "
                f"front_close={front['close_ratio']:.3f}, front_p20={front['p20']:.0f}. "
                f"Override: move {side}"
            )

        # Soft slow-down
        elif front["close_ratio"] > slow_close_ratio_th and front["p20"] < slow_dist_mm:
            scale = (front["p20"] - hard_dist_mm) / (slow_dist_mm - hard_dist_mm)
            scale = float(np.clip(scale, 0.25, 1.0))
            y_vel *= scale

            print(
                f"[BAND SAFETY] SLOW. "
                f"front_close={front['close_ratio']:.3f}, front_p20={front['p20']:.0f}, scale={scale:.2f}"
            )

        return x_vel, y_vel, theta_vel


    def _execute_cmd(self, robot, cmd):
        """Execute single velocity command from LoGoPlanner and map it to LeKiwi base frame."""
        planner_vx, planner_vy, planner_wz = cmd

        # LoGoPlanner -> LeKiwi
        # LeKiwi:
        #   x.vel: +left, -right
        #   y.vel: +forward, -backward
        #   theta.vel: +CCW, -CW
        lekiwi_x = planner_vx
        lekiwi_y = -planner_vy
        lekiwi_wz = planner_wz
        # Optional gains
        linear_gain = 1.0
        angular_gain = 2.0   # turn was too small, amplify it first

        lekiwi_x = float(np.clip(lekiwi_x, -self.args.max_v, self.args.max_v))
        lekiwi_y = float(np.clip(lekiwi_y, -self.args.max_v, self.args.max_v))
        lekiwi_wz = float(np.clip(lekiwi_wz, -self.args.max_w, self.args.max_w))

        lekiwi_x, lekiwi_y, lekiwi_wz = self._apply_depth_safety(
            lekiwi_x, lekiwi_y, lekiwi_wz
        )
        now = time.time()
        if now - self.last_cmd_print_time > 0.5:
            print(
                f"Planner cmd: vx={planner_vx:.3f}, vy={planner_vy:.3f}, wz={planner_wz:.3f} "
                f"=> LeKiwi cmd: x.vel={lekiwi_x:.3f}, y.vel={lekiwi_y:.3f}, theta.vel={lekiwi_wz:.3f}"
            )
            self.last_cmd_print_time = now


        if self.args.dry_run or robot is None:
            return

        base_action = {
            'x.vel': lekiwi_x,
            'y.vel': lekiwi_y,
            'theta.vel': lekiwi_wz,
        }

        arm_action = {
            'arm_shoulder_pan.pos': -4.618768328445739,
            'arm_shoulder_lift.pos': -90.0,
            'arm_elbow_flex.pos': 0.0,
            'arm_wrist_flex.pos': 90.744630071599047,
            'arm_wrist_roll.pos': 0.890109890109898,
            'arm_gripper.pos': 0.0
        }

        try:
            robot.send_action({**base_action, **arm_action})
            time.sleep(self.args.cmd_exec_delay)
        except Exception as e:
            print(f"Command execution failed: {e}")


    def run(self):
        """Start client and run until stopped"""
        # Initialize Realsense first (critical dependency)
        if not self.init_realsense():
            print("Failed to initialize Realsense - exiting")
            return

        # Start worker threads
        comm_thread = threading.Thread(target=self.communication_worker, daemon=True)
        exec_thread = threading.Thread(target=self.execution_worker, daemon=True)

        comm_thread.start()
        exec_thread.start()
        print("Client started. Press Ctrl+C to stop.")

        # Main thread: Wait for stop signal (Ctrl+C)
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping client...")
            self.stop_event.set()

        # Cleanup resources
        comm_thread.join(timeout=2.0)
        exec_thread.join(timeout=2.0)
        self.pipeline.stop()
        print("Client stopped cleanly")

def parse_args():
    parser = argparse.ArgumentParser(description="LeKiwi Client with Realsense + Server Communication")
    
    # Server configuration
    parser.add_argument("--server-url", type=str, default="http://192.168.1.100:8888",
                        help="Flask server URL (e.g., http://192.168.1.100:8888)")
    parser.add_argument("--send-fps", type=int, default=7, help="Frame send frequency to server")
    
    # Navigation goals
    parser.add_argument("--goal-x", type=float, default=10.0, help="Goal X position (meters)")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal Y position (meters)")
    parser.add_argument("--stop-threshold", type=float, default=-1.0, help="Server stop threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Server batch size (must match)")
    
    # Robot control
    parser.add_argument("--cmd-exec-delay", type=float, default=0.1,
                        help="Time to execute each velocity command (seconds)")
    parser.add_argument("--exec-freq", type=int, default=50, help="Command execution loop frequency")
    parser.add_argument("--watchdog-timeout", type=int, default=100000,
                        help="Watchdog timeout (milliseconds) - stops robot if no commands")
    parser.add_argument("--dry-run", action="store_true",
                    help="Do not connect to LeKiwi or send motor commands; only print commands")

    parser.add_argument("--max-plan-steps", type=int, default=3,
                        help="Only execute/inspect first N commands from each predicted command list")

    parser.add_argument("--max-v", type=float, default=0.12,
                        help="Max linear velocity sent to LeKiwi")

    parser.add_argument("--max-w", type=float, default=0.35,
                        help="Max angular velocity sent to LeKiwi")

    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    client = LeKiwiClient(args)
    client.run()
