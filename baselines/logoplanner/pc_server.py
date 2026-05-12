import matplotlib
matplotlib.use('Agg')  # 强制无界面后台画图
import matplotlib.pyplot as plt

from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import LoGoPlanner_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os
import torch 
import traceback
from deployment.mpc_controller import Mpc_controller
from deployment.visualization import plot_trajectory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=19999) 
# 🔥 这里不再需要传你自己的垃圾权重了，可以传 'empty.pth' 让它用官方内部逻辑加载，或者直接置空
parser.add_argument("--checkpoint", type=str, default="empty.pth", help="可以忽略，已改用官方模型加载机制")
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None

@app.route("/navigator_reset", methods=['POST'])
def navdp_reset():
    global navdp_navigator, navdp_fps_writer
    data = request.get_json()
    intrinsic = np.array(data.get('intrinsic'))
    threshold = np.array(data.get('stop_threshold'))
    batchsize = np.array(data.get('batch_size'))
    
    if navdp_navigator is None:
        print(f"🧠 [1/2] 正在唤醒官方 LoGoPlanner 通用导航大模型...")
        navdp_navigator = LoGoPlanner_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            # 🔥 核心解封：把 'empty.pth' 替换成 args.checkpoint
            navi_model=args.checkpoint, 
            device='cuda:0'
        )

        navdp_navigator.reset(batchsize, threshold)
    else:
        navdp_navigator.reset(batchsize, threshold)

    if navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        navdp_fps_writer = imageio.get_writer(f"{format_time}_fps_pointgoal.mp4", fps=2)
    
    return jsonify({"algo": "logoplanner"})

@app.route("/navigator_reset_env", methods=['POST'])
def navdp_reset_env():
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo": "logoplanner"})

@app.route("/pointgoal_step", methods=['POST'])
def navdp_step_xy():
    global navdp_navigator, navdp_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navdp_navigator.batch_size
    
    # ================= 图像预处理 =================
    image = Image.open(image_file.stream).convert('RGB')
    image = image.resize((224, 224), Image.BILINEAR) 
    image = np.asarray(image).astype(np.float32) / 255.0 
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    # ================= 深度图预处理 (免疫环境变化的神器) =================
    depth = Image.open(depth_file.stream).convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis].astype(np.float32)
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    depth[depth > 10000] = 0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    from scipy import ndimage
    zero_mask = (depth == 0)
    if np.any(zero_mask):
        window_size = 80
        non_zero_depth = depth.copy()
        non_zero_depth[depth == 0] = np.inf
        min_filtered = ndimage.minimum_filter(non_zero_depth, size=window_size)
        valid_fill_mask = zero_mask & (min_filtered != np.inf)
        depth[valid_fill_mask] = min_filtered[valid_fill_mask]
    
    depth = depth / 1000.0
    depth[depth < 0] = 0
    
    # ================= 大脑思考 =================
    # 官方模型会吐出真正完美的躲避障碍物的 (X, Y) 坐标点！
    execute_trajectory, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd = navdp_navigator.step_pointgoal(goal, image, depth)
    
    if navdp_fps_writer is not None:
        navdp_fps_writer.append_data(trajectory_mask)

    # 提取未来 20 步的路径点
    execute_trajectory = execute_trajectory[0, 2:]  
    
    # ================= 脑部 X 光透视 (监控模型规划的路径) =================
    try:
        if isinstance(execute_trajectory, torch.Tensor):
            plot_traj = execute_trajectory.detach().cpu().numpy()
        else:
            plot_traj = execute_trajectory
            
        plt.figure(figsize=(6, 6))
        plt.plot(0, 0, 'r*', markersize=15, label='LeKiwi (Origin)')
        
        traj_x = plot_traj[:, 0]
        traj_y = plot_traj[:, 1]
        plt.plot(traj_x, traj_y, 'g-o', label='Planned Path (Waypoints)')
        plt.plot(goal_x, goal_y, 'bx', markersize=12, label='Goal')

        plt.title("LoGoPlanner: Future 20 Waypoints")
        plt.xlabel("Forward X (meters)")
        plt.ylabel("Left/Right Y (meters)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal') 
        plt.savefig("brain_trajectory.png")
        plt.close()
    except Exception as e:
        pass

    # ================= 动作执行与 MPC (将路径坐标转化为底盘速度) =================
    if abs(execute_trajectory[:, 1]).max() > 0.3:
        execute_trajectory[:, 1] *= 2.0

    # 重新请回 MPC，因为官方模型输出的是路径点，不是油门！
    mpc = Mpc_controller(
        execute_trajectory,
        is_omnidirectional=True,
        vx_max=0.8, vy_max=0.8, wz_max=0.3
    )
    mpc.reset()
    
    init_ref_traj = execute_trajectory
    init_theta = mpc.compute_ref_theta(init_ref_traj)[0]
    init_state = np.array([execute_trajectory[0, 0], execute_trajectory[0, 1], init_theta])
    
    x_history = [init_state]
    u_history = []
    
    for _ in range(20): 
        current_pos = x_history[-1][:2]
        end_pos = execute_trajectory[-1, :2]
        if np.linalg.norm(current_pos - end_pos) < 0.05:
            break
        # MPC 算出为了沿着路径走，当前应该踩多少油门
        opt_u, opt_x = mpc.solve(x_history[-1])
        x_history.append(opt_x[1, :])
        u_history.append(opt_u[0, :])

    u_history = np.array(u_history)
    if len(u_history) == 0:
        u_history = np.zeros((10, 3))
        
    u_history[:, 2] = u_history[:, 2] * 180.0 / np.pi  
    return jsonify({'cmd_list': u_history.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=args.port)