import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import LoGoPlanner_Agent
import numpy as np
import cv2
import time
import datetime
import json
import os
from deployment.mpc_controller import Mpc_controller
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=19999)
parser.add_argument("--checkpoint",type=str,default="/home/zbx/NavDP/baselines/logoplanner/logoplanner_policy.ckpt")
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
plot_counter = 0  

@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    
    if navdp_navigator is None:
        print("🧠 [INFO] 正在唤醒大模型...")
        navdp_navigator = LoGoPlanner_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device='cuda:0')
        navdp_navigator.reset(batchsize,threshold)
        print("✅ [INFO] 模型就绪！")
    else:
        navdp_navigator.reset(batchsize,threshold)
    return jsonify({"algo":"logoplanner"})

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    global navdp_navigator, plot_counter
    
    # 🌟 安全检查：防止因顺序问题导致 NoneType 报错 
    if navdp_navigator is None:
        return jsonify({'error': 'model not initialized'}), 400
    
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size
    
    # 图像预处理
    image_pil = Image.open(image_file.stream).convert('RGB')
    
    # 1. 必须先转成 Numpy 数组，才能用 .shape
    image_np = np.asarray(image_pil) 
    
    # 2. 这里的 image.shape[1] 就能正常工作了
    image = image_np.reshape((batch_size, -1, image_np.shape[1], 3))
    
    depth = Image.open(depth_file.stream).convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis].astype(np.float32)
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    depth[depth > 10000] = 0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    # 深度图补全 [cite: 1]
    from scipy import ndimage
    zero_mask = (depth == 0)
    if np.any(zero_mask):
        window_size = 80
        filled_depth = depth.copy()
        non_zero_depth = depth.copy().astype(np.float32)
        non_zero_depth[depth == 0] = np.inf
        min_filtered = ndimage.minimum_filter(non_zero_depth, size=window_size)
        valid_fill_mask = zero_mask & (min_filtered != np.inf)
        filled_depth[valid_fill_mask] = min_filtered[valid_fill_mask]
        depth = filled_depth
    
    depth = depth / 1000.0
    depth[depth < 0] = 0
    
    # 模型预测 [cite: 1]
    execute_trajectory, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd = navdp_navigator.step_pointgoal(goal,image,depth)
    
    # MPC 轨迹追踪 [cite: 1]
    execute_trajectory = execute_trajectory[0, 2:]  
    mpc = Mpc_controller(execute_trajectory, is_omnidirectional=True, vx_max=0.8, vy_max=0.8, wz_max=0.3)
    mpc.reset()
    
    # 🌟 修复起点：从 (0,0) 开始画线 [cite: 21]
    init_theta = mpc.compute_ref_theta(execute_trajectory)[0]
    init_state = np.array([0.0, 0.0, init_theta])
    
    x_history = [init_state]
    u_history = []
    for i in range(20):
        if np.linalg.norm(x_history[-1][:2] - execute_trajectory[-1, :2]) < 0.05: break
        opt_u, opt_x = mpc.solve(x_history[-1])
        x_history.append(opt_x[1, :])
        u_history.append(opt_u[0, :])

    x_history = np.array(x_history)
    u_history = np.array(u_history) if len(u_history) > 0 else np.zeros((10, 3))
        
    # ================= 论文图可视化 [cite: 11] =================
    plot_counter += 1
    if plot_counter % 2 == 0:
        try:
            plt.figure(figsize=(8, 6), dpi=300) 
            plt.plot(execute_trajectory[:, 0], execute_trajectory[:, 1], 'b--', label='NavDP Reference')
            plt.plot(x_history[:, 0], x_history[:, 1], 'g-', label='MPC Path')
            plt.plot(0, 0, 'r*', markersize=15, label='Start')
            plt.plot(goal_x, goal_y, 'kx', markersize=12, label='Goal')
            plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.axis('equal')
            plt.savefig("paper_fig_traj_latest.png"); plt.close()
        except: pass

   # ================= 🌟 终极坐标转换、限速与止抽补丁 =================
    raw_vx, raw_vy, raw_wz = u_history[:, 0].copy(), u_history[:, 1].copy(), u_history[:, 2].copy()

    # 1. 坐标映射 (根据你 PDF 的反馈：Y前进，X横移)
    u_history[:, 0] = raw_vy    # 前进
    u_history[:, 1] = -raw_vx   # 左右 (如果还是左右反了，就把这里的负号去掉)
    u_history[:, 2] = raw_wz * 180.0 / np.pi  
    
    # 2. 🌟 动作降噪 (止抽逻辑)
    # 如果速度指令非常小（比如小于 0.05），强制归零，防止原地微小抽动
    u_history[np.abs(u_history) < 0.05] = 0
    
    # 3. 🌟 全局大幅降速 (保命模式)
    # 1.6 秒的延迟太高了，咱们把速度降到原来的 30%，让它有足够时间反应
    u_history[:, :2] *= 0.3  
    
    # 4. 🌟 补齐口粮 (截断长度增加到 35)
    # 35 * 0.05s = 1.75s，刚好覆盖你 1.6s 的计算延迟
    u_history = u_history[:35]

    return jsonify({'cmd_list': u_history.tolist()})
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=args.port)