import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

@app.route("/navigator_reset", methods=['POST'])
def navdp_reset():
    print("📡 [链路诊断] 树莓派已连接，大脑已唤醒。")
    return jsonify({"algo": "depth_diagnostic_v1"})

@app.route("/pointgoal_step", methods=['POST'])
def navdp_step_xy():
    # 1. 接收来自树莓派的原始数据
    depth_file = request.files['depth']
    
    # 2. 深度图预处理 (16-bit 转为米)
    depth_pil = Image.open(depth_file.stream).convert('I')
    depth = np.asarray(depth_pil).astype(np.float32) / 1000.0
    
    # 3. 🛡️ 核心避障逻辑：只看画面中间的一横条（避开地板和天花板）
    h, w = depth.shape
    # 我们只看垂直方向 40% 到 60% 的区域，横向全扫
    roi = depth[int(h*0.4):int(h*0.6), :]
    
    # 将画面分为左、中、右三份
    third = w // 3
    left_dist = np.median(roi[:, :third][roi[:, :third] > 0.1]) if np.any(roi[:, :third] > 0.1) else 4.0
    mid_dist = np.median(roi[:, third:2*third][roi[:, third:2*third] > 0.1]) if np.any(roi[:, third:2*third] > 0.1) else 4.0
    right_dist = np.median(roi[:, 2*third:][roi[:, 2*third:] > 0.1]) if np.any(roi[:, 2*third:] > 0.1) else 4.0

    print(f"👁️ 深度扫描: 左 {left_dist:.2f}m | 中 {mid_dist:.2f}m | 右 {right_dist:.2f}m")

    # 4. 简单的“条件反射”控制
    vx, vy, wz = 0.0, 0.0, 0.0
    
    if mid_dist < 0.7:  # 正前方有障碍
        print("⚠️ 阻挡！正在避让...")
        vx = -0.05      # 极慢倒车或停下
        wz = 20.0 if left_dist > right_dist else -20.0 # 哪边空往哪边转
    else:
        print("✅ 通畅，直行")
        vx = 0.2        # 正向直行
        wz = 0.0

    # 返回 5 个指令，保证动作连贯
    u_history = [[vx, vy, wz]] * 5
    return jsonify({'cmd_list': u_history})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=19999)