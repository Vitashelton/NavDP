import numpy as np
import requests
import base64
from flask import Flask, request, jsonify
from deployment.mpc_controller import Mpc_controller
from PIL import Image
import json

app = Flask(__name__)

# 🚨 老哥，修改这里！把下面这个 IP 换成你刚才记下的【实验室服务器的 IP】
QWEN_SERVICE_URL = "http://10.33.5.238:8000/get_target"

@app.route("/navigator_reset", methods=['POST'])
def navdp_reset():
    print("📡 [中枢] 分布式导航系统已就绪！")
    return jsonify({"algo": "modular_vlm"})

@app.route("/pointgoal_step", methods=['POST'])
def navdp_step_xy():
    # 1. 接收树莓派发来的照片和深度图
    image_file = request.files['image']
    depth_file = request.files['depth']
    
    # 🌟 核心提速补丁：拆包、把图片缩小、再重新打包发给大模型
    import io
    img_pil = Image.open(image_file.stream).convert('RGB')
    
    # 暴力瘦身：缩小到 224x224。大模型看个大概就够了，极大减少 Token 数量！
    img_resized = img_pil.resize((224, 224)) 
    
    # 重新转成 Base64
    buffered = io.BytesIO()
    img_resized.save(buffered, format="JPEG", quality=70) # 压缩画质提速
    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 深度图处理照旧
    depth_file.seek(0)
    depth = np.asarray(Image.open(depth_file.stream).convert('I')).astype(np.float32) / 1000.0
    h, w = depth.shape
    
    target_pos = None
    # 2. 呼叫组里的服务器，问它纸箱在哪
    try:
        response = requests.post(QWEN_SERVICE_URL, json={"image_b64": image_b64}, timeout=3.0)
        res_data = response.json()
        
        if res_data.get("status") == "success":
            # 把 0-1000 的归一化坐标转回深度图的实际像素点
            u = max(0, min(w-1, int(res_data["x"] * w / 1000)))
            v = max(0, min(h-1, int(res_data["y"] * h / 1000)))
            
            # 取像素点周围 10x10 区域的深度中位数，防止测不准
            roi_depth = depth[max(0, v-5):min(h, v+5), max(0, u-5):min(w, u+5)]
            valid_depths = roi_depth[roi_depth > 0.1]
            z_g = np.median(valid_depths) if len(valid_depths) > 0 else 0
            
            if z_g > 0:
                target_pos = np.array([z_g, -(u - w/2) * z_g / 600])
                print(f"🎯 Qwen 锁定纸箱！距离：{z_g:.2f}米")
    except Exception as e:
        print(f"⚠️ 服务器没反应或超时: {e}")

    # 3. 物理保命逻辑 (绝对底线)
    scan_area = depth[int(h*0.15):int(h*0.35), int(w*0.2):int(w*0.8)]
    valid_scan = scan_area[(scan_area > 0.4) & (scan_area < 3.0)]
    obs_dist = np.percentile(valid_scan, 5) if len(valid_scan) > 50 else 4.0

    # 4. MPC 计算开法
    ref_traj = np.linspace([0,0], target_pos, 20)
    mpc = Mpc_controller(ref_traj, is_omnidirectional=True)
    opt_u, _ = mpc.solve(np.array([0, 0, 0]))
    vx, vy, wz = opt_u[0, 0], opt_u[0, 1], opt_u[0, 2]

    # 5. 一旦有危险，强行接管方向盘
    if obs_dist < 0.65:
        print(f"🛑 [物理拦截] 前方 {obs_dist:.2f}m 有障碍！急停转弯！")
        vx, wz = -0.05, 30.0

    # 给树莓派发 15 帧指令，防止卡顿
    u_history = [[vx * 0.4, vy * 0.4, wz * 180 / np.pi]] * 15 
    return jsonify({'cmd_list': u_history})

if __name__ == "__main__":
    print("💻 笔记本中枢启动，监听 19999 端口...")
    app.run(host='0.0.0.0', port=19999)