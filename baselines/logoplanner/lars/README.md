# LARS: Learning-based Adaptive Residual Safety Adapter

A lightweight safety module for NavDP / LoGoPlanner that adds learned
risk assessment and residual action correction **without modifying the
original policy weights**.

> **Important:** LARS is NOT part of the original NavDP or LoGoPlanner
> method. It is a separate add-on module that freezes the pre-trained
> policy and adds a thin learned safety layer on top.

## Architecture

```
LeKiwi (RPi)                    GPU Server (RTX 5060)
    |                                |
    |  RGB + Depth + Goal            |
    |-------------------------------->|  LoGoPlanner Server (:19999)
    |                                 |     |
    |                                 |     | trajectory waypoints
    |                                 |     v
    |                                 |  LARS Wrapper (:19998)
    |                                 |     |-- DepthToScan
    |                                 |     |-- RiskResidualNet
    |                                 |     |-- SafetyShield
    |                                 |     v
    |  adapted cmd_list               |
    |<--------------------------------|
    |
    v
  LeKiwi base motors
```

## Safety layers (innermost to outermost)

1. **SafetyShield** (hard rules): stop, slowdown, side suppression — runs last.
2. **RiskResidualNet** (learned): risk score + residual action correction.
3. **ActionAdapter**: velocity limits, acceleration limits, deadzone, low-pass.
4. **LeKiwi local depth safety** (unchanged, runs on RPi).
5. **Watchdog** (unchanged, runs on RPi).

## Directory structure

```
baselines/logoplanner/lars/
  README.md
  configs/
    lars.yaml          # Runtime config (risk threshold, model path, etc.)
    safety.yaml        # Hard safety shield thresholds
    train.yaml         # Training hyperparameters
  lars/
    __init__.py
    depth_to_scan.py   # D435i depth -> pseudo-LiDAR scan
    safety_shield.py   # Hard-rule safety layer
    action_adapter.py  # Policy output -> velocity commands
    risk_model.py      # RiskResidualNet (PyTorch)
    risk_dataset.py    # Dataset builder from logs
    residual_adapter.py # Combines risk model + safety shield
    lars_runtime.py    # Main orchestrator
    logger.py          # JSONL per-timestep logger
    metrics.py         # Episode & aggregate metrics
  scripts/
    collect_lars_log.py         # Collect logs from deployment
    build_lars_dataset.py       # Build training dataset from logs
    train_lars.py               # Train RiskResidualNet
    eval_lars_offline.py        # Offline evaluation
    run_lars_realworld_server.py # Real-world deployment wrapper
    replay_lars_log.py          # Replay logs through runtime
  tools/
    plot_scan.py       # Visualize pseudo-LiDAR scans
    plot_actions.py    # Action time series
    plot_risk_curve.py # ROC, PR curves
    export_metrics.py  # Export metrics to CSV/JSON
  logs/                # Output directory for logs
  models/              # Output directory for model checkpoints
```

## Quick start

### 1. Collect logs

With mock data (offline test):

```bash
cd baselines/logoplanner/lars
python scripts/collect_lars_log.py --mock-depth --max-steps 100 \
    --output-dir logs/
```

With real hardware:

```bash
# Terminal 1: Original LoGoPlanner server
python baselines/logoplanner/logoplanner_realworld_server.py --port 19999

# Terminal 2: LARS wrapper
python baselines/logoplanner/lars/scripts/run_lars_realworld_server.py \
    --lars-config configs/lars.yaml --upstream-url http://localhost:19999

# Terminal 3: LeKiwi host (point at LARS wrapper)
python baselines/logoplanner/lekiwi_logoplanner_host.py \
    --server-url http://<gpu-server>:19998 --goal-x 5.0 --goal-y 0.0
```

### 2. Build dataset

```bash
python scripts/build_lars_dataset.py --log-dir logs/ \
    --output logs/lars_dataset.pt --print-stats
```

### 3. Train

```bash
python scripts/train_lars.py --config configs/train.yaml
```

### 4. Evaluate

```bash
python scripts/eval_lars_offline.py --dataset logs/lars_dataset.pt \
    --model models/lars_best.pt
```

### 5. Run real-world deployment

```bash
python scripts/run_lars_realworld_server.py \
    --lars-config configs/lars.yaml \
    --safety-config configs/safety.yaml \
    --model models/lars_best.pt \
    --upstream-url http://localhost:19999
```

### 6. Compute metrics

```bash
python tools/export_metrics.py --log-dir logs/ \
    --output metrics.json --format json --per-episode
```

### 7. Visualize

```bash
python tools/plot_scan.py --log logs/lars_log_20250101_120000.jsonl --step 0
python tools/plot_actions.py --log logs/lars_log_20250101_120000.jsonl
python tools/plot_risk_curve.py --dataset logs/lars_dataset.pt --model models/lars_best.pt
```

## Mock mode (no hardware required)

All modules support mock inputs. Run the following for a minimal smoke test:

```python
import sys
sys.path.insert(0, "baselines/logoplanner/lars")
from lars.lars_runtime import LARSRuntime, MockPolicyOutput
import numpy as np

runtime = LARSRuntime.from_config("baselines/logoplanner/lars/configs/lars.yaml")

for step in range(10):
    policy_output = MockPolicyOutput()()
    depth = np.random.uniform(0, 8000, (480, 640)).astype(np.float32)  # mm
    action, info = runtime.step(policy_output, depth)
    print(f"Step {step}: action={action}, risk={info['risk_score']:.3f}")

runtime.close()
```

## Integration points with original code

LARS integrates at two points without modifying original files:

| Integration point | Original file | How LARS connects |
|---|---|---|
| Server wrapper | `logoplanner_realworld_server.py` | LARS runs as a separate Flask server (port 19998) that forwarding requests to the original server (port 19999) and adapts the responses |
| LeKiwi host | `lekiwi_logoplanner_host.py` | Point `--server-url` to LARS wrapper instead of original server |
| Model loading | `policy_agent.py` | LARS never loads or modifies the LoGoPlanner checkpoint |

If you prefer direct function-call integration (instead of HTTP wrapper),
add these lines to `logoplanner_realworld_server.py` after the MPC block:

```python
# === LARS integration point ===
# After: u_history = mpc.solve(...)
# Before: return jsonify({'cmd_list': u_history.tolist()})

from lars.lars_runtime import LARSRuntime
lars = LARSRuntime.from_config("lars/configs/lars.yaml")

adapted_cmds = []
for cmd in u_history:
    final, info = lars.step(cmd[:3], depth[0, 0, :, :, 0], goal_heading=0.0)
    adapted_cmds.append(final.tolist())

u_history = np.array(adapted_cmds)
# === End LARS integration ===
```

## Citing

If you use LARS in your research, please cite both the LARS module and the
original works it builds upon:

- **NavDP**: ... (see main repository)
- **LoGoPlanner**: ... (see baselines/logoplanner/README.md)
- **LARS**: [add citation info here]

## License

Same as the parent NavDP project.
