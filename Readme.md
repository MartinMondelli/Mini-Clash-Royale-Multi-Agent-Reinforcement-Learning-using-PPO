# Mini Clash Royale: Multi-Agent Reinforcement Learning using PPO (v2.0 — Under Development)

This repository contains an advanced Reinforcement Learning (RL) framework designed to simulate, train, and interactively test competitive autonomous agents within a custom environment based on Clash Royale (`MiniClashRoyaleEnv`). Leveraging **PPO (Proximal Policy Optimization)** policies provided by `Stable-Baselines3`, the system enables simultaneous, self-play training as well as real-time human-vs-model evaluation in a dynamic strategic decision-making arena.

## 🚀 Key Features

- **Multi-Agent Environment Architecture:** Custom wrappers built on top of `Gymnasium`/`Gym` to seamlessly coordinate simultaneous actions from opposing factions (Blue and Red).
- **Interactive Player vs. AI (PvAI) Arena (`play_arena.py`):** A new Version 2.0 capability allowing humans to step directly into matches against trained PPO agents, with optional live on-the-fly model adaptation.
- **Advanced Reward Shaping:** Fine-tuned reward systems penalizing poor placement, enforcing efficient elixir management, and providing positive reinforcement for tower damage and critical crown-victory conditions.
- **Self-Play Infrastructure:** Automated competitive training setups to match current policy iterations against historical checkpoints or randomized baselines to prevent policy stagnation.

## 📺 Gameplay & Agent Demonstration

Here is a 20-second demonstration of the multi-agent environment and trained policies in action:

https://github.com/user-attachments/assets/b3d1d923-f7fa-4be8-854d-3ad5416aa8ad

---

## 🕹️ Interactive Play Arena (`play_arena.py`)

Version 2.0 introduces an interactive play mode allowing you to test your skills directly against trained models from the project folder (alongside `play_vs_model.py`, `play_arena.py`, and `.gitignore`/`Environment.py`).

### Common Usage Examples

- **Basic run** (you play red against the blue model, learning after each match):
  ```bash
  python play_arena.py --model-side blue
  ```

- **Play as blue** against a red model:
  ```bash
  python play_arena.py --model-side red
  ```

- **Play without learning** (the model remains static):
  ```bash
  python play_arena.py --model-side blue --no-learn
  ```

- **Save the updated model** to new files (originals stay untouched):
  ```bash
  python play_arena.py --model-side blue --save-model models/blue_v2.zip --save-vecnormalize models/blue_v2_vecnormalize.pkl
  ```

- **Load a specific checkpoint**:
  ```bash
  python play_arena.py --model-side blue --model models/blue_iter_3.zip --vecnormalize models/blue_iter_3_vecnormalize.pkl
  ```

- **Adjust game speed** (milliseconds per tick; default is `500`, resulting in a ~90-second match):
  - Faster game ($250\text{ms}$ ticks):
    ```bash
    python play_arena.py --model-side blue --tick-ms 250
    ```
  - Slower game ($1000\text{ms}$ ticks):
    ```bash
    python play_arena.py --model-side blue --tick-ms 1000
    ```

- **Combine multiple options**:
  ```bash
  python play_arena.py --model-side red --model models/red_iter_5.zip --vecnormalize models/red_iter_5_vecnormalize.pkl --no-learn --tick-ms 300
  ```

### CLI Options Reference

| Option | Default | What it does |
| :--- | :--- | :--- |
| `--model-side {blue,red}` | `blue` | The side the model plays. You play the opposing side. |
| `--model PATH` | `trained_model_<side>.zip` | Model checkpoint to load. |
| `--vecnormalize PATH` | `trained_model_<side>_vecnormalize.pkl` | Normalization stats to load (must match the model). |
| `--save-model PATH` | Same as `--model` | Where the updated model is saved after each match. |
| `--save-vecnormalize PATH` | Same as `--vecnormalize` | Where the updated normalization stats are saved. |
| `--no-learn` | Off | Play without training or saving updates. |
| `--tick-ms N` | `500` | Milliseconds per game tick. |

*(Tip: Use `--save-model` and `--save-vecnormalize` together to ensure the saved model and its normalization stats stay paired. Use `--help` in your terminal to review this list anytime).*

---

## 📐 Component Architecture (Wrappers)

The framework wraps the core simulator using two modular abstraction layers:

### 1. `MultiAgentWrapper`
Combines the observation and action spaces of both competitors into a unified interface.
- **Action Space:** A `MultiDiscrete` space packing (Card Selection, X-Coordinate, Y-Coordinate) for both players simultaneously.
- **Simulation Management:** Controls the sequential execution of environmental ticks (`_simulate_tick`), synchronizing elixir generation and grid-based unit interaction.

### 2. `CompetitiveWrapper`
Isolates a single-agent perspective (either Blue or Red) to allow asymmetrical training. It encapsulates the opponent's policy to simulate fluid 1v1 matches from a standard single-agent reinforcement learning perspective.

## 📊 Environmental Reward Design

Agent convergence is driven by the following mathematical and logical metrics:
- **Deployment Efficiency:** Positive baseline feedback upon successful unit card deployment, scaled by the card’s intrinsic elixir cost.
- **Critical Penalties:** Substantial negative rewards applied on illegal moves (e.g., deployment with insufficient elixir, out-of-bounds positioning, or empty hand actions).
- **Strategic Combat Metrics:** Rewards tied directly to the health differential (HP) of friendly and opposing towers at every temporal step.

## 🛠️ Installation & Requirements

Ensure you are using Python 3.10 or higher. You can install the required dependencies using:

```bash
pip install shimmy>=0.2.1
pip install stable_baselines3[extra]
pip install gymnasium numpy matplotlib
```

*(Note: The baseline MiniClashRoyaleEnv simulator requires proprietary package access. For integration and licensing of the core environment, please refer to the primary developer guidelines inside the codebase headers).*

---

## ⚠️ Current Scope and Computational Framework (v2.0 Development)

This repository represents **Version 2.0 (Under Development)**, introducing interactive Player versus AI (`play_arena.py`) capabilities. Given the intensive computational requirements inherently associated with training deep multi-agent reinforcement learning models, optimization caps remain applied to hyperparameter scales, batch sizes, and total training timesteps to accommodate standard consumer-grade hardware configurations. 

While the agent successfully masters fundamental placement strategies and resource-conservation loops, higher-tier tactical meta-play and deep behavioral convergence remain bounded by current local compute allocations. Future updates plan to scale these boundaries further via cloud-distributed execution and expanded arena utilities.

🤖 **AI Assistance Declaration**  
This project has been developed with the assistive support of generative artificial intelligence models. AI tools were utilized to optimize syntactic logic across complex reward wrappers, assist in code refactoring, and accelerate the structural compilation of technical documentation. The foundational architecture, programmatic state engineering, and multi-agent competitive designs remain the original conceptual work of the author.

📄 **License & Intellectual Property Rights**  
© 2026. All Rights Reserved.  
No part of this repository—including source code, environment wrapper logic, architectural designs, or documentation—may be reproduced, distributed, modified, copied, or used for commercial purposes without the express, prior written permission of the copyright holder. This project is shared strictly for portfolio exhibition and technical demonstration purposes.