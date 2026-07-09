# Mini Clash Royale: Multi-Agent Reinforcement Learning using PPO

This repository contains an advanced Reinforcement Learning (RL) framework designed to simulate and train competitive autonomous agents within a custom environment based on Clash Royale (`MiniClashRoyaleEnv`). Leveraging **PPO (Proximal Policy Optimization)** policies provided by `Stable-Baselines3`, the system enables simultaneous, self-play training of multiple agents in a dynamic, real-time strategic decision-making arena.

## 🚀 Key Features

- **Multi-Agent Environment Architecture:** Custom wrappers built on top of `Gymnasium`/`Gym` to seamlessly coordinate simultaneous actions from opposing factions (Blue and Red).
- **Advanced Reward Shaping:** Fine-tuned reward systems penalizing poor placement, enforcing efficient elixir management, and providing positive reinforcement for tower damage and critical crown-victory conditions.
- **Self-Play Infrastructure:** Automated competitive training setups to match current policy iterations against historical checkpoints or randomized baselines to prevent policy stagnation.

## 📺 Gameplay & Agent Demonstration

Here is a 19-second demonstration of the multi-agent environment and trained policies in action:

https://github.com/user-attachments/assets/ac635934-c175-40da-8311-bf1cfffc7750



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


(Note: The baseline MiniClashRoyaleEnv simulator requires proprietary package access. For integration and licensing of the core environment, please refer to the primary developer guidelines inside the codebase headers).

⚠️ Current Scope and Computational Framework
This repository represents Version 1.0 (Initial Release) of the project. Given the intensive computational requirements inherently associated with training deep multi-agent reinforcement learning models, this initial iteration has been calibrated to accommodate standard consumer-grade hardware configurations.
To maintain training viability under restricted localized hardware budgets, optimization caps have been applied to hyperparameter scales, batch sizes, and total training timesteps. While the agent successfully masters fundamental placement strategies and resource-conservation loops, higher-tier tactical meta-play and deep behavioral convergence remain bounded by current compute allocations. Future versions plan to scale these boundaries via cloud-distributed execution.

🤖 AI Assistance Declaration
This project has been developed with the assistive support of generative artificial intelligence models. AI tools were utilized to optimize syntactic logic across complex reward wrappers, assist in code refactoring, and accelerate the structural compilation of technical documentation. The foundational architecture, programmatic state engineering, and multi-agent competitive designs remain the original conceptual work of the author.

📄 License & Intellectual Property Rights
© 2026. All Rights Reserved.
No part of this repository—including source code, environment wrapper logic, architectural designs, or documentation—may be reproduced, distributed, modified, copied, or used for commercial purposes without the express, prior written permission of the copyright holder. This project is shared strictly for portfolio exhibition and technical demonstration purposes.
