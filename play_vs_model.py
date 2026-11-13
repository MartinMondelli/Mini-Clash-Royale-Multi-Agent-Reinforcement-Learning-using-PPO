#!/usr/bin/env python3
"""Play Mini Clash Royale against a trained PPO agent and update it after each match."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces

# Environment.py still imports the legacy "gym" name before importing Gymnasium.
sys.modules.setdefault("gym", gymnasium)
sys.path.insert(0, str(Path(__file__).resolve().parent / ".gitignore"))
import Environment  # noqa: E402

# Some copies of Environment.py omit the module-level constants while keeping
# MiniClashRoyaleEnv. Supply the simulator's defaults only when they are absent.
_ENV_DEFAULTS = {
    "ARENA_WIDTH": 18,
    "ARENA_HEIGHT": 32,
    "RIVER_Y": 16,
    "BRIDGE_X": [4, 13],
    "MAX_ELIXIR": 10,
    "MATCH_TIME": 180,
    "tower_damage": 0.5,
    "troop_damage": 0.8,
    "ELIXIR_REGEN_RATE": 0.1,
    "multiplier": 2,
}
for _name, _value in _ENV_DEFAULTS.items():
    if not hasattr(Environment, _name):
        setattr(Environment, _name, _value)

if not hasattr(Environment, "Tower"):
    @dataclass
    class Tower:
        x: int
        y: int
        hp: int
        damage: int = 0
        range: int = 0
        is_destroyed: bool = False

    Environment.Tower = Tower

if not hasattr(Environment, "ALL_CARDS") and "ALL_CARDS" in globals():
    Environment.ALL_CARDS = ALL_CARDS

if not hasattr(Environment, "CARD_LIST_MASTER") and hasattr(Environment, "ALL_CARDS"):
    Environment.CARD_LIST_MASTER = list(Environment.ALL_CARDS)


DECK = [
    "knight",
    "archers",
    "giant",
    "baby_dragon",
    "fireball",
    "zap",
    "cannon",
    "hog_rider",
]


class HumanOpponent:
    """Ask the human to choose an action from their current hand."""

    def __init__(
        self,
        game: Environment.MiniClashRoyaleEnv,
        human_side: str,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.game = game
        self.human_side = human_side
        self.input_fn = input_fn

    def get_action(self):
        side = self.human_side
        min_y, max_y = (0, Environment.RIVER_Y - 1) if side == "blue" else (
            Environment.RIVER_Y + 1,
            Environment.ARENA_HEIGHT - 1,
        )

        print(
            f"\nTime left: {self.game.time} | "
            f"Elixir: blue={self.game.elixir['blue']:.1f}, red={self.game.elixir['red']:.1f}"
        )
        print(f"{side.title()} hand:")
        for index, name in enumerate(self.game.hands[side]):
            cost = Environment.ALL_CARDS[name].elixir
            print(f"  {index}: {name} ({cost} elixir)")

        while True:
            card = self.input_fn("Choose a card index (or q to quit): ").strip().lower()
            if card == "q":
                raise KeyboardInterrupt
            try:
                card_index = int(card)
            except ValueError:
                print("Enter a card index from the hand, or q to quit.")
                continue
            if not 0 <= card_index < len(self.game.hands[side]):
                print("That card index is not in your current hand.")
                continue
            break

        while True:
            coords = self.input_fn(
                f"Enter x,y coordinates (x 0-{Environment.ARENA_WIDTH - 1}, "
                f"y {min_y}-{max_y}): "
            ).strip().lower()
            try:
                x_text, y_text = coords.split(",", maxsplit=1)
                x, y = int(x_text), int(y_text)
            except ValueError:
                print("Enter two comma-separated integers, such as 4,8.")
                continue
            if not 0 <= x < Environment.ARENA_WIDTH or not min_y <= y <= max_y:
                print(f"Coordinates must be within x=0..{Environment.ARENA_WIDTH - 1}, y={min_y}..{max_y}.")
                continue
            return np.array([card_index, x, y], dtype=np.int64)


class HumanMatchEnv(gymnasium.Env):
    """Single-agent view of a match, with a human controlling the opposing side."""

    def __init__(
        self,
        game: Environment.MiniClashRoyaleEnv,
        agent_side: str,
        human: HumanOpponent,
    ) -> None:
        super().__init__()
        self.game = game
        self.agent_side = agent_side
        self.human_side = "red" if agent_side == "blue" else "blue"
        self.human = human
        self.action_space = spaces.MultiDiscrete(
            [
                len(game.blue_initial_deck_names)
                if agent_side == "blue"
                else len(game.red_initial_deck_names),
                Environment.ARENA_WIDTH,
                Environment.ARENA_HEIGHT,
            ]
        )
        self.observation_space = game.observation_space
        self.last_result: dict[str, int | str] | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed, options=options)
        return self.game._get_obs(self.agent_side), {}

    def step(self, action):
        actions = {
            self.agent_side: np.asarray(action, dtype=np.int64),
            self.human_side: self.human.get_action(),
        }
        for side in ("blue", "red"):
            card_id, x, y = actions[side]
            if not self.game.hands[side]:
                continue
            x = int(np.clip(x, 0, Environment.ARENA_WIDTH - 1))
            if side == "blue":
                y = int(np.clip(y, 0, Environment.RIVER_Y - 1))
            else:
                y = int(np.clip(y, Environment.RIVER_Y + 1, Environment.ARENA_HEIGHT - 1))
            card_id = int(np.clip(card_id, 0, len(self.game.hands[side]) - 1))
            card_name = self.game.hands[side][card_id]
            card = Environment.ALL_CARDS[card_name]
            lane = "left" if x < Environment.ARENA_WIDTH / 2 else "right"
            if self.game.elixir[side] >= card.elixir and self.game._apply_card(
                card_name, x, y, lane, side
            ):
                self.game.elixir[side] -= card.elixir

        self.game._simulate_tick()
        self.game.time -= 1
        terminated = self.game._game_over()
        reward = 0.0
        if terminated:
            blue_destroyed = self.game.towers["blue"]["king"].is_destroyed
            red_destroyed = self.game.towers["red"]["king"].is_destroyed
            if self.agent_side == "blue" and red_destroyed:
                agent_reward = 1.0
            elif self.agent_side == "blue" and blue_destroyed:
                agent_reward = -1.0
            elif self.agent_side == "red" and blue_destroyed:
                agent_reward = 1.0
            elif self.agent_side == "red" and red_destroyed:
                agent_reward = -1.0
            else:
                tower_hp = {
                    side: sum(tower.hp for tower in self.game.towers[side].values())
                    for side in ("blue", "red")
                }
                agent_reward = (
                    (tower_hp["blue"] - tower_hp["red"])
                    if self.agent_side == "blue"
                    else (tower_hp["red"] - tower_hp["blue"])
                ) * 0.001
                if self.game.time > 0:
                    agent_reward = 0.0
            self.last_result = {
                "winner": (
                    self.agent_side
                    if agent_reward > 0
                    else self.human_side
                    if agent_reward < 0
                    else "draw"
                ),
                "time": self.game.time,
            }
            reward = agent_reward
        else:
            reward = -0.001

        return self.game._get_obs(self.agent_side), reward, terminated, False, {}


def build_environment(agent_side: str, human: HumanOpponent):
    match_env = HumanMatchEnv(human.game, agent_side, human)
    return DummyVecEnv([lambda: match_env]), match_env


def make_rollout_buffer(model: PPO, env: VecNormalize, capacity: int) -> RolloutBuffer:
    return RolloutBuffer(
        buffer_size=capacity,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        gae_lambda=model.gae_lambda,
        gamma=model.gamma,
        n_envs=env.num_envs,
    )


def train_on_match(model: PPO, env: VecNormalize, max_steps: int) -> int:
    """Collect and optimize one complete human-vs-agent episode."""
    transitions = []
    obs = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    steps = 0

    model.policy.set_training_mode(False)
    while True:
        obs_tensor = obs_as_tensor(obs, model.device)
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor)

        actions_np = actions.cpu().numpy()
        next_obs, rewards, dones, _ = env.step(actions_np)
        transitions.append(
            (obs.copy(), actions_np, rewards.copy(), episode_starts.copy(), values, log_probs)
        )
        steps += 1
        obs = next_obs
        episode_starts = dones

        if dones[0]:
            break
        if steps >= max_steps:
            raise RuntimeError(
                f"The match exceeded the configured {max_steps}-step rollout capacity."
            )

    buffer = make_rollout_buffer(model, env, steps)
    for transition in transitions:
        buffer.add(*transition)
    last_values = torch.zeros((env.num_envs, 1), device=model.device)
    buffer.compute_returns_and_advantage(
        last_values=last_values,
        dones=np.ones((env.num_envs,), dtype=bool),
    )
    model.rollout_buffer = buffer
    model._current_progress_remaining = 1.0
    model.num_timesteps += steps
    model.train()
    return steps


def match_result(match: dict[str, int | str], agent_side: str) -> str:
    if match["winner"] == "draw":
        return "Draw on tower health."
    return "The model wins!" if match["winner"] == agent_side else "You win!"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play against a trained model; after each match, PPO trains on that match."
    )
    parser.add_argument("--model-side", choices=("blue", "red"), default="blue")
    parser.add_argument("--model", type=Path, help="Model checkpoint (default: trained_model_<side>.zip).")
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        help="Normalization checkpoint (default: trained_model_<side>_vecnormalize.pkl).",
    )
    parser.add_argument("--save-model", type=Path, help="Where to save the updated model.")
    parser.add_argument(
        "--save-vecnormalize",
        type=Path,
        help="Where to save the updated VecNormalize statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_side = args.model_side
    human_side = "red" if model_side == "blue" else "blue"
    model_path = args.model or Path(f"trained_model_{model_side}.zip")
    vecnormalize_path = args.vecnormalize or Path(
        f"trained_model_{model_side}_vecnormalize.pkl"
    )
    save_model_path = args.save_model or model_path
    save_vecnormalize_path = args.save_vecnormalize or vecnormalize_path

    for path in (model_path, vecnormalize_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required checkpoint not found: {path}. Train/save the model and its "
                "VecNormalize statistics from MultiAgent_Environment_PPO.ipynb first."
            )

    game = Environment.MiniClashRoyaleEnv(
        blue_initial_deck=DECK,
        red_initial_deck=DECK,
    )
    human = HumanOpponent(game, human_side)
    base_env, match_env = build_environment(model_side, human)
    env = VecNormalize.load(str(vecnormalize_path), base_env)
    env.training = True
    env.norm_reward = True
    model = PPO.load(str(model_path), env=env)
    model.set_logger(configure(folder=None, format_strings=["stdout"]))

    max_steps = Environment.MATCH_TIME + 1
    print(f"Playing as {human_side}; the trained model plays {model_side}.")
    print("After each match, PPO updates from that match and saves the model and normalization stats.")

    match_number = 0
    try:
        while True:
            print(f"\n=== Match {match_number + 1} ===")
            steps = train_on_match(model, env, max_steps)
            match_number += 1
            if match_env.last_result is None:
                raise RuntimeError("The match ended without recording a result.")
            result = match_result(match_env.last_result, model_side)
            print(f"\nMatch finished in {steps} turns. {result}")
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            save_vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_model_path))
            env.save(str(save_vecnormalize_path))
            print(f"Updated model saved to {save_model_path}")
            print(f"Updated normalization stats saved to {save_vecnormalize_path}")
    except KeyboardInterrupt:
        print("\nExiting. The last completed match was saved.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
