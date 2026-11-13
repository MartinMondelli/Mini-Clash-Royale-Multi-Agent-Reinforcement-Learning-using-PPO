#!/usr/bin/env python3
"""Drag-and-drop arena: play Mini Clash Royale against the trained PPO agent.

Drag a card from your hand onto your half of the arena to play it. The model
plays the other side in real time and, when each match ends, runs a PPO update
on that match before saving (same reward scheme as play_vs_model.py).

Put this file next to play_vs_model.py (it reuses its imports and Environment shim).
Run:  python play_arena.py --model-side blue
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import play_vs_model as pvm  # also sets up the Environment module and DECK
from play_vs_model import Environment

ARENA_W = Environment.ARENA_WIDTH
ARENA_H = Environment.ARENA_HEIGHT
RIVER_Y = Environment.RIVER_Y
BRIDGE_X = Environment.BRIDGE_X
STEP_PENALTY = -0.001  # same per-turn penalty as the training environment
CELL = 20  # pixels per arena tile

# Short codes drawn on each unit so you can tell cards apart at a glance.
# Hover a unit to see its full name.
SHORT_NAMES = {
    "knight": "KNI", "archers": "ARC", "goblins": "GOB", "skeletons": "SKE",
    "minions": "MIN", "baby_dragon": "BDR", "valkyrie": "VAL", "mega_knight": "MEG",
    "giant": "GIA", "royal_giant": "RGI", "golem": "GOL", "hog_rider": "HOG",
    "cannon": "CAN", "tesla": "TES", "bomb_tower": "BOM", "spear_goblins": "SPG",
    "minion_horde": "MHO", "fireball": "FIR", "arrows": "ARW", "zap": "ZAP", "poison": "POI",
}


def short_name(card_name: str) -> str:
    return SHORT_NAMES.get(card_name, card_name[:3].upper())


class _ObsOnlyEnv(gymnasium.Env):
    """Minimal env so VecNormalize can be loaded; used only for observation normalization."""

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space


class Agent:
    """Wraps the PPO model: acts while the match runs, records turns, learns after the match."""

    def __init__(self, model_path: Path, vecnormalize_path: Path, game, learn: bool,
                 save_model: Path, save_vecnormalize: Path):
        dummy = DummyVecEnv([lambda: _ObsOnlyEnv(game.observation_space, game.action_space)])
        self.vn = VecNormalize.load(str(vecnormalize_path), dummy)
        self.vn.training = True  # keep updating obs statistics, as in training
        self.vn.norm_reward = False
        self.model = PPO.load(str(model_path), device="cpu")
        self.learn_enabled = learn
        self.save_model = save_model
        self.save_vecnormalize = save_vecnormalize
        self.items: list = []

    def act(self, raw_obs: np.ndarray):
        """Sample an action (stochastic, as in training) and return what PPO needs to learn from it."""
        obs = self.vn.normalize_obs(raw_obs[None, ...])
        obs_t = obs_as_tensor(obs, self.model.device)
        with torch.no_grad():
            actions, values, log_probs = self.model.policy(obs_t)
        return obs, actions.cpu().numpy(), values, log_probs

    def record(self, obs, actions, reward, episode_start, values, log_probs):
        self.items.append((
            obs.copy(), actions, np.array([reward], dtype=np.float32),
            np.array([episode_start]), values, log_probs,
        ))

    def learn_from_match(self) -> int:
        """Run one PPO update on the recorded match, then save the model and normalization stats."""
        n = len(self.items)
        if n == 0 or not self.learn_enabled:
            self.items = []
            return 0
        m = self.model
        if not hasattr(m, "_logger") or m._logger is None:
            m.set_logger(configure(folder=None, format_strings=["stdout"]))
        buffer = RolloutBuffer(
            buffer_size=n,
            observation_space=m.observation_space,
            action_space=m.action_space,
            device=m.device,
            gae_lambda=m.gae_lambda,
            gamma=m.gamma,
            n_envs=1,
        )
        for item in self.items:
            buffer.add(*item)
        buffer.compute_returns_and_advantage(
            last_values=torch.zeros((1, 1), device=m.device),
            dones=np.ones((1,), dtype=bool),
        )
        m.rollout_buffer = buffer
        m._current_progress_remaining = 1.0
        m.num_timesteps += n
        m.train()
        self.items = []

        self.save_model.parent.mkdir(parents=True, exist_ok=True)
        self.save_vecnormalize.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(self.save_model))
        self.vn.save(str(self.save_vecnormalize))
        return n


def play_card(game, side: str, card_index: int, x: int, y: int) -> bool:
    """Play a card for `side` using the same rules as the training environment."""
    hand = game.hands[side]
    if not hand:
        return False
    x = int(np.clip(x, 0, ARENA_W - 1))
    if side == "blue":
        y = int(np.clip(y, 0, RIVER_Y - 1))
    else:
        y = int(np.clip(y, RIVER_Y + 1, ARENA_H - 1))
    card_index = int(np.clip(card_index, 0, len(hand) - 1))
    name = hand[card_index]
    card = Environment.ALL_CARDS[name]
    lane = "left" if x < ARENA_W / 2 else "right"
    if game.elixir[side] >= card.elixir and game._apply_card(name, x, y, lane, side):
        game.elixir[side] -= card.elixir
        return True
    return False


def terminal_reward(game, agent_side: str) -> float:
    """Final-turn reward from the agent's perspective (same as HumanMatchEnv)."""
    blue_dead = game.towers["blue"]["king"].is_destroyed
    red_dead = game.towers["red"]["king"].is_destroyed
    own_dead = blue_dead if agent_side == "blue" else red_dead
    enemy_dead = red_dead if agent_side == "blue" else blue_dead
    if enemy_dead and not own_dead:
        return 1.0
    if own_dead and not enemy_dead:
        return -1.0
    hp = {s: sum(t.hp for t in game.towers[s].values()) for s in ("blue", "red")}
    enemy = "red" if agent_side == "blue" else "blue"
    if game.time > 0:
        return 0.0
    return (hp[agent_side] - hp[enemy]) * 0.001


class ArenaApp:
    def __init__(self, root: tk.Tk, game, agent: Agent, model_side: str, tick_ms: int):
        self.root = root
        self.game = game
        self.agent = agent
        self.model_side = model_side
        self.human_side = "red" if model_side == "blue" else "blue"
        self.tick_ms = tick_ms
        self.over = False
        self.started = False
        self.first_turn = True
        self.drag_index: int | None = None
        self.ghost: tk.Label | None = None
        self.matches_played = 0

        root.title("Mini Clash Royale")
        root.configure(bg="#1b1b1b")

        self.info = tk.Label(root, font=("Helvetica", 13, "bold"), fg="white", bg="#1b1b1b")
        self.info.pack(pady=(10, 4))

        self.canvas = tk.Canvas(
            root, width=ARENA_W * CELL, height=ARENA_H * CELL,
            bg="#2e7d32", highlightthickness=0,
        )
        self.canvas.pack(padx=10)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", lambda e: self.hover.config(text=""))

        self.hover = tk.Label(root, text="", font=("Helvetica", 11), fg="#ffd54f", bg="#1b1b1b")
        self.hover.pack()

        side_pos = "top" if self.human_side == "red" else "bottom"
        self.status = tk.Label(root, text=f"Press Start. You are {self.human_side} ({side_pos} of the arena).",
                               font=("Helvetica", 12), fg="white", bg="#1b1b1b",
                               wraplength=ARENA_W * CELL)
        self.status.pack(pady=6)

        hand_frame = tk.Frame(root, bg="#1b1b1b")
        hand_frame.pack(pady=(0, 6))
        self.card_frames = []
        self.card_labels = []
        for i in range(4):
            frame = tk.Frame(hand_frame, width=120, height=72, bd=2, relief="raised", bg="#eeeeee")
            frame.grid(row=0, column=i, padx=5)
            frame.pack_propagate(False)
            label = tk.Label(frame, text="", bg="#eeeeee", font=("Helvetica", 11, "bold"))
            label.pack(expand=True, fill="both")
            for w in (frame, label):
                w.bind("<ButtonPress-1>", lambda e, i=i: self.on_press(e, i))
                w.bind("<B1-Motion>", self.on_drag)
                w.bind("<ButtonRelease-1>", self.on_release)
            self.card_frames.append(frame)
            self.card_labels.append(label)

        self.start_btn = tk.Button(root, text="Start", font=("Helvetica", 12, "bold"),
                                   command=self.start)
        self.start_btn.pack(pady=4, before=self.canvas)
        self.again_btn = tk.Button(root, text="New game", font=("Helvetica", 12, "bold"),
                                   command=self.new_match)

        self.redraw()
        root.after(self.tick_ms, self.tick)

    # ---------- game loop ----------
    def tick(self):
        if self.started and not self.over:
            raw = self.game._get_obs(self.model_side)
            obs, actions, values, log_probs = self.agent.act(raw)
            play_card(self.game, self.model_side, *actions[0])
            self.game._simulate_tick()
            self.game.time -= 1
            done = self.game._game_over()
            reward = terminal_reward(self.game, self.model_side) if done else STEP_PENALTY
            self.agent.record(obs, actions, reward, self.first_turn, values, log_probs)
            self.first_turn = False
            if done:
                self.end_match(reward)
            self.redraw()
        self.root.after(self.tick_ms, self.tick)

    def end_match(self, reward: float):
        self.over = True
        if reward > 0:
            winner = self.model_side
        elif reward < 0:
            winner = self.human_side
        else:
            winner = "draw"
        if winner == "draw":
            text = "Draw on tower health."
        elif winner == self.model_side:
            text = "The model wins!"
        else:
            text = "You win!"
        self.matches_played += 1
        self.status.config(text=text, font=("Helvetica", 16, "bold"), fg="#ffd54f")
        self.again_btn.pack(pady=4)
        self.root.update_idletasks()

        if self.agent.learn_enabled:
            self.status.config(text=f"{text}\nLearning from this match...")
            self.root.update_idletasks()
            try:
                turns = self.agent.learn_from_match()
                self.status.config(text=f"{text}\nModel updated from {turns} turns (match {self.matches_played}).")
            except Exception as exc:  # keep the window alive if training fails
                self.status.config(text=f"{text}\nTraining failed: {exc}")

    def on_canvas_motion(self, event):
        cell = self.cell_at(event)
        text = ""
        if cell is not None:
            x, y = cell
            for u in self.game.units:
                if int(u["x"]) == x and int(u["y"]) == y:
                    owner = "you" if u["owner"] == self.human_side else "model"
                    text = f"{u['card'].name} ({owner}) - HP {max(0, int(u['hp']))}/{u['card'].hp}"
                    break
        self.hover.config(text=text)

    def start(self):
        self.started = True
        self.start_btn.pack_forget()
        self.status.config(text="Drag a card onto your half of the arena.")

    def new_match(self):
        self.game.reset()
        self.over = False
        self.started = True
        self.first_turn = True
        self.start_btn.pack_forget()
        self.again_btn.pack_forget()
        self.status.config(text="Drag a card onto your half of the arena.",
                           font=("Helvetica", 12), fg="white")
        self.redraw()

    # ---------- drag and drop ----------
    def on_press(self, event, i):
        if self.over or i >= len(self.game.hands[self.human_side]):
            return
        self.drag_index = i
        name = self.game.hands[self.human_side][i]
        self.ghost = tk.Label(self.root, text=name, bg="#ffd54f", fg="black",
                              relief="solid", bd=1, font=("Helvetica", 11, "bold"))
        self.move_ghost(event)

    def on_drag(self, event):
        self.move_ghost(event)

    def move_ghost(self, event):
        if self.ghost is None:
            return
        x = event.x_root - self.root.winfo_rootx()
        y = event.y_root - self.root.winfo_rooty()
        self.ghost.place(x=x - 40, y=y - 12)
        self.ghost.lift()
        self.update_hint(event)

    def cell_at(self, event):
        cx = event.x_root - self.canvas.winfo_rootx()
        cy = event.y_root - self.canvas.winfo_rooty()
        if 0 <= cx < ARENA_W * CELL and 0 <= cy < ARENA_H * CELL:
            return cx // CELL, ARENA_H - 1 - (cy // CELL)
        return None

    def is_legal_tile(self, x, y) -> bool:
        if not (0 <= x < ARENA_W and 0 <= y < ARENA_H):
            return False
        if self.human_side == "blue":
            return y <= RIVER_Y - 1
        return y >= RIVER_Y + 1

    def update_hint(self, event):
        self.canvas.delete("hint")
        if self.drag_index is None:
            return
        cell = self.cell_at(event)
        if cell is None:
            return
        x, y = cell
        name = self.game.hands[self.human_side][self.drag_index]
        cost = Environment.ALL_CARDS[name].elixir
        ok = self.is_legal_tile(x, y) and self.game.elixir[self.human_side] >= cost
        self.canvas.create_rectangle(
            x * CELL, (ARENA_H - 1 - y) * CELL, (x + 1) * CELL, (ARENA_H - y) * CELL,
            outline="#00e676" if ok else "#ff1744", width=3, tags="hint",
        )

    def on_release(self, event):
        if self.ghost is not None:
            self.ghost.destroy()
            self.ghost = None
        self.canvas.delete("hint")
        i = self.drag_index
        self.drag_index = None
        if i is None or self.over:
            return
        cell = self.cell_at(event)
        if cell is None or i >= len(self.game.hands[self.human_side]):
            return  # dropped outside the arena: card stays in hand
        x, y = cell
        name = self.game.hands[self.human_side][i]
        cost = Environment.ALL_CARDS[name].elixir
        if not self.is_legal_tile(x, y):
            self.status.config(text="Place cards on your half of the arena.")
        elif self.game.elixir[self.human_side] < cost:
            self.status.config(text=f"Not enough elixir for {name}.")
        else:
            lane = "left" if x < ARENA_W / 2 else "right"
            if self.game._apply_card(name, x, y, lane, self.human_side):
                self.game.elixir[self.human_side] -= cost
                self.status.config(text=f"You played {name}.")
            else:
                self.status.config(text="You can't place that there.")
        self.redraw()

    # ---------- drawing ----------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        g = self.game
        top_river = (ARENA_H - (RIVER_Y + 1)) * CELL
        bottom_river = (ARENA_H - (RIVER_Y - 1)) * CELL
        c.create_rectangle(0, top_river, ARENA_W * CELL, bottom_river, fill="#4fc3f7", outline="")
        for bx in BRIDGE_X:
            c.create_rectangle((bx - 1) * CELL, top_river, (bx + 1) * CELL, bottom_river,
                               fill="#8d6e63", outline="#5d4037")

        colors = {"blue": "#1e88e5", "red": "#e53935"}
        for side in ("blue", "red"):
            for name, t in g.towers[side].items():
                px = t.x * CELL
                py = (ARENA_H - (t.y + 1)) * CELL
                size = 2 * CELL if name == "king" else CELL
                if t.is_destroyed:
                    c.create_rectangle(px, py, px + size, py + size, outline=colors[side], dash=(3, 3))
                    continue
                c.create_rectangle(px, py, px + size, py + size, fill=colors[side], outline="white")
                max_hp = 4824 if name == "king" else 3052
                c.create_rectangle(px, py - 6, px + size * max(0, t.hp) / max_hp, py - 2,
                                   fill="#66bb6a", outline="")

        for u in g.units:
            px = u["x"] * CELL
            py = (ARENA_H - (u["y"] + 1)) * CELL
            c.create_rectangle(px + 2, py + 2, px + CELL - 2, py + CELL - 2,
                               fill=colors[u["owner"]], outline="white")
            c.create_text(px + CELL / 2, py + CELL / 2, text=short_name(u["card"].name),
                          fill="white", font=("Helvetica", 7, "bold"))
            ratio = max(0.0, u["hp"] / u["card"].hp) if u["card"].hp else 0
            c.create_rectangle(px, py - 4, px + CELL * ratio, py - 1, fill="#66bb6a", outline="")

        self.info.config(text=(
            f"Time {g.time}   |   You ({self.human_side}) elixir {int(g.elixir[self.human_side])}"
            f"   |   Model ({self.model_side}) elixir {int(g.elixir[self.model_side])}"
        ))

        hand = g.hands[self.human_side]
        for i in range(4):
            frame, label = self.card_frames[i], self.card_labels[i]
            if i < len(hand):
                name = hand[i]
                cost = Environment.ALL_CARDS[name].elixir
                bg = "#ffffff" if g.elixir[self.human_side] >= cost else "#9e9e9e"
                label.config(text=f"{name}\n{cost} elixir", bg=bg)
                frame.config(bg=bg)
            else:
                label.config(text="", bg="#424242")
                frame.config(bg="#424242")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drag-and-drop arena; the model learns after each match.")
    parser.add_argument("--model-side", choices=("blue", "red"), default="blue")
    parser.add_argument("--model", type=Path, help="Model checkpoint (default: trained_model_<side>.zip).")
    parser.add_argument("--vecnormalize", type=Path,
                        help="Normalization stats (default: trained_model_<side>_vecnormalize.pkl).")
    parser.add_argument("--save-model", type=Path,
                        help="Where to save the updated model (default: overwrite --model).")
    parser.add_argument("--save-vecnormalize", type=Path,
                        help="Where to save the updated normalization stats (default: overwrite --vecnormalize).")
    parser.add_argument("--no-learn", action="store_true", help="Play only; do not train after matches.")
    parser.add_argument("--tick-ms", type=int, default=500,
                        help="Milliseconds per game tick (lower is faster). Default 500.")
    args = parser.parse_args()

    model_path = args.model or Path(f"trained_model_{args.model_side}.zip")
    vn_path = args.vecnormalize or Path(f"trained_model_{args.model_side}_vecnormalize.pkl")
    for path in (model_path, vn_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint not found: {path}")

    game = Environment.MiniClashRoyaleEnv(blue_initial_deck=pvm.DECK, red_initial_deck=pvm.DECK)
    agent = Agent(
        model_path, vn_path, game,
        learn=not args.no_learn,
        save_model=args.save_model or model_path,
        save_vecnormalize=args.save_vecnormalize or vn_path,
    )

    root = tk.Tk()
    ArenaApp(root, game, agent, args.model_side, args.tick_ms)
    root.mainloop()


if __name__ == "__main__":
    main()