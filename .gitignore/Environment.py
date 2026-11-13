# =========================
# IMPORTS
# =========================
import gym
from gym import spaces
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from IPython.display import HTML
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# =========================
# ENVIRONMENT
# =========================
class MiniClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, blue_initial_deck=None, red_initial_deck=None):
        super().__init__()

        # Use provided decks or default to random selection from CARD_LIST_MASTER
        self.blue_initial_deck_names = blue_initial_deck if blue_initial_deck is not None else np.random.choice(CARD_LIST_MASTER, 8, replace=False).tolist()
        self.red_initial_deck_names = red_initial_deck if red_initial_deck is not None else np.random.choice(CARD_LIST_MASTER, 8, replace=False).tolist()

        # Action space for the blue player (assuming single-player control of blue)
        self.action_space = spaces.MultiDiscrete([len(self.blue_initial_deck_names), ARENA_WIDTH, ARENA_HEIGHT])
        self.observation_space = spaces.Box(low=0, high=10000, shape=(12,), dtype=np.float32) # Hand now contains indices from CARD_LIST_MASTER

        self.units = []
        self.hands = {"blue": [], "red": []} # Stores card names in hand
        self.player_decks = {"blue": deque(), "red": deque()} # Stores card names in draw pile
        self.player_discard_piles = {"blue": deque(), "red": deque()} # Stores card names in discard pile
        self.elixir = {}
        self.towers = {}
        self.time = MATCH_TIME
        self.last_cards_played = {"blue": deque(maxlen=5), "red": deque(maxlen=5)}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = MATCH_TIME
        self.elixir = {"blue": 5, "red": 5}
        self.units = []

        # Initialize decks, hands, and discard piles
        self.hands = {"blue": [], "red": []}
        self.player_discard_piles = {"blue": deque(), "red": deque()}

        # Shuffle initial decks into draw piles
        self.player_decks["blue"] = deque(np.random.permutation(self.blue_initial_deck_names).tolist())
        self.player_decks["red"] = deque(np.random.permutation(self.red_initial_deck_names).tolist())

        # Draw initial hands (4 cards each)
        for owner in ["blue", "red"]:
            for _ in range(4):
                if self.player_decks[owner]: # Ensure deck is not empty
                    self.hands[owner].append(self.player_decks[owner].popleft())

        self.towers = {
            "blue": {"king": Tower(8, 2, 4824, damage = 109*multiplier, range = 7, is_destroyed=False),
                     "left": Tower(4, 2, 3052, damage = 136*multiplier, range = 8, is_destroyed=False),
                     "right": Tower(13, 2, 3052, damage = 136*multiplier, range = 8, is_destroyed=False)},
            "red":  {"king": Tower(8, 29, 4824, damage = 109*multiplier, range = 7, is_destroyed=False),
                     "left": Tower(4, 29, 3052, damage = 136*multiplier, range = 8, is_destroyed=False),
                     "right": Tower(13, 29, 3052, damage = 136*multiplier, range = 8, is_destroyed=False)}
        }
        self.last_cards_played = {"blue": deque(maxlen=5), "red": deque(maxlen=5)}
        return self._get_obs("blue"), {}

    def step(self, action):
        card_id, x, y = action

        reward = 0.0
        terminated = False
        truncated = False
        played_card = False

        # Acción inválida
        if card_id >= len(self.hands["blue"]):
            reward -= 0.1
            self._enemy_play()
            self._simulate_tick()
            self.time -= 1
            return self._get_obs("blue"), reward, False, False, {}

        card_name = self.hands["blue"][card_id]
        card = ALL_CARDS[card_name]
        lane = "left" if x < ARENA_WIDTH / 2 else "right"

        # Intentar jugar carta
        if self.elixir["blue"] >= card.elixir:
            before_hp = sum(t.hp for t in self.towers["red"].values())

            self.elixir["blue"] -= card.elixir
            success = self._apply_card(card_name, x, y, lane, owner="blue")

            if success:
                played_card = True
                reward += 0.02  # ⭐ jugar carta es bueno

            after_hp = sum(t.hp for t in self.towers["red"].values())
            damage = before_hp - after_hp
            reward += damage * 0.01  # ⭐ daño a torre

        # Penalización por no hacer nada
        if not played_card:
            reward -= 0.01

        # Turno enemigo + simulación
        self._enemy_play()
        self._simulate_tick()
        self.time -= 1

        # Fin del juego
        if self._game_over():
            reward += 10.0
            terminated = True

        reward = np.clip(reward, -1.0, 1.0)

        return self._get_obs("blue"), reward, terminated, truncated, {}


    # Modified to accept card_name
    def _apply_card(self, card_name, x, y, lane, owner="blue"):
        card = ALL_CARDS[card_name] # Get card object from its name

        # Restricción de zona
        if owner == "blue" and y >= RIVER_Y and self.towers["red"]["king"].hp > 0: return 0
        if owner == "red"  and y <= RIVER_Y and self.towers["blue"]["king"].hp > 0: return 0

        # If card is a troop or building, apply to units
        if card.card_type == "troop" or card.card_type == "building":
            self.units.append({
                "card": card,
                "x": float(x),
                "y": float(y),
                "lane": lane,
                "owner": owner,
                "hp": card.hp
            })
            self.last_cards_played[owner].append(card.name)

            # Card cycling: remove from hand, add to discard, draw new
            self.hands[owner].remove(card_name)
            self.player_discard_piles[owner].append(card_name)

            if not self.player_decks[owner]:
                # Reshuffle discard pile into draw pile
                np.random.shuffle(self.player_discard_piles[owner])
                self.player_decks[owner].extend(self.player_discard_piles[owner])
                self.player_discard_piles[owner].clear() # Discard pile is now empty

            if self.player_decks[owner]: # Only draw if there are cards to draw
                self.hands[owner].append(self.player_decks[owner].popleft())
            return 1
        return 0

    def _enemy_play(self):
        # IA roja aleatoria
        if not self.hands["red"]: # No cards in hand, cannot play
            return

        red_card_id = np.random.choice(len(self.hands["red"]))
        red_card_name = self.hands["red"][red_card_id]
        red_card = ALL_CARDS[red_card_name]

        x = BRIDGE_X[0] if np.random.rand() < 0.5 else BRIDGE_X[1]
        y = np.random.randint(RIVER_Y + 2, ARENA_HEIGHT)
        lane = "left" if x < ARENA_WIDTH / 2 else "right"

        if self.elixir["red"] >= red_card.elixir:
            self.elixir["red"] -= red_card.elixir
            self._apply_card(red_card_name, x, y, lane, owner="red")

    def _simulate_tick(self):
        for side in ["blue", "red"]:
            self.elixir[side] = min(MAX_ELIXIR, self.elixir[side] + ELIXIR_REGEN_RATE)

        alive_units = []
        for unit in self.units:
            enemy_side = "red" if unit["owner"] == "blue" else "blue"

            # Selección de objetivo (Ultimate Target for movement)
            ultimate_target_x, ultimate_target_y = None, None

            # Buscar unidad enemiga más cercana (para movimiento y ataque)
            closest_enemy_unit = None
            min_dist_to_enemy_unit = float('inf')
            for other_unit in self.units:
                if other_unit["owner"] == unit["owner"]:
                    continue
                dx_u = other_unit["x"] - unit["x"]
                dy_u = other_unit["y"] - unit["y"]
                dist_u = (dx_u**2 + dy_u**2)**0.5
                if dist_u < min_dist_to_enemy_unit:
                    min_dist_to_enemy_unit = dist_u
                    closest_enemy_unit = other_unit

            # Determine the ultimate target for the unit
            if unit["card"].target_preference == "any" and closest_enemy_unit is not None:
                ultimate_target_x = closest_enemy_unit["x"]
                ultimate_target_y = closest_enemy_unit["y"]
            else:
                # If "tower" unit or "any" unit without enemies near, prioritize towers
                tower_name = "left" if unit["lane"] == "left" else "right"
                lane_tower = self.towers[enemy_side][tower_name]
                king_tower = self.towers[enemy_side]["king"]

                if not lane_tower.is_destroyed:
                    ultimate_target_x = lane_tower.x
                    ultimate_target_y = lane_tower.y
                elif not king_tower.is_destroyed:
                    ultimate_target_x = king_tower.x
                    ultimate_target_y = king_tower.y
                else:
                    # All enemy towers destroyed, target general enemy territory
                    ultimate_target_x = ARENA_WIDTH / 2
                    ultimate_target_y = ARENA_HEIGHT if enemy_side == "red" else 0

            # --- Lógica de cruce de río (applied to ultimate target) ---
            river_y_min = RIVER_Y - 1 # y=15
            river_y_max = RIVER_Y + 1 # y=17 (exclusive if drawing height 2 from y_min)

            # Check if unit is on one side of river and its ULTIMATE target is on the other
            unit_is_on_blue_side = unit['y'] < river_y_min
            unit_is_on_red_side = unit['y'] > river_y_max
            ultimate_target_is_on_blue_side = ultimate_target_y < river_y_min
            ultimate_target_is_on_red_side = ultimate_target_y > river_y_max

            needs_to_cross_river = (unit_is_on_blue_side and ultimate_target_is_on_red_side) or \
                                   (unit_is_on_red_side and ultimate_target_is_on_blue_side)

            # Check if unit is currently on any bridge using robust overlap check
            on_a_bridge = False
            for bridge_x_coord in BRIDGE_X:
                # Unit's x-span: [unit['x'], unit['x'] + 1]
                # Bridge's x-span: [bridge_x_coord - 1, bridge_x_coord + 1]
                x_overlap = max(unit['x'], bridge_x_coord - 1) < min(unit['x'] + 1, bridge_x_coord + 1)

                # Unit's y-span: [unit['y'], unit['y'] + 1]
                # River's y-span: [river_y_min, river_y_max]
                y_overlap = max(unit['y'], river_y_min) < min(unit['y'] + 1, river_y_max)

                if x_overlap and y_overlap:
                    on_a_bridge = True
                    break

            # Set actual movement target based on river crossing logic
            current_movement_target_x = ultimate_target_x
            current_movement_target_y = ultimate_target_y

            if needs_to_cross_river and not on_a_bridge:
                # Redirect to the nearest bridge entry point in its lane
                bridge_x_center = BRIDGE_X[0] + 0.5 if unit['lane'] == 'left' else BRIDGE_X[1] + 0.5
                current_movement_target_x = bridge_x_center
                current_movement_target_y = RIVER_Y # Aim for the center of the river at the bridge
            # --- Fin Lógica de cruce de río ---

            # Mover hacia current_movement_target
            dx = current_movement_target_x - unit["x"]
            dy = current_movement_target_y - unit["y"]
            dist = (dx**2 + dy**2)**0.5
            if dist > 0: # Ensure dist is not zero to prevent division by zero and allow precise movement
                step_size = min(unit["card"].speed, dist) # Move at most 'speed', but not past target
                unit["x"] += step_size * dx / dist
                unit["y"] += step_size * dy / dist
            unit["x"] = np.clip(unit["x"], 0, ARENA_WIDTH-1)
            unit["y"] = np.clip(unit["y"], 0, ARENA_HEIGHT-1)

            # Ataque
            attacked = False

            # 1. Prioridad: unidad enemiga más cercana
            if closest_enemy_unit and min_dist_to_enemy_unit <= unit["card"].range:
                closest_enemy_unit["hp"] -= unit["card"].damage*troop_damage
                attacked = True

            # 2. Si no hay unidad → ataca torre de la lane
            if not attacked:
                tower_name = "left" if unit["lane"] == "left" else "right"
                tower = self.towers[enemy_side][tower_name]
                if not tower.is_destroyed and ((tower.x - unit["x"])**2 + (tower.y - unit["y"])**2)**0.5 <= unit["card"].range:
                    tower.hp -= unit["card"].damage*tower_damage
                    if tower.hp <= 0: tower.is_destroyed = True
                    attacked = True

            # 3. Último recurso: rey
            if not attacked:
                king = self.towers[enemy_side]["king"]
                if not king.is_destroyed and ((king.x - unit["x"])**2 + (king.y - unit["y"])**2)**0.5 <= unit["card"].range:
                    king.hp -= unit["card"].damage*tower_damage
                    if king.hp <= 0: king.is_destroyed = True

            if unit["hp"] > 0:
                alive_units.append(unit)

        self.units = alive_units

    def _game_over(self):
        return (self.towers["blue"]["king"].is_destroyed or
                self.towers["red"]["king"].is_destroyed or
                self.time <= 0)

    def _get_obs(self, player):
        enemy = "red" if player=="blue" else "blue"
        obs = [self.elixir[player], self.time]
        obs += [t.hp for t in self.towers[player].values()]
        obs += [t.hp for t in self.towers[enemy].values()]
        obs += [CARD_LIST_MASTER.index(c_name) for c_name in self.hands[player]] # Use CARD_LIST_MASTER for obs indexing
        return np.array(obs, dtype=np.float32)


# =========================
# RENDERER
# =========================
class ClashReplayRenderer:
    def __init__(self, env, fig, ax):
        self.env = env
        self.fig = fig
        self.ax = ax
        # Generate a color map for all cards in CARD_LIST_MASTER
        self.card_colors = self._generate_card_colors(CARD_LIST_MASTER)

    def _generate_card_colors(self, card_list):
        colors = plt.cm.get_cmap('tab20', len(card_list))
        return {card_name: colors(i) for i, card_name in enumerate(card_list)}

    def render(self, alpha=1.0):
        self.ax.clear()
        self.ax.set_xlim(0, ARENA_WIDTH)
        self.ax.set_ylim(0, ARENA_HEIGHT)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Mini Clash Royale — Partida")

        # River
        river = patches.Rectangle((0, RIVER_Y-1), ARENA_WIDTH, 2, color='deepskyblue', alpha=0.3)
        self.ax.add_patch(river)

        for bx in BRIDGE_X:
            bridge = patches.Rectangle((bx-1, RIVER_Y-1), 2, 2, color='saddlebrown')
            self.ax.add_patch(bridge)

        # Torres
        for side, color in [('blue', 'blue'), ('red', 'red')]:
            for name, tower in self.env.towers[side].items():
                r = 0.8 if name == 'king' else 0.6
                alpha = 0.9 if not tower.is_destroyed else 0.2 # Faded if destroyed
                self.ax.add_patch(patches.Circle((tower.x + 0.5, tower.y + 0.5), r, color=color, alpha=alpha))
                if not tower.is_destroyed and tower.hp > 0: # Only draw HP bar for active towers with HP > 0
                    hp_ratio = tower.hp / (4000 if name == 'king' else 2500)
                    self.ax.add_patch(patches.Rectangle((tower.x, tower.y + r + 0.1), 1.0 * hp_ratio, 0.15, color='green'))

        # Tropas
        for unit in self.env.units:
            color = 'deepskyblue' if unit['owner'] == 'blue' else 'orangered'
            self.ax.add_patch(patches.Rectangle((unit['x'], unit['y']), 1, 1, color=color))
            hp_ratio = unit['hp'] / unit['card'].hp
            self.ax.add_patch(patches.Rectangle((unit['x'], unit['y'] + 1.05), hp_ratio, 0.15, color='green'))

        # Current Elixir and Time
        self.ax.text(0.5, ARENA_HEIGHT - 2, f"Elixir: {int(self.env.elixir['blue'])}" , color='blue', fontsize=10, ha='left')
        self.ax.text(ARENA_WIDTH - 0.5, ARENA_HEIGHT - 2, f"Elixir: {int(self.env.elixir['red'])}", color='red', fontsize=10, ha='right')
        self.ax.text(ARENA_WIDTH / 2, ARENA_HEIGHT - 2, f"Time: {self.env.time}", color='black', fontsize=10, ha='center')

        # Initial Decks (fixed for the game), with colors
        self.ax.text(0.5, ARENA_HEIGHT - 3, "Blue Deck:", color='blue', fontsize=8, ha='left')
        for i, card_name in enumerate(self.env.blue_initial_deck_names):
            self.ax.text(0.5, ARENA_HEIGHT - 3.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='left')

        self.ax.text(ARENA_WIDTH - 0.5, ARENA_HEIGHT - 3, "Red Deck:", color='red', fontsize=8, ha='right')
        for i, card_name in enumerate(self.env.red_initial_deck_names):
            self.ax.text(ARENA_WIDTH - 0.5, ARENA_HEIGHT - 3.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='right')

        # Current Hands, with colors
        self.ax.text(0.5, ARENA_HEIGHT - 8, "Blue Hand:", color='blue', fontsize=8, ha='left')
        for i, card_name in enumerate(self.env.hands['blue']):
            self.ax.text(0.5, ARENA_HEIGHT - 8.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='left')

        self.ax.text(ARENA_WIDTH - 0.5, ARENA_HEIGHT - 8, "Red Hand:", color='red', fontsize=8, ha='right')
        for i, card_name in enumerate(self.env.hands['red']):
            self.ax.text(ARENA_WIDTH - 0.5, ARENA_HEIGHT - 8.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='right')

        # Últimas cartas jugadas (these are already present)
        y_start_last_played = ARENA_HEIGHT - 12
        self.ax.text(0.5, y_start_last_played, "Blue Last Played:", color='blue', fontsize=8, ha='left')
        for i, card_name in enumerate(self.env.last_cards_played['blue']):
            self.ax.text(0.5, y_start_last_played - 0.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='left')

        self.ax.text(ARENA_WIDTH - 0.5, y_start_last_played, "Red Last Played:", color='red', fontsize=8, ha='right')
        for i, card_name in enumerate(self.env.last_cards_played['red']):
            self.ax.text(ARENA_WIDTH - 0.5, y_start_last_played - 0.5 - i * 0.5, card_name, color=self.card_colors.get(card_name, 'black'), fontsize=7, ha='right')
