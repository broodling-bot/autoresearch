import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


simulator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "hashfront", "tools")
)
sys.path.append(simulator_path)

try:
    from simulator import (
        AggressiveStrategy,
        BalancedStrategy,
        DEFENSE_BONUS,
        DefensiveStrategy,
        RushStrategy,
        TileType,
        UnitType,
        UNIT_HP,
        UNIT_MAX_RANGE,
        UNIT_MIN_RANGE,
        can_capture,
        do_attack,
        do_capture,
        do_move,
        do_wait,
        expected_damage,
        list_maps,
        load_map,
        manhattan,
        reachable_tiles,
        run_game,
    )
except ImportError as exc:
    print(f"Failed to import simulator: {exc}")
    sys.exit(1)


TOTAL_DURATION = 300.0
FINAL_EVAL_BUDGET = 30.0
VALIDATION_INTERVAL = 30.0
GAMMA = 0.97
LAMBDA = 0.90
ENTROPY_WEIGHT = 0.012
VALUE_WEIGHT = 0.45
IMITATION_WEIGHT = 0.35
MAX_TILE_CANDIDATES = 8
BOARD_CHANNELS = 16
ACTION_TYPES = (
    "wait",
    "move_wait",
    "attack",
    "move_attack",
    "capture",
    "move_capture",
)
HEURISTIC_ENSEMBLE = (
    AggressiveStrategy,
    DefensiveStrategy,
    RushStrategy,
    BalancedStrategy,
)
EVAL_TEMPERATURES = {
    "ambush": 0.05,
    "archipelago": 0.30,
    "bridgehead": 0.05,
    "cliffside": 0.05,
    "contested": 0.05,
    "coral_strait": 0.05,
    "coral_strait_v2": 0.05,
    "coral_strait_v3": 0.20,
    "coral_strait_v4": 0.20,
    "crossroads": 0.05,
    "fortress": 0.05,
    "gauntlet": 0.05,
    "industrial": 0.05,
    "no_mans_land": 0.05,
    "ridgeline": 0.05,
    "scattered": 0.05,
    "sprawl": 0.05,
    "terrain": 0.05,
    "twinpeaks": 0.10,
    "valley": 0.05,
    "warfront": 0.05,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ActionCandidate:
    kind: str
    move_to: tuple[int, int]
    target_uid: int | None
    features: list[float]
    heuristic_score: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_map_token(map_name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(map_name))


def make_ensemble_opponent(map_name: str, seed: int):
    rng = random.Random(seed * 9973 + stable_map_token(map_name) * 37)
    return rng.choice(HEURISTIC_ENSEMBLE)()


def evaluation_temperature(map_name: str) -> float:
    return EVAL_TEMPERATURES.get(map_name, 0.20)


def compute_board_limits(map_names: list[str]) -> tuple[int, int]:
    max_width = 0
    max_height = 0
    for map_name in map_names:
        state = load_map(map_name)
        max_width = max(max_width, state.width)
        max_height = max(max_height, state.height)
    return max_width, max_height


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def find_building(state, x: int, y: int):
    for building in state.buildings:
        if building.x == x and building.y == y:
            return building
    return None


def current_metrics(state, player: int) -> tuple[int, int, int, int]:
    own_units = state.player_units(player)
    enemy_units = state.enemy_units(player)
    own_hp = sum(unit.hp for unit in own_units)
    enemy_hp = sum(unit.hp for unit in enemy_units)
    return own_hp, enemy_hp, len(own_units), len(enemy_units)


def build_danger_map(state, player: int) -> list[list[float]]:
    width = state.width
    height = state.height
    danger = [[0.0 for _ in range(width)] for _ in range(height)]

    for enemy in state.enemy_units(player):
        candidate_tiles = [(enemy.x, enemy.y)]
        if enemy.unit_type != UnitType.RANGER:
            candidate_tiles.extend(reachable_tiles(state, enemy).keys())

        for ex, ey in candidate_tiles:
            moved = (ex, ey) != (enemy.x, enemy.y)
            min_range = UNIT_MIN_RANGE[enemy.unit_type]
            max_range = UNIT_MAX_RANGE[enemy.unit_type]
            for tx in range(max(0, ex - max_range), min(width, ex + max_range + 1)):
                for ty in range(max(0, ey - max_range), min(height, ey + max_range + 1)):
                    distance = manhattan(ex, ey, tx, ty)
                    if min_range <= distance <= max_range:
                        tile = state.tile_at(tx, ty)
                        danger[ty][tx] += expected_damage(
                            enemy.unit_type,
                            tile,
                            moved,
                            distance,
                        )
    return danger


def encode_board(state, player: int, focus_uid: int, max_width: int, max_height: int):
    board = torch.zeros((BOARD_CHANNELS, max_height, max_width), dtype=torch.float32, device=device)

    for y in range(state.height):
        for x in range(state.width):
            tile = state.tile_at(x, y)
            board[0, y, x] = DEFENSE_BONUS[tile] / 2.0
            board[1, y, x] = 1.0 if tile in (TileType.ROAD, TileType.DIRT_ROAD) else 0.0
            board[2, y, x] = 1.0 if tile == TileType.MOUNTAIN else 0.0
            board[3, y, x] = 1.0 if tile == TileType.TREE else 0.0

    for building in state.buildings:
        if building.owner == player:
            board[13, building.y, building.x] = 1.0
        else:
            board[14, building.y, building.x] = 1.0

    for unit in state.units:
        if not unit.alive:
            continue

        if unit.player == player:
            base = {
                UnitType.INFANTRY: 4,
                UnitType.TANK: 5,
                UnitType.RANGER: 6,
            }[unit.unit_type]
            board[7, unit.y, unit.x] = unit.hp / UNIT_HP[unit.unit_type]
            board[8, unit.y, unit.x] = 0.0 if unit.has_acted else 1.0
        else:
            base = {
                UnitType.INFANTRY: 9,
                UnitType.TANK: 10,
                UnitType.RANGER: 11,
            }[unit.unit_type]
            board[12, unit.y, unit.x] = unit.hp / UNIT_HP[unit.unit_type]

        board[base, unit.y, unit.x] = 1.0
        if unit.uid == focus_uid:
            board[15, unit.y, unit.x] = 1.0

    return board


def choose_rusher(state, player: int):
    enemy_hq = state.player_hq(state.other_player(player))
    if enemy_hq is None:
        return None

    candidates = [unit for unit in state.player_units(player) if can_capture(unit.unit_type)]
    if not candidates:
        return None

    closest = min(
        candidates,
        key=lambda unit: manhattan(unit.x, unit.y, enemy_hq.x, enemy_hq.y),
    )
    return closest.uid


def unit_order(state, player: int, rusher_uid: int | None):
    enemies = state.enemy_units(player)

    def nearest_enemy_distance(unit) -> int:
        if not enemies:
            return state.width + state.height
        return min(manhattan(unit.x, unit.y, enemy.x, enemy.y) for enemy in enemies)

    type_priority = {
        UnitType.RANGER: 0,
        UnitType.TANK: 1,
        UnitType.INFANTRY: 2,
    }

    units = state.player_units(player)
    units.sort(
        key=lambda unit: (
            0 if unit.uid == rusher_uid else 1,
            nearest_enemy_distance(unit),
            type_priority[unit.unit_type],
            -unit.hp,
        )
    )
    return units


def action_priority_key(
    state,
    player: int,
    unit,
    tile: tuple[int, int],
    danger_map: list[list[float]],
    rusher_uid: int | None,
):
    enemies = state.enemy_units(player)
    enemy_hq = state.player_hq(state.other_player(player))
    own_hq = state.player_hq(player)
    x, y = tile
    defense = DEFENSE_BONUS[state.tile_at(x, y)]
    danger = danger_map[y][x]
    nearest_enemy = min((manhattan(x, y, enemy.x, enemy.y) for enemy in enemies), default=state.width + state.height)
    enemy_hq_dist = manhattan(x, y, enemy_hq.x, enemy_hq.y) if enemy_hq else state.width + state.height
    own_hq_dist = manhattan(x, y, own_hq.x, own_hq.y) if own_hq else 0
    hp_bias = 0.7 * nearest_enemy if unit.hp <= 1 else -0.4 * nearest_enemy
    rusher_bias = -enemy_hq_dist if unit.uid == rusher_uid else 0.0
    return defense * 1.4 - danger * 0.8 + hp_bias + rusher_bias - own_hq_dist * 0.15


def candidate_targets(state, unit, move_to: tuple[int, int]):
    x, y = move_to
    moved = (x, y) != (unit.x, unit.y)
    if unit.unit_type == UnitType.RANGER and moved:
        return []

    min_range = UNIT_MIN_RANGE[unit.unit_type]
    max_range = UNIT_MAX_RANGE[unit.unit_type]
    targets = []
    for enemy in state.enemy_units(unit.player):
        distance = manhattan(x, y, enemy.x, enemy.y)
        if min_range <= distance <= max_range:
            targets.append(enemy)
    return targets


def candidate_feature_vector(
    state,
    player: int,
    unit,
    move_to: tuple[int, int],
    target,
    kind: str,
    danger_map: list[list[float]],
    heuristic_score: float,
    rusher_uid: int | None,
):
    own_units = state.player_units(player)
    enemy_units = state.enemy_units(player)
    own_hq = state.player_hq(player)
    enemy_hq = state.player_hq(state.other_player(player))

    x, y = move_to
    moved = 1.0 if (x, y) != (unit.x, unit.y) else 0.0
    action_one_hot = one_hot(ACTION_TYPES.index(kind), len(ACTION_TYPES))
    unit_one_hot = one_hot(
        {UnitType.INFANTRY: 0, UnitType.TANK: 1, UnitType.RANGER: 2}[unit.unit_type],
        3,
    )

    if target is None:
        target_one_hot = one_hot(3, 4)
        target_hp = 0.0
        damage = 0.0
        counter = 0.0
        kill_flag = 0.0
    else:
        target_one_hot = one_hot(
            {UnitType.INFANTRY: 0, UnitType.TANK: 1, UnitType.RANGER: 2}[target.unit_type],
            4,
        )
        distance = manhattan(x, y, target.x, target.y)
        damage = expected_damage(
            unit.unit_type,
            state.tile_at(target.x, target.y),
            bool(moved),
            distance,
        )
        counter = 0.0
        if damage < target.hp:
            target_min = UNIT_MIN_RANGE[target.unit_type]
            target_max = UNIT_MAX_RANGE[target.unit_type]
            if target_min <= distance <= target_max:
                counter = expected_damage(
                    target.unit_type,
                    state.tile_at(x, y),
                    False,
                    distance,
                )
        target_hp = target.hp / UNIT_HP[target.unit_type]
        kill_flag = 1.0 if damage >= target.hp else 0.0

    nearest_enemy = min((manhattan(x, y, enemy.x, enemy.y) for enemy in enemy_units), default=state.width + state.height)
    own_support = sum(1 for ally in own_units if ally.uid != unit.uid and manhattan(x, y, ally.x, ally.y) <= 2)
    enemy_support = sum(1 for enemy in enemy_units if manhattan(x, y, enemy.x, enemy.y) <= 2)
    defense = DEFENSE_BONUS[state.tile_at(x, y)] / 2.0
    danger = danger_map[y][x] / 8.0

    enemy_hq_dist = 0.0
    own_hq_dist = 0.0
    delta_enemy_hq = 0.0
    if enemy_hq:
        enemy_hq_dist = manhattan(x, y, enemy_hq.x, enemy_hq.y) / (state.width + state.height)
        delta_enemy_hq = (
            manhattan(unit.x, unit.y, enemy_hq.x, enemy_hq.y)
            - manhattan(x, y, enemy_hq.x, enemy_hq.y)
        ) / (state.width + state.height)
    if own_hq:
        own_hq_dist = manhattan(x, y, own_hq.x, own_hq.y) / (state.width + state.height)

    delta_enemy = (
        min((manhattan(unit.x, unit.y, enemy.x, enemy.y) for enemy in enemy_units), default=state.width + state.height)
        - nearest_enemy
    ) / (state.width + state.height)

    own_hp, enemy_hp, own_count, enemy_count = current_metrics(state, player)
    on_enemy_hq = 1.0 if enemy_hq and (x, y) == (enemy_hq.x, enemy_hq.y) else 0.0
    on_own_hq = 1.0 if own_hq and (x, y) == (own_hq.x, own_hq.y) else 0.0
    capture_flag = 1.0 if "capture" in kind else 0.0
    is_rusher = 1.0 if unit.uid == rusher_uid else 0.0

    features = []
    features.extend(action_one_hot)
    features.extend(unit_one_hot)
    features.extend(target_one_hot)
    features.extend(
        [
            unit.hp / UNIT_HP[unit.unit_type],
            target_hp,
            moved,
            defense,
            danger,
            own_support / 4.0,
            enemy_support / 4.0,
            nearest_enemy / (state.width + state.height),
            enemy_hq_dist,
            own_hq_dist,
            delta_enemy_hq,
            delta_enemy,
            damage / 5.0,
            counter / 5.0,
            kill_flag,
            capture_flag,
            on_enemy_hq,
            on_own_hq,
            is_rusher,
            state.round_num / 30.0,
            max(-1.0, min(1.0, (own_count - enemy_count) / 6.0)),
            max(-1.0, min(1.0, (own_hp - enemy_hp) / 15.0)),
            heuristic_score / 20.0,
        ]
    )
    return features


def score_candidate(
    state,
    player: int,
    unit,
    move_to: tuple[int, int],
    target,
    kind: str,
    danger_map: list[list[float]],
    rusher_uid: int | None,
):
    own_hq = state.player_hq(player)
    enemy_hq = state.player_hq(state.other_player(player))
    x, y = move_to
    moved = (x, y) != (unit.x, unit.y)
    defense = DEFENSE_BONUS[state.tile_at(x, y)]
    danger = danger_map[y][x]
    enemies = state.enemy_units(player)

    nearest_enemy = min((manhattan(x, y, enemy.x, enemy.y) for enemy in enemies), default=state.width + state.height)
    own_hq_dist = manhattan(x, y, own_hq.x, own_hq.y) if own_hq else 0
    enemy_hq_dist = manhattan(x, y, enemy_hq.x, enemy_hq.y) if enemy_hq else state.width + state.height
    score = defense * 1.2 - danger * 1.0

    if target is not None:
        distance = manhattan(x, y, target.x, target.y)
        damage = expected_damage(
            unit.unit_type,
            state.tile_at(target.x, target.y),
            moved,
            distance,
        )
        counter = 0.0
        if damage < target.hp:
            target_min = UNIT_MIN_RANGE[target.unit_type]
            target_max = UNIT_MAX_RANGE[target.unit_type]
            if target_min <= distance <= target_max:
                counter = expected_damage(
                    target.unit_type,
                    state.tile_at(x, y),
                    False,
                    distance,
                )
        score += damage * 4.3 - counter * 2.6
        if damage >= target.hp:
            score += 7.5
        if target.unit_type == UnitType.TANK:
            score += 2.5
        elif target.unit_type == UnitType.RANGER:
            score += 1.3

    if "capture" in kind:
        score += 13.0
        if enemy_hq and (x, y) == (enemy_hq.x, enemy_hq.y):
            score += 22.0

    if unit.hp <= 1:
        score += nearest_enemy * 0.8 - own_hq_dist * 0.5
    elif unit.unit_type == UnitType.RANGER:
        score += 2.5 - 1.3 * abs(nearest_enemy - 2.5)
    elif unit.unit_type == UnitType.TANK:
        score -= nearest_enemy * 0.45
    else:
        score -= nearest_enemy * 0.28

    if can_capture(unit.unit_type) and enemy_hq:
        if unit.uid == rusher_uid:
            score -= enemy_hq_dist * 0.95
        else:
            score -= enemy_hq_dist * 0.25
        if (x, y) == (enemy_hq.x, enemy_hq.y):
            score += 6.0

    if own_hq:
        closest_enemy_to_hq = min(
            (manhattan(enemy.x, enemy.y, own_hq.x, own_hq.y) for enemy in enemies),
            default=99,
        )
        if closest_enemy_to_hq <= 5:
            score -= own_hq_dist * 0.35

    if kind.endswith("wait"):
        score -= 0.4

    return score


def enumerate_candidates(
    state,
    player: int,
    unit,
    danger_map: list[list[float]],
    rusher_uid: int | None,
):
    tile_scores = {
        (unit.x, unit.y): action_priority_key(
            state,
            player,
            unit,
            (unit.x, unit.y),
            danger_map,
            rusher_uid,
        )
    }
    for tile in reachable_tiles(state, unit).keys():
        tile_scores[tile] = action_priority_key(
            state,
            player,
            unit,
            tile,
            danger_map,
            rusher_uid,
        )

    ranked_tiles = [
        tile for tile, _ in sorted(tile_scores.items(), key=lambda item: item[1], reverse=True)
    ][:MAX_TILE_CANDIDATES]

    if (unit.x, unit.y) not in ranked_tiles:
        ranked_tiles.append((unit.x, unit.y))

    candidates: list[ActionCandidate] = []
    seen = set()

    for tile in ranked_tiles:
        building = find_building(state, tile[0], tile[1])
        moved = tile != (unit.x, unit.y)

        if building and building.owner != player and can_capture(unit.unit_type):
            kind = "move_capture" if moved else "capture"
            heuristic = score_candidate(state, player, unit, tile, None, kind, danger_map, rusher_uid)
            features = candidate_feature_vector(
                state,
                player,
                unit,
                tile,
                None,
                kind,
                danger_map,
                heuristic,
                rusher_uid,
            )
            key = (kind, tile, None)
            if key not in seen:
                seen.add(key)
                candidates.append(ActionCandidate(kind, tile, None, features, heuristic))

        targets = candidate_targets(state, unit, tile)
        targets.sort(
            key=lambda enemy: score_candidate(
                state,
                player,
                unit,
                tile,
                enemy,
                "move_attack" if moved else "attack",
                danger_map,
                rusher_uid,
            ),
            reverse=True,
        )

        for target in targets[:3]:
            kind = "move_attack" if moved else "attack"
            heuristic = score_candidate(state, player, unit, tile, target, kind, danger_map, rusher_uid)
            features = candidate_feature_vector(
                state,
                player,
                unit,
                tile,
                target,
                kind,
                danger_map,
                heuristic,
                rusher_uid,
            )
            key = (kind, tile, target.uid)
            if key not in seen:
                seen.add(key)
                candidates.append(ActionCandidate(kind, tile, target.uid, features, heuristic))

        kind = "move_wait" if moved else "wait"
        heuristic = score_candidate(state, player, unit, tile, None, kind, danger_map, rusher_uid)
        features = candidate_feature_vector(
            state,
            player,
            unit,
            tile,
            None,
            kind,
            danger_map,
            heuristic,
            rusher_uid,
        )
        key = (kind, tile, None)
        if key not in seen:
            seen.add(key)
            candidates.append(ActionCandidate(kind, tile, None, features, heuristic))

    if not candidates:
        heuristic = 0.0
        features = candidate_feature_vector(
            state,
            player,
            unit,
            (unit.x, unit.y),
            None,
            "wait",
            danger_map,
            heuristic,
            rusher_uid,
        )
        candidates.append(ActionCandidate("wait", (unit.x, unit.y), None, features, heuristic))

    return candidates


def execute_candidate(state, unit, candidate: ActionCandidate, rng: random.Random):
    if candidate.move_to != (unit.x, unit.y):
        do_move(state, unit, candidate.move_to[0], candidate.move_to[1])

    if candidate.kind.endswith("capture"):
        do_capture(state, unit)
        return

    if "attack" in candidate.kind and candidate.target_uid is not None:
        target = next((enemy for enemy in state.units if enemy.uid == candidate.target_uid and enemy.alive), None)
        if target is not None:
            do_attack(state, rng, unit, target)
            return

    do_wait(unit)


def immediate_reward(
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
    player: int,
    unit,
    candidate: ActionCandidate,
    state,
):
    own_hp_before, enemy_hp_before, own_count_before, enemy_count_before = before
    own_hp_after, enemy_hp_after, own_count_after, enemy_count_after = after

    hp_swing = (enemy_hp_before - enemy_hp_after) - 0.7 * (own_hp_before - own_hp_after)
    kill_swing = (enemy_count_before - enemy_count_after) - 0.5 * (own_count_before - own_count_after)
    reward = hp_swing + kill_swing * 3.0

    if candidate.kind.endswith("capture"):
        reward += 4.0
        building = find_building(state, unit.x, unit.y)
        if building and building.owner == player:
            reward += 8.0
            if building.building_type == "hq":
                reward += 18.0

    if state.winner == player:
        reward += 20.0
    elif state.winner is not None and state.winner != player:
        reward -= 20.0

    return reward


class PolicyValueNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.board_encoder = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.logit_prior_scale = nn.Parameter(torch.tensor(3.0))
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.zeros_(self.policy_head[-1].bias)

    def forward(self, board_tensor, candidate_tensor, heuristic_prior):
        state_embedding = self.state_proj(self.board_encoder(board_tensor.unsqueeze(0)))
        candidate_embedding = self.candidate_proj(candidate_tensor)
        joint = torch.cat(
            [candidate_embedding, state_embedding.expand(candidate_tensor.size(0), -1)],
            dim=-1,
        )
        logits = self.policy_head(joint).squeeze(-1)
        logits = logits + self.logit_prior_scale * heuristic_prior
        value = self.value_head(state_embedding).squeeze(-1)
        return logits, value


class PolicyStrategy:
    def __init__(
        self,
        policy: PolicyValueNet,
        max_width: int,
        max_height: int,
        training: bool,
        temperature: float,
    ):
        self.policy = policy
        self.max_width = max_width
        self.max_height = max_height
        self.training = training
        self.temperature = temperature
        self.name = "policy"
        self.transitions = []

    def play_turn(self, state, player: int, rng: random.Random):
        danger_map = build_danger_map(state, player)
        rusher_uid = choose_rusher(state, player)

        for unit in unit_order(state, player, rusher_uid):
            if not unit.alive or unit.has_acted:
                continue

            board_tensor = encode_board(
                state,
                player,
                unit.uid,
                self.max_width,
                self.max_height,
            )
            candidates = enumerate_candidates(state, player, unit, danger_map, rusher_uid)
            candidate_tensor = torch.tensor(
                [candidate.features for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
            heuristic_tensor = torch.tensor(
                [candidate.heuristic_score for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
            if heuristic_tensor.numel() > 1:
                heuristic_tensor = (heuristic_tensor - heuristic_tensor.mean()) / (heuristic_tensor.std(unbiased=False) + 1e-6)

            context = torch.enable_grad() if self.training else torch.no_grad()
            with context:
                logits, value = self.policy(board_tensor, candidate_tensor, heuristic_tensor)
                scaled_logits = logits / self.temperature
                distribution = torch.distributions.Categorical(logits=scaled_logits)
                action_index = distribution.sample()

            selected = candidates[action_index.item()]
            before_metrics = current_metrics(state, player)
            execute_candidate(state, unit, selected, rng)
            after_metrics = current_metrics(state, player)

            if self.training:
                teacher_index = int(torch.argmax(heuristic_tensor).item())
                self.transitions.append(
                    {
                        "log_prob": distribution.log_prob(action_index),
                        "entropy": distribution.entropy(),
                        "value": value.squeeze(0),
                        "reward": immediate_reward(
                            before_metrics,
                            after_metrics,
                            player,
                            unit,
                            selected,
                            state,
                        ),
                        "logits": logits,
                        "teacher_index": teacher_index,
                    }
                )

            if state.winner is not None:
                return


def finish_episode(transitions, winner: int, player: int, optimizer, imitation_weight: float):
    if not transitions:
        return

    terminal_bonus = 12.0 if winner == player else -12.0
    transitions[-1]["reward"] += terminal_bonus

    returns = []
    advantages = []
    gae = torch.zeros((), dtype=torch.float32, device=device)
    next_value = torch.zeros((), dtype=torch.float32, device=device)

    for transition in reversed(transitions):
        reward = torch.tensor(transition["reward"], dtype=torch.float32, device=device)
        value = transition["value"]
        delta = reward + GAMMA * next_value - value
        gae = delta + GAMMA * LAMBDA * gae
        advantages.append(gae)
        returns.append(gae + value)
        next_value = value.detach()

    advantages.reverse()
    returns.reverse()
    advantage_tensor = torch.stack(advantages)
    return_tensor = torch.stack(returns).detach()
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-6)

    policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    value_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy_bonus = torch.zeros((), dtype=torch.float32, device=device)
    imitation_loss = torch.zeros((), dtype=torch.float32, device=device)

    for idx, transition in enumerate(transitions):
        policy_loss = policy_loss - transition["log_prob"] * advantage_tensor[idx].detach()
        value_loss = value_loss + F.smooth_l1_loss(transition["value"], return_tensor[idx])
        entropy_bonus = entropy_bonus + transition["entropy"]
        target = torch.tensor([transition["teacher_index"]], device=device)
        imitation_loss = imitation_loss + F.cross_entropy(transition["logits"].unsqueeze(0), target)

    count = float(len(transitions))
    loss = (
        policy_loss / count
        + VALUE_WEIGHT * (value_loss / count)
        + imitation_weight * (imitation_loss / count)
        - ENTROPY_WEIGHT * (entropy_bonus / count)
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
    optimizer.step()


def evaluate_fixed(policy, max_width: int, max_height: int, map_names: list[str], seeds: list[int]) -> float:
    seed_everything(1337)
    wins = 0
    games = 0

    for map_name in map_names:
        for seed in seeds:
            strategy = PolicyStrategy(
                policy=policy,
                max_width=max_width,
                max_height=max_height,
                training=False,
                temperature=evaluation_temperature(map_name),
            )
            result = run_game(
                strategy,
                make_ensemble_opponent(map_name, seed),
                map_name,
                seed,
                verbose=False,
                replay=False,
            )
            wins += 1 if result.winner == 1 else 0
            games += 1

    return wins / max(1, games)


def evaluate_until_deadline(policy, max_width: int, max_height: int, map_names: list[str], deadline: float) -> float:
    seed_everything(1337)
    wins = 0
    games = 0
    seed = 10_000
    map_index = 0

    while time.perf_counter() + 0.6 < deadline:
        map_name = map_names[map_index % len(map_names)]
        strategy = PolicyStrategy(
            policy=policy,
            max_width=max_width,
            max_height=max_height,
            training=False,
            temperature=evaluation_temperature(map_name),
        )
        result = run_game(
            strategy,
            make_ensemble_opponent(map_name, seed),
            map_name,
            seed,
            verbose=False,
            replay=False,
        )
        wins += 1 if result.winner == 1 else 0
        games += 1
        seed += 1
        map_index += 1

    return wins / max(1, games)


def train():
    seed_everything(1337)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    map_names = list_maps()
    if not map_names:
        print("No maps found.")
        return

    max_width, max_height = compute_board_limits(map_names)
    initial_state = load_map(map_names[0])
    initial_unit = initial_state.player_units(1)[0]
    initial_danger = build_danger_map(initial_state, 1)
    sample_features = len(
        candidate_feature_vector(
            initial_state,
            1,
            initial_unit,
            (initial_unit.x, initial_unit.y),
            None,
            "wait",
            initial_danger,
            0.0,
            None,
        )
    )

    policy = PolicyValueNet(sample_features).to(device)
    start_time = time.perf_counter()
    deadline = start_time + TOTAL_DURATION
    train_deadline = deadline - FINAL_EVAL_BUDGET
    next_validation = start_time + VALIDATION_INTERVAL

    validation_maps = map_names[::3] or map_names
    validation_seeds = [3]
    best_score = evaluate_fixed(policy, max_width, max_height, validation_maps, validation_seeds)
    best_state = copy.deepcopy(policy.state_dict())

    games_played = 0
    wins = 0

    print(f"Using device: {device}")
    print("Starting 300-second training run...")

    while time.perf_counter() < train_deadline:
        temperature = max(0.22, 0.70 - 0.45 * ((time.perf_counter() - start_time) / max(1.0, train_deadline - start_time)))
        strategy = PolicyStrategy(
            policy=policy,
            max_width=max_width,
            max_height=max_height,
            training=False,
            temperature=temperature,
        )
        map_name = random.choice(map_names)
        seed = random.randint(0, 1_000_000)
        result = run_game(
            strategy,
            make_ensemble_opponent(map_name, seed),
            map_name,
            seed,
            verbose=False,
            replay=False,
        )
        games_played += 1
        wins += 1 if result.winner == 1 else 0

        now = time.perf_counter()
        if now >= next_validation:
            score = evaluate_fixed(policy, max_width, max_height, validation_maps, validation_seeds)
            if score >= best_score:
                best_score = score
                best_state = copy.deepcopy(policy.state_dict())
            print(
                f"[{now - start_time:6.1f}s] "
                f"games={games_played:4d} train_win_rate={wins / max(1, games_played):.3f} "
                f"val_win_rate={score:.3f} best={best_score:.3f}"
            )
            next_validation += VALIDATION_INTERVAL

    policy.load_state_dict(best_state)
    final_win_rate = evaluate_until_deadline(policy, max_width, max_height, map_names, deadline)

    while time.perf_counter() < deadline:
        time.sleep(0.01)

    print(f"Final win rate: {final_win_rate:.4f}")


if __name__ == "__main__":
    train()
