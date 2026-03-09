import sys
import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# Add simulator to path
simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hashfront", "tools"))
sys.path.append(simulator_path)

try:
    from simulator import GameState, load_map, list_maps, run_game, STRATEGIES, BalancedStrategy, do_attack, do_wait, get_attack_targets
except ImportError as e:
    print(f"Failed to import simulator: {e}")
    sys.exit(1)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def get_observation(state: GameState, player: int):
    # Dummy observation: [round_num, num_own_units, num_enemy_units]
    return [state.round_num, len(state.player_units(player)), len(state.enemy_units(player))]

class RLStrategy:
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.name = "rl"
        
    def play_turn(self, state, player, rng):
        units = state.player_units(player)
        for unit in units:
            if not unit.alive or unit.has_acted: continue
            
            obs = get_observation(state, player)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = self.policy_net(obs_tensor)
            
            # Simple dummy action logic
            targets = get_attack_targets(state, unit)
            if targets:
                t = rng.choice(targets)
                do_attack(state, rng, unit, t)
            else:
                do_wait(unit)

def train(duration=300):
    start_time = time.time()
    obs_dim = 3
    act_dim = 4
    
    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    maps = list_maps()
    if not maps:
        print("No maps found.")
        return
        
    games_played = 0
    wins = 0
    
    print(f"Starting training for {duration} seconds...")
    while time.time() - start_time < duration:
        map_name = random.choice(maps)
        rl_strat = RLStrategy(policy)
        p2_strat = BalancedStrategy()
        
        seed = random.randint(0, 1000000)
        
        try:
            result = run_game(rl_strat, p2_strat, map_name, seed, verbose=False, replay=False)
            games_played += 1
            if result.winner == 1:
                wins += 1
            
            optimizer.zero_grad()
            # Dummy optimization
            optimizer.step()
        except Exception as e:
            print(f"Error running game: {e}")
            break
            
        if games_played % 10 == 0:
            print(f"[{time.time() - start_time:.1f}s] Played {games_played} games, wins: {wins}, win rate: {wins/games_played:.2%}")
            
    win_rate = wins / max(1, games_played)
    print(f"Final win rate: {win_rate:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300, help="Training duration in seconds")
    args = parser.parse_args()
    
    train(duration=args.duration)
