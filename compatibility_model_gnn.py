#!/usr/bin/env python3
"""
compatibility_model.py - player compatibility scoring with graph neural networks

pipeline overview:
1. loads statsbomb event data from csv batches
2. assigns possession ids using team change heuristic
3. computes expected threat (xt) map on 12x8 grid with bellman iterations
4. derives enriched player feature vectors (core metrics, position, pass style, pressure)
5. builds possession graphs with on-pitch players as nodes and passes as edges
6. trains a gnn to predict delta-xt (threat change) for pass events
7. extracts 128-d embeddings from trained gnn node representations
8. trains a learned scorer on player pair co-occurrence patterns
9. computes zone-level profiles (heatmaps, pass destinations, threat profiles)
10. outputs compatibility score via cli

dependencies: pandas, numpy, torch, torch-geometric

usage:
    python compatibility_model.py --playerA 5207 --playerB 5574 --epochs 3

optional flags:
    --csv-dir csv_data (directory with events_batch_*.csv files)
    --epochs 3 (training epochs)
    --batch-size 32 (graphs per batch)
    --min-events-player 0 (filter sparse players)
    --grid-x 12 --grid-y 8 (xt map resolution)
    --no-train (skip gnn training)
"""
from __future__ import annotations
import os
import glob
import math
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    # PyTorch Geometric (optional)
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAVE_PYG = True
except Exception:
    HAVE_PYG = False

################################################################################
# Utility helpers
################################################################################

def safe_div(a: float, b: float) -> float:
    """safe division with zero handling"""
    return a / b if b else 0.0


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """min-max normalization of vector"""
    if vec.size == 0:
        return vec
    vmin = vec.min()
    vmax = vec.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """jensen-shannon divergence between two distributions"""
    p = p.astype(float)
    q = q.astype(float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity between two vectors"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

################################################################################
# 1. Load events
################################################################################

def load_events(csv_dir: str) -> pd.DataFrame:
    """load and harmonize statsbomb csv event batches
    
    converts raw statsbomb event data into unified schema with:
    - match_id, team_id, player_id, event_type (base identifiers)
    - x_start, y_start, x_end, y_end (normalized to [0,1] pitch)
    - pass_recipient_id (for pass events)
    """
    files = sorted(glob.glob(os.path.join(csv_dir, 'events_batch_*.csv')))
    if not files:
        raise FileNotFoundError(f"No event batch CSVs found in {csv_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    events = pd.concat(dfs, ignore_index=True)

    # Derive event_type from type_name if necessary
    if 'event_type' not in events.columns:
        if 'type_name' in events.columns:
            events['event_type'] = events['type_name']
        else:
            raise ValueError("Neither 'event_type' nor 'type_name' present in events data")

    needed_base = ['match_id', 'team_id', 'player_id', 'event_type']
    for c in needed_base:
        if c not in events.columns:
            raise ValueError(f"Required column '{c}' missing in events data")

    # Initialize unified coordinate columns
    for coord in ['x_start', 'y_start', 'x_end', 'y_end']:
        if coord not in events.columns:
            events[coord] = np.nan

    # Start locations: prefer location_x/location_y then fallback to existing
    if 'location_x' in events.columns:
        events.loc[events['x_start'].isna(), 'x_start'] = events['location_x']
    if 'location_y' in events.columns:
        events.loc[events['y_start'].isna(), 'y_start'] = events['location_y']

    # End locations for passes
    if 'pass_end_location_x' in events.columns:
        pass_mask = events['event_type'] == 'Pass'
        events.loc[pass_mask, 'x_end'] = events.loc[pass_mask, 'pass_end_location_x']
    if 'pass_end_location_y' in events.columns:
        pass_mask = events['event_type'] == 'Pass'
        events.loc[pass_mask, 'y_end'] = events.loc[pass_mask, 'pass_end_location_y']

    # End locations for shots
    if 'shot_end_location_x' in events.columns:
        shot_mask = events['event_type'] == 'Shot'
        events.loc[shot_mask, 'x_end'] = events.loc[shot_mask, 'shot_end_location_x']
    if 'shot_end_location_y' in events.columns:
        shot_mask = events['event_type'] == 'Shot'
        events.loc[shot_mask, 'y_end'] = events.loc[shot_mask, 'shot_end_location_y']

    # If still missing end coords, fallback to start (zero movement assumption)
    events.loc[events['x_end'].isna(), 'x_end'] = events.loc[events['x_end'].isna(), 'x_start']
    events.loc[events['y_end'].isna(), 'y_end'] = events.loc[events['y_end'].isna(), 'y_start']

    # Pass recipient fallback
    if 'pass_recipient_id' not in events.columns:
        events['pass_recipient_id'] = np.nan

    # Sort chronologically within match if timestamp exists
    sort_cols = [c for c in ['match_id', 'timestamp'] if c in events.columns]
    if sort_cols:
        events = events.sort_values(sort_cols).reset_index(drop=True)

    # Normalize pitch dimensions to [0,1]
    for col in ['x_start', 'y_start', 'x_end', 'y_end']:
        if events[col].notna().any():
            vmax = events[col].max()
            if vmax and vmax > 2:
                events[col] = events[col] / vmax

    return events

################################################################################
# 2. Possession IDs
################################################################################

def assign_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """assign possession id by tracking team changes"""
    possession_ids = []
    current = 0
    last_team = None
    for _, row in events.iterrows():
        team = row['team_id']
        if last_team is None:
            current = 0
        elif team != last_team:
            current += 1
        possession_ids.append(current)
        last_team = team
    events['possession_id'] = possession_ids
    return events

################################################################################
# 3. xT Map + ΔxT
################################################################################

def get_zone(x: float, y: float, grid_x: int, grid_y: int) -> Optional[int]:
    """map pitch coordinates to zone index"""
    if np.isnan(x) or np.isnan(y):
        return None
    ix = min(grid_x - 1, max(0, int(x * grid_x)))
    iy = min(grid_y - 1, max(0, int(y * grid_y)))
    return iy * grid_x + ix  # row-major indexing


def compute_xt(events: pd.DataFrame, grid_x: int = 12, grid_y: int = 8,
               max_iter: int = 200, tol: float = 1e-5) -> Tuple[np.ndarray, pd.DataFrame]:
    Z = grid_x * grid_y
    # Assign zones
    events['zone_start'] = [get_zone(xs, ys, grid_x, grid_y) for xs, ys in zip(events['x_start'], events['y_start'])]
    events['zone_end'] = [get_zone(xe, ye, grid_x, grid_y) for xe, ye in zip(events['x_end'], events['y_end'])]

    # Transition counts
    trans = np.zeros((Z, Z), dtype=float)
    counts_zone = np.zeros(Z, dtype=float)
    shots_zone = np.zeros(Z, dtype=float)
    goals_zone = np.zeros(Z, dtype=float)

    for _, row in events.iterrows():
        zs = row['zone_start']
        ze = row['zone_end']
        valid_zs = isinstance(zs, (int, np.integer))
        valid_ze = isinstance(ze, (int, np.integer))
        if valid_zs:
            counts_zone[zs] += 1
            if row['event_type'] == 'Shot':
                shots_zone[zs] += 1
                # Determine goal
                shot_outcome = None
                for c in ['shot_outcome','outcome']:
                    if c in events.columns:
                        shot_outcome = row.get(c, None)
                        break
                if isinstance(shot_outcome, str) and shot_outcome.lower() == 'goal':
                    goals_zone[zs] += 1
        if valid_zs and valid_ze:
            trans[zs, ze] += 1

    # Normalize transitions row-wise
    for z in range(Z):
        row_sum = trans[z].sum()
        if row_sum > 0:
            trans[z] /= row_sum

    P_shot = np.array([safe_div(shots_zone[z], counts_zone[z]) for z in range(Z)])
    P_goal = np.array([safe_div(goals_zone[z], shots_zone[z]) for z in range(Z)])

    # Bellman iteration: xT[z] = P_shot[z]*P_goal[z] + sum P(z->z')*xT[z']
    xT = np.zeros(Z, dtype=float)
    for _ in range(max_iter):
        new_xT = P_shot * P_goal + trans.dot(xT)
        if np.max(np.abs(new_xT - xT)) < tol:
            xT = new_xT
            break
        xT = new_xT

    # ΔxT per event
    delta_xt = []
    for _, row in events.iterrows():
        zs = row['zone_start']
        ze = row['zone_end']
        if not (isinstance(zs, (int, np.integer)) and isinstance(ze, (int, np.integer))):
            delta_xt.append(0.0)
        else:
            delta_xt.append(xT[ze] - xT[zs])
    events['delta_xT'] = delta_xt

    return xT, events

################################################################################
# 4. Player feature vectors (enriched with position, pass type, pressure context)
################################################################################

def get_player_position_encoding(events: pd.DataFrame, pid: int) -> np.ndarray:
    """One-hot encode most common position for player (0 if no position found)."""
    p_events = events[events['player_id'] == pid]
    positions = p_events['position_id'].dropna()
    if len(positions) == 0:
        return np.zeros(7)  # 7 common positions
    most_common_pos = int(positions.mode()[0]) if len(positions.mode()) > 0 else 0
    # Common positions: 1=GK, 2=Def, 3=MF, 4=FW (map to 0-6 bins)
    pos_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    pos_idx = pos_map.get(most_common_pos, 0)
    enc = np.zeros(7)
    enc[pos_idx] = 1.0
    return enc


def get_player_pass_signature(events: pd.DataFrame, pid: int) -> np.ndarray:
    """Compute pass characteristics: avg_length, avg_angle, short_pct, long_pct, aerial_pct."""
    p_events = events[events['player_id'] == pid]
    passes = p_events[p_events['event_type'] == 'Pass']
    
    if len(passes) == 0:
        return np.zeros(5)
    
    # Average pass length
    pass_lengths = passes['pass_length'].dropna()
    avg_length = pass_lengths.mean() if len(pass_lengths) > 0 else 0.0
    
    # Average pass angle (absolute)
    pass_angles = passes['pass_angle'].dropna().abs()
    avg_angle = pass_angles.mean() if len(pass_angles) > 0 else 0.0
    
    # Pass type distribution (if available)
    short_pct = ((passes['pass_type_name'] == 'Short Pass').sum() / len(passes)) if len(passes) > 0 else 0.0
    long_pct = ((passes['pass_type_name'] == 'Long Pass').sum() / len(passes)) if len(passes) > 0 else 0.0
    
    # Height distribution (aerial: pass_height_name contains 'High' or 'Head')
    aerial_pct = 0.0
    if 'pass_height_name' in passes.columns:
        aerial_mask = passes['pass_height_name'].fillna('').str.contains('High', case=False, na=False)
        aerial_pct = (aerial_mask.sum() / len(passes)) if len(passes) > 0 else 0.0
    
    return np.array([avg_length, avg_angle, short_pct, long_pct, aerial_pct], dtype=float)


def get_player_pressure_profile(events: pd.DataFrame, pid: int) -> np.ndarray:
    """Compute under_pressure decision-making: pct_events_under_pressure, pass_acc_under_pressure."""
    p_events = events[events['player_id'] == pid]
    
    if len(p_events) == 0:
        return np.zeros(2)
    
    pressure_events = (p_events['under_pressure'].fillna(False) == True).sum()
    pressure_pct = pressure_events / len(p_events) if len(p_events) > 0 else 0.0
    
    # Pass accuracy while under pressure
    passes_under_pressure = p_events[(p_events['event_type'] == 'Pass') & 
                                     (p_events['under_pressure'].fillna(False) == True)]
    if len(passes_under_pressure) > 0:
        completed = (passes_under_pressure['pass_recipient_id'].notna()).sum()
        pressure_pass_acc = completed / len(passes_under_pressure)
    else:
        pressure_pass_acc = 0.0
    
    return np.array([pressure_pct, pressure_pass_acc], dtype=float)


def compute_player_features(events: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Enhanced player feature vectors combining:
    - Core metrics (9): passes, accuracy, dribbles, tackles, interceptions, clearances, key_passes, xT_created, xT_received
    - Position (7): one-hot encoded position
    - Pass signature (5): avg_length, avg_angle, short_pct, long_pct, aerial_pct
    - Pressure profile (2): pressure_pct, pressure_pass_acc
    Total: 23 dimensions
    """
    player_ids = events['player_id'].dropna().unique()
    features = {}
    
    for pid in player_ids:
        p_events = events[events['player_id'] == pid]
        
        # Core metrics (9 dims)
        total_passes = (p_events['event_type'] == 'Pass').sum()
        completed_passes = ((p_events['event_type'] == 'Pass') & (p_events['pass_recipient_id'].notna())).sum()
        pass_accuracy = safe_div(completed_passes, total_passes)
        dribbles = (p_events['event_type'] == 'Dribble').sum()
        tackles = (p_events['event_type'] == 'Tackle').sum()
        interceptions = (p_events['event_type'] == 'Interception').sum()
        clearances = (p_events['event_type'] == 'Clearance').sum() if 'Clearance' in events['event_type'].unique() else 0
        key_passes = 0  # TODO: link to shots on target
        xT_created = p_events['delta_xT'].sum()
        received_events = events[events['pass_recipient_id'] == pid]
        xT_received = received_events['delta_xT'].sum()
        
        core = np.array([
            total_passes, pass_accuracy, dribbles, tackles, interceptions,
            clearances, key_passes, xT_created, xT_received
        ], dtype=float)
        
        # Position encoding (7 dims)
        position_enc = get_player_position_encoding(events, pid)
        
        # Pass signature (5 dims)
        pass_sig = get_player_pass_signature(events, pid)
        
        # Pressure profile (2 dims)
        pressure_prof = get_player_pressure_profile(events, pid)
        
        # Concatenate all features
        vec = np.concatenate([core, position_enc, pass_sig, pressure_prof])
        features[pid] = vec

    # Normalize each component across players (per-dimension)
    if features:
        mat = np.vstack(list(features.values()))
        mat_norm = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            mat_norm[:, j] = normalize_vec(col)
        for i, pid in enumerate(features.keys()):
            features[pid] = mat_norm[i]
    
    return features

################################################################################
# 5. Graph construction (enriched with on-pitch context)
################################################################################

def build_graphs(events: pd.DataFrame, player_features: Dict[int, np.ndarray],
                 sample_limit: Optional[int] = None) -> Tuple[List[Data], Dict[int, List[int]]]:
    """Build pass event graphs with all on-pitch players as nodes and enriched edges.
    
    Edge attributes include:
    - Positional (x_start, y_start, x_end, y_end): 4 dims
    - Threat (delta_xT): 1 dim
    - Pass characteristics (pass_length, pass_angle, pass_height_id, body_part_id, outcome): 5 dims
    Total: 10 dims per edge
    
    Returns:
        graphs: List of Data objects (pass events)
        pass_player_pairs: Dict mapping (pidA, pidB) tuples to list of pass indices where both are on-pitch
    """
    if not HAVE_TORCH:
        print("Torch not available: graph building will be skipped.")
        return [], {}
    
    pass_events = events[events['event_type'] == 'Pass'].reset_index(drop=True)
    # Use ALL graphs - no sampling limit

    graphs = []
    feature_dim = len(next(iter(player_features.values()))) if player_features else 0
    pass_player_pairs = {}  # Track which passes involve each player pair
    
    for pass_idx, (_, row) in enumerate(pass_events.iterrows()):
        sender = row['player_id']
        receiver = row['pass_recipient_id']
        if pd.isna(sender) or pd.isna(receiver):
            continue
        sender = int(sender)
        receiver = int(receiver)
        
        # Get all players in possession (match + period based)
        match_id = row.get('match_id')
        possession_id = row.get('possession_id')
        
        # Get all events in this possession
        poss_events = events[
            (events['match_id'] == match_id) & 
            (events['possession_id'] == possession_id) &
            (events['player_id'].notna())
        ]
        
        # All unique players in this possession (on-pitch context)
        on_pitch = list(set(poss_events['player_id'].dropna().unique()))
        on_pitch = [int(p) for p in on_pitch if p in player_features]
        
        if len(on_pitch) < 2 or sender not in on_pitch or receiver not in on_pitch:
            continue
        
        # Build node feature matrix
        x = []
        pid_to_idx = {}
        for idx, pid in enumerate(on_pitch):
            pid_to_idx[pid] = idx
            feat = player_features.get(pid, np.zeros(feature_dim))
            x.append(feat)
        x = torch.tensor(np.vstack(x), dtype=torch.float)
        
        # Build enriched edge attributes for the main pass
        x_start = row.get('x_start', 0.0) or 0.0
        y_start = row.get('y_start', 0.0) or 0.0
        x_end = row.get('x_end', 0.0) or 0.0
        y_end = row.get('y_end', 0.0) or 0.0
        delta_xT = row.get('delta_xT', 0.0) or 0.0
        
        # Pass characteristics
        pass_length = row.get('pass_length', 0.0) or 0.0
        pass_angle = row.get('pass_angle', 0.0) or 0.0
        pass_height_id = row.get('pass_height_id', 1.0) or 1.0  # Default to normal height
        body_part_id = row.get('pass_body_part_id', 1.0) or 1.0  # Default to foot
        
        # Pass outcome (success=1.0, failure=0.0)
        pass_outcome = 1.0 if pd.notna(row.get('pass_recipient_id')) else 0.0
        
        # Main pass edge with enriched attributes
        sender_idx = pid_to_idx[sender]
        receiver_idx = pid_to_idx[receiver]
        
        edges = [[sender_idx, receiver_idx]]
        edge_attrs = [[
            x_start, y_start, x_end, y_end,  # Position (4)
            delta_xT,                          # Threat (1)
            pass_length, pass_angle,           # Pass style (2)
            pass_height_id, body_part_id,      # Technical (2)
            pass_outcome                       # Outcome (1)
        ]]  # Total: 10 dims
        
        # Add edges from sender to all teammates (context, zero attributes)
        for tidx, tid in enumerate(on_pitch):
            if tid != sender and tid != receiver:
                edges.append([sender_idx, tidx])
                edge_attrs.append([0.0] * 10)  # Context edges: all zeros
        
        if not edges:
            continue
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([delta_xT], dtype=torch.float)
        
        if HAVE_PYG:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            class SimpleData:
                def __init__(self, x, edge_index, edge_attr, y):
                    self.x = x
                    self.edge_index = edge_index
                    self.edge_attr = edge_attr
                    self.y = y
            data = SimpleData(x, edge_index, edge_attr, y)
        
        graphs.append(data)
        
        # Track this pass for player pair analysis
        for i, p1 in enumerate(on_pitch):
            for p2 in on_pitch[i+1:]:
                key = tuple(sorted([int(p1), int(p2)]))
                if key not in pass_player_pairs:
                    pass_player_pairs[key] = []
                pass_player_pairs[key].append(pass_idx)
    
    print(f"Built {len(graphs)} graphs covering {len(pass_player_pairs)} player pairs")
    return graphs, pass_player_pairs

################################################################################
# 6. Train GNN
################################################################################

class GoalNetGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        if HAVE_PYG:
            # GCN layers with edge attributes support
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.conv3 = GCNConv(hidden, hidden)
            # Edge feature MLP (input: edge_attr + node_pair features)
            self.edge_mlp = nn.Sequential(
                nn.Linear(10, 32), nn.ReLU(),  # 10 edge dims -> 32
                nn.Linear(32, 16)
            )
        else:
            # Fallback MLP on mean pooled features
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU()
            )
        self.readout = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, batch_graphs: List[Data]):
        preds = []
        for g in batch_graphs:
            x = g.x
            if HAVE_PYG:
                # Process edge features through MLP
                if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                    edge_features = self.edge_mlp(g.edge_attr)
                else:
                    edge_features = None
                
                # Apply GCN layers
                x1 = self.conv1(x, g.edge_index)
                x1 = torch.relu(x1)
                x2 = self.conv2(x1, g.edge_index)
                x2 = torch.relu(x2)
                x3 = self.conv3(x2, g.edge_index)
                x3 = torch.relu(x3)
                pooled = x3.mean(dim=0)
            else:
                pooled = self.mlp(x.mean(dim=0))
            pred = self.readout(pooled)
            preds.append(pred.squeeze())
        return torch.stack(preds)


################################################################################
# NEW: Learned Compatibility Scorer (Graph-Centric)
################################################################################

class LearnedCompatibilityScorer(nn.Module):
    """
    Learned scoring network that uses GNN embeddings as primary signal.
    
    Input: Two player embeddings (learned by GNN)
    Output: Compatibility score [0, 1]
    
    This replaces the hardcoded 7-signal fusion with a learned neural network
    that discovers what makes two players compatible based on graph structure.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Main scoring network: concat embeddings -> score
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary task: pass quality network
        # Learns: "given two players, how good is a pass between them?"
        self.pass_quality = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary task: threat flow network
        # Learns: "how well does threat propagate from A to B?"
        self.threat_flow = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary task: position synergy network
        # Learns: "how complementary are these positions?"
        self.position_synergy = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emb_A: torch.Tensor, emb_B: torch.Tensor):
        """
        Compute compatibility score from two embeddings.
        
        Args:
            emb_A: Player A embedding [hidden_dim]
            emb_B: Player B embedding [hidden_dim]
        
        Returns:
            score: Compatibility [0, 1]
            pass_quality: Learned pass quality
            threat_flow: Learned threat propagation
            position_synergy: Learned position synergy
        """
        combined = torch.cat([emb_A, emb_B], dim=-1)
        
        score = self.scorer(combined)
        pass_qual = self.pass_quality(combined)
        threat = self.threat_flow(combined)
        synergy = self.position_synergy(combined)
        
        return score, pass_qual, threat, synergy
    
    def score_pair(self, emb_A: np.ndarray, emb_B: np.ndarray) -> float:
        """Score a pair from numpy arrays (inference mode)."""
        with torch.no_grad():
            emb_A_t = torch.tensor(emb_A, dtype=torch.float32)
            emb_B_t = torch.tensor(emb_B, dtype=torch.float32)
            score, _, _, _ = self.forward(emb_A_t, emb_B_t)
            return float(score.item())

################################################################################
# 7. Create Compatibility Training Dataset
################################################################################

def create_compatibility_dataset(embeddings: Dict[int, np.ndarray],
                                  pass_player_pairs: Dict[tuple, List[int]],
                                  events: pd.DataFrame,
                                  sample_negatives: int = 500) -> Tuple[List[tuple], List[float]]:
    """
    Create training pairs for compatibility scorer.
    
    Positive pairs: Players who frequently pass to each other
    Negative pairs: Random players who don't co-occur
    
    FIX: Use percentile-based labeling to handle extreme class imbalance.
    - Positive labels (top 25% co-occurrence): [0.5, 1.0]
    - Negative labels (non co-occurring): [0.0, 0.3]
    This prevents the network from collapsing to near-zero outputs.
    
    Returns:
        pairs: List of (emb_A, emb_B) tuples
        labels: List of labels [0, 1] (1 = highly compatible, 0 = incompatible)
    """
    pairs = []
    labels = []
    
    player_ids = list(embeddings.keys())
    
    # POSITIVE PAIRS: Players who pass together (co-occur frequently)
    pass_counts = [len(v) for v in pass_player_pairs.values()]
    pass_75_percentile = np.percentile(pass_counts, 75) if pass_counts else 1.0
    pass_90_percentile = np.percentile(pass_counts, 90) if pass_counts else 1.0
    
    print(f"   Co-occurrence pass count stats:")
    print(f"      - Min: {min(pass_counts)}, Max: {max(pass_counts)}, Mean: {np.mean(pass_counts):.1f}")
    print(f"      - 75th percentile: {pass_75_percentile:.0f}")
    print(f"      - 90th percentile: {pass_90_percentile:.0f}")
    
    for (pid_a, pid_b), pass_list in pass_player_pairs.items():
        if pid_a in embeddings and pid_b in embeddings:
            # Label: HIGH for frequently co-occurring (top 25%), LOW for others
            # Use percentile ranking instead of raw normalization
            pass_count = len(pass_list)
            if pass_count >= pass_75_percentile:
                # Top tier: score [0.7, 1.0]
                co_freq = 0.7 + 0.3 * (pass_count - pass_75_percentile) / max(1, pass_90_percentile - pass_75_percentile)
            else:
                # Lower tier: score [0.4, 0.7]
                co_freq = 0.4 + 0.3 * (pass_count / pass_75_percentile)
            
            co_freq = max(0.4, min(1.0, co_freq))  # Clamp to [0.4, 1.0] for positives
            
            pairs.append((embeddings[pid_a], embeddings[pid_b]))
            labels.append(co_freq)
            
            # Add reverse pair
            pairs.append((embeddings[pid_b], embeddings[pid_a]))
            labels.append(co_freq)
    
    # NEGATIVE PAIRS: Random players who don't co-occur often
    for _ in range(min(sample_negatives, len(player_ids) * 10)):
        pid_a, pid_b = np.random.choice(player_ids, 2, replace=False)
        pair_key = tuple(sorted([pid_a, pid_b]))
        
        # Only if they don't co-occur frequently
        if pair_key not in pass_player_pairs or len(pass_player_pairs[pair_key]) < 2:
            # Negative labels: [0.0, 0.3] to allow network to distinguish
            neg_label = np.random.uniform(0.0, 0.3)
            pairs.append((embeddings[pid_a], embeddings[pid_b]))
            labels.append(neg_label)
    
    return pairs, labels


################################################################################
# 7. Extract player embeddings (pair-scoped)
################################################################################

def extract_embeddings(model: GoalNetGNN, graphs: List[Data], player_features: Dict[int, np.ndarray],
                      pass_player_pairs: Dict[int, List[int]] = None) -> Dict[int, np.ndarray]:
    """Extract embeddings, optionally scoped to a player pair's shared graphs."""
    embeddings: Dict[int, List[np.ndarray]] = {}
    
    if not HAVE_TORCH or not graphs or model is None:
        # If no training, just return raw player features padded to consistent size
        return {pid: feats for pid, feats in player_features.items()}
    
    model.eval()
    with torch.no_grad():
        for g_idx, g in enumerate(graphs):
            x = g.x
            if HAVE_PYG:
                h1 = model.conv1(x, g.edge_index)
                h1 = torch.relu(h1)
                h2 = model.conv2(h1, g.edge_index)
                h2 = torch.relu(h2)
                node_embs = h2
            else:
                node_embs = model.mlp[0](x)
                node_embs = torch.relu(node_embs)
            
            # For each node, store embedding keyed by best-match player
            for idx, emb in enumerate(node_embs):
                node_feat = x[idx].numpy()
                pid_match = min(player_features.keys(), 
                              key=lambda pid: np.linalg.norm(player_features[pid] - node_feat))
                embeddings.setdefault(pid_match, []).append(emb.numpy())
    
    # Aggregate embeddings
    final = {}
    for pid, embs in embeddings.items():
        final[pid] = np.mean(np.vstack(embs), axis=0)
    
    # Fallback for missing players
    for pid, feats in player_features.items():
        if pid not in final:
            final[pid] = feats
    
    return final

################################################################################
# 8. Zone profiles
################################################################################

def compute_zone_profiles(events: pd.DataFrame, xT_map: np.ndarray, grid_x: int, grid_y: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    Z = grid_x * grid_y
    heatmaps = {}
    pass_dists = {}
    xt_profiles = {}
    players = events['player_id'].dropna().unique()
    for pid in players:
        p_events = events[events['player_id'] == pid]
        heat = np.zeros(Z)
        for z in p_events['zone_start']:
            if isinstance(z, (int, np.integer)):
                heat[z] += 1
        passes = p_events[p_events['event_type'] == 'Pass']
        pass_dist = np.zeros(Z)
        for z in passes['zone_end']:
            if isinstance(z, (int, np.integer)):
                pass_dist[z] += 1
        xt_prof = np.zeros(Z)
        for z, dx in zip(p_events['zone_start'], p_events['delta_xT']):
            if isinstance(z, (int, np.integer)):
                xt_prof[z] += dx
        heatmaps[int(pid)] = normalize_vec(heat)
        pass_dists[int(pid)] = normalize_vec(pass_dist)
        xt_profiles[int(pid)] = normalize_vec(xt_prof)
    return heatmaps, pass_dists, xt_profiles

################################################################################
# 9. Compatibility score
################################################################################

def load_player_names(csv_dir: str = 'csv_data') -> Dict[int, str]:
    """Load player ID -> name mapping from lineups CSV."""
    lineups_path = os.path.join(csv_dir, 'lineups.csv')
    try:
        lineups = pd.read_csv(lineups_path)
        # Convert player_id to int to avoid float issues
        lineups['player_id'] = lineups['player_id'].astype(int)
        name_map = dict(zip(lineups['player_id'], lineups['player_name']))
        return name_map
    except Exception as e:
        print(f"Warning: Could not load lineups: {e}")
        return {}


def diagnose_players(events: pd.DataFrame, playerA: int, playerB: int, 
                     embeddings: Dict[int, np.ndarray], 
                     player_features: Dict[int, np.ndarray],
                     name_map: Dict[int, str] = None):
    """Debug: check if players exist in data and embeddings."""
    if name_map is None:
        name_map = {}
    
    name_a = name_map.get(playerA, "Unknown")
    name_b = name_map.get(playerB, "Unknown")
    
    print(f"\n=== Player Diagnostic ===")
    print(f"Looking for: A={playerA} ({name_a}), B={playerB} ({name_b})")
    
    # Check raw events
    a_events = (events['player_id'] == playerA).sum()
    b_events = (events['player_id'] == playerB).sum()
    print(f"Events: A={a_events}, B={b_events}")
    
    # Check features
    a_in_features = playerA in player_features
    b_in_features = playerB in player_features
    print(f"In features: A={a_in_features}, B={b_in_features}")
    
    # Check embeddings
    a_in_emb = playerA in embeddings
    b_in_emb = playerB in embeddings
    print(f"In embeddings: A={a_in_emb}, B={b_in_emb}")
    
    # List available players (sample)
    avail = list(embeddings.keys())[:10]
    print(f"Sample available player IDs: {avail}")
    print(f"Total players in embeddings: {len(embeddings)}")
    print()


################################################################################
# 10. Main orchestration
################################################################################

def run_pipeline(args):
    print("\n" + "="*80)
    print("PLAYER COMPATIBILITY SCORER - GRAPH ML PIPELINE")
    print("="*80)
    
    # loading events from statsbomb data
    print("\nStep 1: Loading StatsBomb event data")
    print(f"   Reading from: {args.csv_dir}")
    events = load_events(args.csv_dir)
    print(f"   Successfully loaded {len(events):,} events")
    print(f"   Available columns: {list(events.columns[:10])}...")
    
    # loading player name mappings from lineups
    print("\nStep 2: Loading player name mapping")
    name_map = load_player_names(args.csv_dir)
    print(f"   Found names for {len(name_map):,} players")
    
    # assigning possession ids by tracking team changes
    print("\nStep 3: Assigning possession IDs")
    events = assign_possessions(events)
    num_possessions = events['possession_id'].max() + 1
    print(f"   Identified {num_possessions:,} distinct possessions")
    
    # computing expected threat map using bellman iterations on zone transitions
    print("\nStep 4: Computing Expected Threat (xT) map")
    print(f"   Using grid resolution: {args.grid_x}x{args.grid_y}")
    xT_map, events = compute_xt(events, grid_x=args.grid_x, grid_y=args.grid_y)
    print(f"   Generated xT map with shape {xT_map.shape}")
    print(f"   Threat values range: [{xT_map.min():.4f}, {xT_map.max():.4f}]")
    print(f"   Computed delta xT for all events")
    
    # computing rich player feature vectors including position encoding, pass style, and pressure handling
    print("\nStep 5: Computing enriched player feature vectors")
    player_features = compute_player_features(events)
    print(f"   Generated features for {len(player_features):,} players")
    sample_player_id = list(player_features.keys())[0]
    sample_features = player_features[sample_player_id]
    print(f"   Each player has {len(sample_features)} feature dimensions:")
    print(f"      - 9 core metrics (passes, accuracy, actions)")
    print(f"      - 7 position encoding (one-hot)")
    print(f"      - 5 pass signature (length, angle, types)")
    print(f"      - 2 pressure profile (composure under pressure)")

    # optionally filter out players with very few events
    if args.min_events_player > 0:
        print(f"\nStep 5b: Filtering sparse players (minimum {args.min_events_player} events)")
        counts = events['player_id'].value_counts()
        original_players = len(player_features)
        player_features = {pid: feat for pid, feat in player_features.items() if counts.get(pid, 0) >= args.min_events_player}
        filtered_players = len(player_features)
        print(f"   Retained {filtered_players:,} players (removed {original_players - filtered_players:,} sparse ones)")

    # building possession graphs where passes and player context are captured
    print("\nStep 6: Building possession graphs from pass events")
    graphs, pass_player_pairs = build_graphs(events, player_features, sample_limit=None)
    print(f"   Created {len(graphs):,} graph structures from pass events")
    print(f"   These graphs represent {len(pass_player_pairs):,} unique player pairs")
    avg_nodes = np.mean([g.x.shape[0] for g in graphs]) if graphs else 0
    avg_edges = np.mean([g.edge_index.shape[1] for g in graphs]) if graphs else 0
    print(f"   Average graph composition: {avg_nodes:.1f} players, {avg_edges:.1f} pass connections")

    if HAVE_TORCH and player_features and not args.no_train and len(graphs) >= 10:
        # training graph neural network to learn threat-aware player embeddings
        print("\n" + "="*80)
        print("PHASE 1: Training Graph Neural Network (Threat Prediction)")
        print("="*80)
        
        gnn_model = GoalNetGNN(in_dim=len(next(iter(player_features.values()))))
        gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
        gnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gnn_optimizer, T_max=args.epochs)
        
        print(f"\nGraph neural network architecture:")
        print(f"   Input layer: {len(next(iter(player_features.values())))} player features")
        print(f"   Hidden layers:")
        print(f"      - Graph convolution: {len(next(iter(player_features.values())))} → 128 dims")
        print(f"      - Graph convolution: 128 → 128 dims")
        print(f"      - Graph convolution: 128 → 128 dims")
        print(f"      - Readout network: 128 → 64 → 32 → 1 (threat prediction)")
        
        gnn_model.train()
        print(f"\nTraining configuration:")
        print(f"   Dataset: {len(graphs):,} possession graphs")
        print(f"   Training epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size} graphs per iteration")
        print(f"   Learning rate: 1e-3 (with cosine annealing schedule)")
        
        print(f"\nStarting training loop...")
        for epoch in range(args.epochs):
            random.shuffle(graphs)
            total_loss = 0.0
            total_weight = 0.0
            batch_count = 0
            
            for i in range(0, len(graphs), args.batch_size):
                batch = graphs[i:i+args.batch_size]
                preds = gnn_model(batch)
                y_true = torch.stack([g.y.squeeze() for g in batch])
                
                base_loss = (preds - y_true) ** 2
                weights = torch.ones_like(base_loss)
                loss = (base_loss * weights).sum() / weights.sum() if weights.sum() > 0 else base_loss.mean()
                
                gnn_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
                gnn_optimizer.step()
                
                total_loss += loss.item() * len(batch)
                total_weight += len(batch)
                batch_count += 1
            
            gnn_scheduler.step()
            avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
            print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Processed {batch_count:,} batches")
        
        print(f"\nPhase 1 complete: GNN training finished")
        
        # extracting learned player embeddings from trained graph network
        print(f"\nStep 7: Extracting player embeddings from trained GNN")
        embeddings = extract_embeddings(gnn_model, graphs, player_features, pass_player_pairs)
        print(f"   Generated embeddings for {len(embeddings):,} players")
        sample_emb = next(iter(embeddings.values()))
        print(f"   Each embedding has {len(sample_emb)} dimensions")
        print(f"   These represent learned threat-aware player characteristics")
        
        # training a separate scorer to predict player compatibility from embeddings
        print("\n" + "="*80)
        print("PHASE 2: Training Learned Compatibility Scorer (Graph-Centric)")
        print("="*80)
        
        print(f"\nStep 8: Creating compatibility training dataset")
        train_pairs, train_labels = create_compatibility_dataset(
            embeddings, pass_player_pairs, events, sample_negatives=500
        )
        print(f"   Created {len(train_pairs):,} training examples")
        
        num_high_compat = sum(1 for label in train_labels if label > 0.6)
        num_medium_compat = sum(1 for label in train_labels if 0.3 < label <= 0.6)
        num_low_compat = sum(1 for label in train_labels if label <= 0.3)
        print(f"   Label distribution:")
        print(f"      - High compatibility (>0.6): {num_high_compat:,}")
        print(f"      - Medium compatibility (0.3-0.6): {num_medium_compat:,}")
        print(f"      - Low compatibility (<=0.3): {num_low_compat:,}")
        print(f"   Label range: [{min(train_labels):.2f}, {max(train_labels):.2f}]")
        print(f"   Average label value: {np.mean(train_labels):.3f}")
        
        if len(train_pairs) > 0:
            # initializing the compatibility scorer network
            print(f"\nScorer architecture:")
            embedding_dim = len(next(iter(embeddings.values())))
            scorer = LearnedCompatibilityScorer(embedding_dim=embedding_dim)
            print(f"   Input: {embedding_dim * 2} dimensions (concatenated player embeddings)")
            print(f"   Main scoring path: {embedding_dim * 2} → 256 → 128 → 64 → 32 → 1")
            print(f"   Auxiliary learning tasks:")
            print(f"      - Pass quality prediction network")
            print(f"      - Threat flow assessment network")
            print(f"      - Position synergy evaluation network")
            print(f"   Loss function: weighted mean squared error with auxiliary tasks")
            
            scorer_optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-3)
            scorer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(scorer_optimizer, T_max=args.epochs)
            
            scorer.train()
            print(f"\nTraining configuration:")
            print(f"   Training pairs: {len(train_pairs):,}")
            print(f"   Training epochs: {args.epochs}")
            print(f"   Batch size: {args.batch_size} pairs per iteration")
            
            # training loop for compatibility scorer
            print(f"\nStarting training loop...")
            for epoch in range(args.epochs):
                indices = np.random.permutation(len(train_pairs))
                total_loss = 0.0
                batch_count = 0
                
                for i in range(0, len(indices), args.batch_size):
                    batch_idx = indices[i:i+args.batch_size]
                    
                    # prepare batch
                    batch_emb_As = []
                    batch_emb_Bs = []
                    batch_labels = []
                    
                    for j in batch_idx:
                        emb_a = train_pairs[j][0]
                        emb_b = train_pairs[j][1]
                        
                        # ensure fixed dimensions
                        if len(emb_a) != embedding_dim:
                            emb_a = np.pad(emb_a, (0, embedding_dim - len(emb_a)))[:embedding_dim]
                        if len(emb_b) != embedding_dim:
                            emb_b = np.pad(emb_b, (0, embedding_dim - len(emb_b)))[:embedding_dim]
                        
                        batch_emb_As.append(emb_a)
                        batch_emb_Bs.append(emb_b)
                        batch_labels.append(train_labels[j])
                    
                    emb_As = torch.tensor(np.vstack(batch_emb_As), dtype=torch.float32)
                    emb_Bs = torch.tensor(np.vstack(batch_emb_Bs), dtype=torch.float32)
                    labels = torch.tensor(np.array(batch_labels), dtype=torch.float32)
                    
                    # forward pass through all networks
                    scores, pass_qual, threat, synergy = scorer(emb_As, emb_Bs)
                    
                    # class weight: penalize low labels (negative pairs) less
                    # high labels (positive pairs): weight 1.0
                    # low labels (negative pairs): weight 0.5 (less critical to fit perfectly)
                    weights = torch.where(labels > 0.4, torch.ones_like(labels), 0.5 * torch.ones_like(labels))
                    
                    # multi-task loss with weighted mse
                    main_loss = (weights * (scores.squeeze() - labels) ** 2).mean()
                    aux_loss = (
                        0.1 * (weights * (pass_qual.squeeze() - labels) ** 2).mean() +
                        0.1 * (weights * (threat.squeeze() - labels) ** 2).mean() +
                        0.1 * (weights * (synergy.squeeze() - labels) ** 2).mean()
                    )
                    loss = main_loss + aux_loss
                    
                    scorer_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                    scorer_optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                scorer_scheduler.step()
                avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
                print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Processed {batch_count:,} batches")
            
            print(f"\nPhase 2 complete: Compatibility scorer trained")
        else:
            print("Skipping scorer training: insufficient training pairs")
            scorer = None
    else:
        print("Skipping neural network training: torch not available or insufficient data")
        gnn_model = None
        embeddings = player_features
        scorer = None
        embeddings = extract_embeddings(gnn_model, graphs, player_features, pass_player_pairs)
    
    # computing zone profiles and threat heatmaps
    print(f"\nStep 9: Computing zone profiles and threat heatmaps")
    heatmaps, pass_dists, xt_profiles = compute_zone_profiles(events, xT_map, args.grid_x, args.grid_y)
    print(f"   Computed heatmaps for {len(heatmaps):,} players")
    print(f"   Pitch divided into {args.grid_x} x {args.grid_y} = {args.grid_x * args.grid_y} zones")

    # diagnosing target players for compatibility scoring
    print(f"\nStep 10: Diagnosing target players")
    diagnose_players(events, args.playerA, args.playerB, embeddings, player_features, name_map)

    # final step: computing compatibility score for target pair
    print("\n" + "="*80)
    print("FINAL STEP: Computing Compatibility Score")
    print("="*80 + "\n")
    
    if scorer is None:
        print("Error: scorer was not successfully initialized")
        print("Cannot proceed without trained neural network")
        return 0.0
    
    print("Using learned compatibility scorer (trained on co-occurrence patterns)")
    print("Input: GNN-learned player embeddings (128 dimensions each)")
    print("Method: Multi-task neural network with auxiliary learning objectives")
    print()
    
    if args.playerA in embeddings and args.playerB in embeddings:
        print(f"Scoring player pair:")
        print(f"   Player A: {args.playerA} ({name_map.get(args.playerA, 'Unknown')})")
        print(f"   Player B: {args.playerB} ({name_map.get(args.playerB, 'Unknown')})")
        print(f"   Embedding dimensions: {embeddings[args.playerA].shape} and {embeddings[args.playerB].shape}")
        print()
        
        score = scorer.score_pair(embeddings[args.playerA], embeddings[args.playerB])
    else:
        print(f"Error: one or both players not found in embeddings")
        print(f"Available players in dataset: {len(embeddings)}")
        score = 0.0
    
    name_a = name_map.get(args.playerA, "Unknown")
    name_b = name_map.get(args.playerB, "Unknown")
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Compatibility Score: {args.playerA} {name_a} -> {args.playerB} {name_b}")
    print(f"Score: {score:.4f}")
    print("="*80 + "\n")
    
    return score

################################################################################
# CLI
################################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Player compatibility score model (GoalNet-lite)")
    p.add_argument('--csv-dir', default='csv_data', help='Directory containing events_batch_*.csv')
    p.add_argument('--playerA', type=int, required=True, help='Player ID A')
    p.add_argument('--playerB', type=int, required=True, help='Player ID B')
    p.add_argument('--epochs', type=int, default=3, help='Training epochs for GNN')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size of graphs')
    p.add_argument('--sample-pass-graphs', type=int, default=500, help='Cap number of pass graphs sampled')
    p.add_argument('--min-events-player', type=int, default=0, help='Minimum events per player to include')
    p.add_argument('--grid-x', type=int, default=12, help='xT grid X dimension')
    p.add_argument('--grid-y', type=int, default=8, help='xT grid Y dimension')
    p.add_argument('--no-train', action='store_true', help='Skip GNN training and use raw features as embeddings')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise
