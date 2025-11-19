# Graph-Based Player Compatibility Model (GoalNet-Lite)

A sophisticated Machine Learning pipeline that uses Graph Neural Networks (GNNs) to quantify compatibility between football players by learning embeddings from match event data and predicting compatibility using a multi-task neural network.

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install torch pandas numpy tqdm scikit-learn
pip install torch-geometric  # Optional but recommended
```

### Running the Model

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320
```

This computes a compatibility score between player 5503 (Messi) and player 4320 (Neymar Jr).

**Finding Player IDs:**
```bash
grep -i "messi" csv_data/lineups.csv  # Linux/macOS
findstr /I "messi" csv_data\lineups.csv  # Windows
```

### Configuration Options

```bash
python compatibility_model_gnn.py \
  --playerA 5503 \
  --playerB 4320 \
  --epochs 5 \              # GNN training epochs (default: 3)
  --batch-size 32 \         # Batch size (default: 32)
  --grid-x 12 \             # xT grid columns (default: 12)
  --grid-y 8 \              # xT grid rows (default: 8)
  --min-events-player 30    # Filter sparse players (default: 0)
```

### Output Interpretation

The model outputs a compatibility score (0-1):
- **0.80-1.00**: Elite partnership potential
- **0.60-0.79**: Strong compatibility  
- **0.40-0.59**: Moderate compatibility
- **0.20-0.39**: Poor compatibility
- **0.00-0.19**: Very poor compatibility

## Overview

The pipeline implemented in `compatibility_model_gnn.py` performs the following:

The pipeline implemented in `compatibility_model_gnn.py` performs the following:

1. **Loads StatsBomb Event Data**: Processes raw match events from CSV batches.
2. **Calculates Expected Threat (xT)**: Computes grid-based threat values using a Markov Chain approach.
3. **Constructs Event Graphs**: Models each pass as a graph with players as nodes.
4. **Learns Player Embeddings**: Trains a GCN to predict delta xT, generating 128-dimensional player embeddings.
5. **Predicts Compatibility**: Uses a multi-task MLP to score player pairs.

## Data Requirements

Place StatsBomb CSV files in a `csv_data/` folder:

```
csv_data/
├── events_batch_0.csv, events_batch_1.csv, ...  # Event data (required)
├── lineups.csv                                   # Player ID → Name mapping (required)
├── matches.csv                                   # Match metadata (optional)
└── competitions.csv                              # Competition info (optional)
```

**Minimum Required Columns (events_batch_*.csv):**
- `match_id`, `team_id`, `player_id`, `event_type`
- `x_start`, `y_start`, `x_end`, `y_end`
- `pass_recipient_id`, `timestamp`
- `pass_angle`, `pass_length`, `pass_height_name`, `under_pressure`

**Lineups File:**
- `player_id` (Integer)
- `player_name` (String)

## Model Architecture

### 11-Stage Pipeline

**Stages 1-3**: Load events, assign possession chains, sort chronologically.

**Stage 4**: Compute Expected Threat (xT) map on a 12x8 grid using Markov Chain analysis. Calculate delta xT (threat change) for every action.

**Stage 5**: Build 23-dimensional player feature vectors:
- Core metrics (9): Passes, accuracy, dribbles, tackles, xT created/received
- Position encoding (7): One-hot encoded tactical role
- Pass signature (5): Average length, angle, type distribution, aerial percentage
- Pressure profile (2): Performance under pressure

**Stage 6**: Construct pass event graphs where nodes are on-pitch players and edges encode spatial/threat information.

**Stage 7**: Train a 3-layer GCN to predict delta xT, learning 128-dimensional player embeddings.

**Stage 8**: Extract player embeddings by averaging across all graphs.

**Stage 9**: Compute zone profiles and heatmaps (secondary signals).

**Stage 10**: Train a multi-task MLP scorer that takes concatenated embeddings [h_A || h_B] and outputs:
- Main compatibility score (0-1)
- Pass quality prediction
- Threat flow prediction
- Position synergy prediction

**Stage 11**: Score target player pair and output final result.

## Installation and Setup Instructions

Before running the pipeline, you need to set up your Python environment with the required dependencies. This section provides step-by-step instructions for different scenarios.

### System Requirements

The code requires Python 3.8 or later. Ensure you have Python installed on your system by running:

```bash
python --version
```

If you need to install Python, visit https://www.python.org/downloads/ and follow the installation instructions for your operating system.

### Required Dependencies

The core dependencies required for running the pipeline are:

```
torch              # PyTorch - deep learning framework
pandas             # Data manipulation and analysis
numpy              # Numerical computing
tqdm               # Progress bar utilities
scikit-learn       # Machine learning utilities
```

### Installation Step 1: Create a Virtual Environment (Recommended)

It is strongly recommended to create an isolated Python environment for this project to avoid dependency conflicts with other projects. Using a virtual environment prevents version conflicts and makes the project more reproducible.

On Linux or macOS:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Installation Step 2: Install Core Dependencies

Install the core required packages using pip:

```bash
pip install --upgrade pip
pip install torch pandas numpy tqdm scikit-learn
```

This command will install:
- torch: The PyTorch deep learning framework
- pandas: For loading and manipulating CSV data
- numpy: For numerical operations
- tqdm: For progress bar display during training
- scikit-learn: For machine learning utilities like normalization

The installation may take several minutes depending on your internet speed and whether PyTorch needs to compile components for your system.

### Installation Step 3: Install Optional but Recommended Dependency

For significantly faster graph neural network training, PyTorch Geometric is highly recommended. This library provides optimized GPU-accelerated graph operations:

```bash
pip install torch-geometric
```

Note: If the automatic installation of torch-geometric fails, refer to the official installation guide at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

If torch-geometric is not installed, the pipeline will automatically fall back to a slower CPU-based implementation using standard PyTorch operations. The results will be identical, but training will take longer.

### Installation Step 4: Verify Installation

Verify that all dependencies are correctly installed by running:

```bash
python -c "import torch; import pandas; import numpy; print('All core dependencies installed successfully')"
```

If you installed torch-geometric, also verify:

```bash
python -c "import torch_geometric; print('PyTorch Geometric installed successfully')"
```

### Data Preparation

Before running the pipeline, ensure your data is properly organized:

1. Create a directory named `csv_data` in the project root directory (if it doesn't already exist)
2. Place all StatsBomb CSV files in this directory:
   - `events_batch_0.csv`, `events_batch_1.csv`, etc. (all event batches)
   - `lineups.csv` (player name mappings)
   - `matches.csv` and `competitions.csv` (optional, for reference)

The directory structure should look like:

```
project_root/
    compatibility_model_gnn.py
    csv_data/
        events_batch_0.csv
        events_batch_1.csv
        events_batch_2.csv
        ... (more batch files)
        lineups.csv
        matches.csv
        competitions.csv
    venv/  (or your virtual environment folder)
```

## Running the Pipeline: Complete Usage Guide

Once the installation is complete, you can run the compatibility model from the command line. This section provides comprehensive instructions and examples.

### Basic Execution

The simplest way to run the model is to specify two player IDs:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320
```

In this example, player 5503 (Lionel Messi) is being compared to player 4320 (Neymar Jr). The pipeline will run through all 11 stages and output a compatibility score.

### Finding Player IDs

If you don't know the player IDs, you can search the lineups.csv file. For example, to find Messi's ID on Linux or macOS:

```bash
grep -i "messi" csv_data/lineups.csv
```

On Windows:

```bash
findstr /I "messi" csv_data\lineups.csv
```

This will display the line containing Messi's information, including his ID. Similarly, search for any player by name using grep or findstr.

### Configured Execution with Options

The pipeline supports several optional parameters to customize its behavior:

```bash
python compatibility_model_gnn.py \
  --playerA 5503 \
  --playerB 4320 \
  --epochs 10 \
  --batch-size 64 \
  --csv-dir ./csv_data \
  --grid-x 12 \
  --grid-y 8 \
  --min-events-player 30
```

### Command Line Arguments Reference

**Required Arguments:**

- `--playerA INTEGER` (required): The ID of the first player to analyze
- `--playerB INTEGER` (required): The ID of the second player to analyze

**Optional Arguments:**

- `--csv-dir PATH` (default: ./csv_data): Directory containing the event CSV files. Use this if your CSV files are in a different location than the default csv_data folder.

- `--epochs INTEGER` (default: 3): Number of training epochs for the Graph Neural Network. More epochs allow the network to learn better but take longer. Recommended range: 3-10. Use higher values if you have a large dataset or want more refined embeddings.

- `--batch-size INTEGER` (default: 32): Number of graphs to process simultaneously during training. Larger batch sizes can speed up training but require more GPU memory. If you encounter out-of-memory errors, reduce this value. Recommended range: 16-128.

- `--grid-x INTEGER` (default: 12): Number of columns in the Expected Threat grid. This divides the pitch horizontally. Standard is 12 (creating 20-yard sections on a 120-yard pitch).

- `--grid-y INTEGER` (default: 8): Number of rows in the Expected Threat grid. This divides the pitch vertically. Standard is 8 (creating 10-yard sections on an 80-yard pitch).

- `--min-events-player INTEGER` (default: 0): Minimum number of events a player must have to be included in analysis. Use this to filter out sparse players who appeared in very few events. For example, set to 30 to only include players with at least 30 recorded actions.

- `--no-train` (flag, no argument): Skip GNN training entirely. The system will use raw feature vectors as embeddings instead of learning them. This is much faster but less accurate. Useful for quick tests.

- `--sample-pass-graphs INTEGER`: Limit the number of pass events (graphs) used for training. Useful for quick tests on large datasets. For example, `--sample-pass-graphs 500` uses only 500 randomly selected pass events.

### Example Scenarios

**Scenario 1: Quick Test Run (Fast Execution)**

For testing purposes with minimal training:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320 --epochs 2 --batch-size 16
```

This completes in roughly 1-2 minutes and gives a quick compatibility estimate. Useful for verifying the pipeline works before running full analyses.

**Scenario 2: Accurate Analysis (Recommended)**

For accurate results with reasonable runtime:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320 --epochs 5 --batch-size 32
```

This typically completes in 3-5 minutes and provides reliable compatibility scores. This is the recommended setting for most use cases.

**Scenario 3: High-Quality Analysis (Thorough)**

For the most accurate embeddings with extended training:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320 --epochs 10 --batch-size 64
```

This may take 10-20 minutes but produces the highest quality embeddings. Use when computational resources are not a constraint.

**Scenario 4: Sparse Player Filtering**

If your dataset contains many players with very few events (substitutes, youth players, etc.), filter them:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320 --min-events-player 50
```

This removes any player with fewer than 50 events, improving data quality by focusing on players with substantial playing time.

**Scenario 5: High-Resolution Expected Threat**

For more granular threat calculations using a finer grid:

```bash
python compatibility_model_gnn.py --playerA 5503 --playerB 4320 --grid-x 16 --grid-y 12
```

This creates a 16x12 grid (192 zones instead of 96), allowing finer spatial resolution at the cost of slightly longer computation time.

### Understanding the Output

When the pipeline completes, it prints the final result in this format:

```
================================================================================
FINAL RESULT
================================================================================
Compatibility Score (5503 Lionel Andrés Messi Cuccittini → 4320 Neymar da Silva Santos Junior): 0.8199
================================================================================
```

The output shows:
- Player A's ID and full name
- Player B's ID and full name
- A decimal score between 0.0 and 1.0

### Interpreting Compatibility Scores

The compatibility score should be interpreted as follows:

- **0.80 to 1.00**: Elite partnership potential. The players have highly complementary embeddings and playing styles. This is typical of famous duos who have played together successfully.

- **0.60 to 0.79**: Strong compatibility. The players would likely work well together and understand each other's positioning.

- **0.40 to 0.59**: Moderate compatibility. The players have some compatible qualities but may not be ideal partners.

- **0.20 to 0.39**: Poor compatibility. Significant stylistic differences suggest they would struggle to coordinate effectively.

- **0.00 to 0.19**: Very poor compatibility. The players have fundamentally incompatible playing styles and zones of influence.

### Troubleshooting Execution

**Problem: "Player not found in embeddings"**
Solution: Verify the player ID exists in your lineups.csv file. Check that the player appeared in at least one event in the event data.

**Problem: "Out of memory" error**
Solution: Reduce the batch size using `--batch-size 16` or `--batch-size 8`. Also reduce the number of epochs.

**Problem: Very slow execution without PyTorch Geometric**
This is expected. If installed, the pipeline uses the faster PyTorch Geometric library. If not installed and you have GPU available, installation will provide a significant speedup.

**Problem: Script crashes with "No pass graphs found"**
Solution: Ensure your event data contains Pass events. Check that your CSV files are in the correct format and location.

## Technical Methodology

### Why Graph-Based Representation?

Football is inherently relational. By representing match moments as graphs with players as nodes and interactions as edges, the model captures spatial context, implicit pass patterns, style matching, and temporal continuity that pure statistics miss.

### The Markov Chain xT Model

Expected Threat is calculated iteratively using zone transition counts and shot statistics, solving for the fixed point where xT values reflect both immediate shooting threats and future pass opportunities.

### Embedding Concatenation

The model uses concatenation [h_A || h_B] to preserve all information from both player embeddings. This allows the downstream MLP to discover relevant interaction patterns, unlike dot products (lose magnitude) or cosine similarity (lose scale).

### Multi-Task Learning

The scorer trains on multiple related tasks (main compatibility, pass quality, threat flow, position synergy), which regularizes learning and improves generalization.

## Citation and Acknowledgments

This project uses event data provided by StatsBomb. The GNN architecture is based on Graph Convolutional Networks (Kipf & Welling, 2017).

## License

This project is open-source and available for educational and research purposes. The code is provided without warranty. The underlying match data is provided by StatsBomb under their Open Data license. Please refer to StatsBomb's terms when using the data.

For questions, issues, or contributions, please open an issue on the GitHub repository.

