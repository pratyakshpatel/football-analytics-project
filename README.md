# Graph-Based Player Compatibility Model (GoalNet-Lite)

A sophisticated Machine Learning pipeline that utilizes Graph Neural Networks (GNNs) to quantify the chemical compatibility between football players. Unlike traditional approaches that rely on weighted averages of statistics, this project learns player embeddings directly from match event data and predicts compatibility using a multi-task neural network trained on actual on-pitch interactions.

## Overview

This repository contains a complete end-to-end pipeline implemented in `compatibility_model copy.py`. The system performs the following sequence of operations:

1. **Ingests StatsBomb Event Data**: Processes raw match events from CSV batches into a structured dataframe containing all relevant match action data.

2. **Calculates Advanced Threat Metrics**: Computes a grid-based Expected Threat (xT) model using a Markov Chain approach. This assigns threat values to different zones of the pitch based on the probability of generating a goal from that zone. Additionally, for every action, it calculates the delta xT (change in threat), which measures how much the threat increased or decreased from the start to the end of that action.

3. **Constructs Event Graphs**: Models every pass event as a graph where nodes represent all players currently on the pitch and edges encode spatial relationships, pass characteristics, and threat changes. This graph representation captures the contextual information surrounding each pass.

4. **Learns Player Embeddings**: Trains a Graph Convolutional Network (GCN) to predict the delta xT (change in threat) for each pass. During this training process, the network learns a dense 128-dimensional embedding for each player that captures their playing style, spatial preferences, and threat-generating capabilities.

5. **Predicts Compatibility**: Uses a learned neural network scorer to evaluate how well two players' embeddings complement each other. Instead of using hardcoded weights, the system learns from data which embedding patterns correlate with successful partnerships.

## Data Structure and Requirements

The model expects data to be organized in a specific directory structure. All input CSV files should be placed in a folder named `csv_data/` in the project root directory. This section describes the required data files and their attributes.

### Directory Layout and File Organization

The following directory structure is required for the pipeline to function correctly:

```
project_root/
csv_data/
    events_batch_0.csv    # Event data chunk 0 (main event information)
    events_batch_1.csv    # Event data chunk 1
    events_batch_2.csv    # Event data chunk 2
    ...more batches...
    lineups.csv           # Player roster and ID to name mapping
    matches.csv           # Match metadata and fixtures
    competitions.csv      # Competition and league information
```

The events are split into batches to manage file size. The pipeline automatically discovers and loads all `events_batch_*.csv` files in the csv_data directory in numerical order.

### Event Data Attributes (events_batch_*.csv)

Each event batch CSV file should contain the following columns. These represent the standard StatsBomb format converted into a flat CSV structure:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `match_id` | Integer | Unique identifier for each match in the dataset |
| `team_id` | Integer | Identifier for the team performing the action |
| `player_id` | Integer | Identifier for the player performing the action |
| `event_type` | String | Classification of the event (Pass, Shot, Carry, Dribble, Tackle, Interception, etc.) |
| `x_start` | Float | Starting X-coordinate of the action (normalized to 0-1 scale) |
| `y_start` | Float | Starting Y-coordinate of the action (normalized to 0-1 scale) |
| `x_end` | Float | Ending X-coordinate of the action (for passes and shots) |
| `y_end` | Float | Ending Y-coordinate of the action (for passes and shots) |
| `pass_recipient_id` | Integer | Player ID of the pass recipient (only for Pass events) |
| `timestamp` | String/Datetime | Time at which the event occurred (used for chronological ordering) |
| `pass_angle` | Float | Angle of the pass trajectory in radians |
| `pass_length` | Float | Distance covered by the pass in pitch units |
| `pass_height_name` | String | Pass height classification (Ground Pass, High Pass, Head Pass) |
| `pass_type_name` | String | Pass type classification (Short Pass, Long Pass, Cross, etc.) |
| `under_pressure` | Boolean | Flag indicating whether the player was under defensive pressure |
| `position_id` | Integer | Tactical position of the player (1=GK, 2=Defense, 3=Midfield, 4=Forward) |
| `shot_outcome` | String | Outcome of shots (Saved, Goal, Off Target, Blocked, Post) |
| `outcome` | String | General outcome (Success, Failure, Incomplete) |

Note: The pipeline includes graceful fallback handling for missing columns. If certain columns are not present, the system will either use default values or skip that particular feature extraction step with a warning message.

### Lineups File (lineups.csv)

The lineups file is essential for mapping player IDs to human-readable names in the output. It should contain at minimum the following columns:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `player_id` | Integer | Unique identifier for the player |
| `player_name` | String | Full name of the player (e.g., "Lionel Andrés Messi Cuccittini") |

The pipeline uses this mapping to display human-readable output when computing compatibility scores.

## Model Architecture and Technical Details

The complete pipeline is implemented as a single Python script (`compatibility_model copy.py`) and consists of 11 sequential processing stages. Each stage transforms the data and extracts features that feed into the next stage.

### Stage 1-3: Data Loading and Possession Assignment

The pipeline begins by loading all StatsBomb event data from the CSV files in the csv_data directory. The events are then sorted chronologically within each match. To prepare for graph construction, possession chains are assigned. A new possession ID is created each time the ball changes from one team to another, creating chains of actions by the same team. This stage is purely data preparation and does not involve any machine learning.

### Stage 4: Expected Threat (xT) Calculation

Expected Threat is a fundamental metric in modern football analytics. The pitch is divided into a grid (default 12 columns by 8 rows, creating 96 zones). For each zone, the model calculates the probability of scoring from that zone using a Markov Chain approach:

The algorithm first counts all transitions between zones (where the ball moves from one zone to another) and tracks shots and goals from each zone. Using these statistics, it calculates:
- P(shot): Probability of taking a shot from this zone
- P(goal | shot): Probability of scoring given a shot from this zone
- Transition probabilities: Where the ball typically moves next

These are combined iteratively to compute xT for each zone. For every pass event, the system then calculates delta xT: the change in threat value from the start point to the end point. A pass that moves the ball to a higher-threat zone has positive delta xT, indicating a good pass that increased scoring probability.

### Stage 5: Enriched Player Feature Vectors (23 Dimensions)

Rather than using raw statistics, the system constructs feature vectors that capture different aspects of each player's playing style. These 23-dimensional vectors are composed of four components:

**Core Metrics (9 dimensions)**: Counting statistics including number of passes completed, pass completion percentage, number of dribbles, tackles won, interceptions, clearances, key passes, xT created (total threat generated), and xT received (total threat created for this player by teammates).

**Position Encoding (7 dimensions)**: One-hot encoding of the player's most common tactical position. This creates a 7-dimensional binary vector where exactly one dimension is 1 and the rest are 0, representing goalkeeper, defender, midfielder, forward, or other positions.

**Pass Signature (5 dimensions)**: Characteristics of the player's passing style including average pass length, average pass angle, percentage of short passes, percentage of long passes, and percentage of aerial passes.

**Pressure Profile (2 dimensions)**: Performance metrics under pressure including the percentage of actions where the player was under defensive pressure and pass completion percentage while under pressure.

These features are normalized across all players to ensure comparable scales, preventing any single feature from dominating due to unit differences.

### Stage 6: Graph Construction from Pass Events

For every pass event in the dataset, the system constructs a graph representation. In this graph:

- **Nodes**: All players currently on the pitch (22 players total)
- **Node Features**: The 23-dimensional player feature vectors described above
- **Edges**: Connections between the passer and all other on-pitch players
- **Edge Attributes**: 10 features including starting coordinates (x, y), ending coordinates (x, y), delta xT (threat change), pass length, pass angle, pass height, body part used, and pass outcome

Building graphs for all pass events creates a large dataset on which the GNN can be trained. On average, each graph contains 22 nodes and dozens of edges, capturing the rich relational structure of a moment in the match.

### Stage 7: Graph Neural Network Training

A Graph Convolutional Network (GCN) with 3 layers is trained on these graphs. The objective is to predict the delta xT value for each edge (pass). By training to predict how much threat each pass creates, the network learns to extract player embeddings that capture threat-generation capabilities.

During the forward pass, node features are updated using graph convolutions, aggregating information from neighboring nodes. After several layers of message passing, each node has a learned representation (embedding) that encodes both its own characteristics and its relationship to teammates.

The network is trained with standard backpropagation for typically 3-10 epochs. The training process minimizes the Mean Squared Error between predicted and actual delta xT values.

### Stage 8: Player Embedding Extraction

Once the GNN is trained, embeddings are extracted for each player. For players appearing in multiple graphs, embeddings are averaged across all appearances. This results in a single 128-dimensional embedding per player that summarizes their threat-generation patterns and playing style as learned by the neural network.

These embeddings are the core representation used in the compatibility scoring stage. Unlike raw features which are predefined, these embeddings are learned from data and can capture complex, nonlinear relationships that the GNN discovered.

### Stage 9: Zone Profiles and Heatmaps

For each player, the system computes several spatial profiles. These include heatmaps showing where on the pitch the player typically acts, pass destination distributions showing where they typically pass to, and xT profiles showing which zones they tend to create the most threat from. These spatial profiles are used for additional compatibility calculations but are secondary to the learned embeddings.

### Stage 10: Learned Compatibility Scorer

Rather than using hardcoded weights, the system uses a multi-task neural network to predict compatibility. This network takes as input the concatenated embeddings of two players ([h_A || h_B], a 256-dimensional vector) and outputs multiple predictions:

**Main Compatibility Head**: The primary output, a score between 0 and 1 indicating how compatible the two players are based on their learned embeddings.

**Auxiliary Head 1 - Pass Quality**: A prediction of how successful passes between the two players would be.

**Auxiliary Head 2 - Threat Potential**: A prediction of how much collective threat the pair generates.

**Auxiliary Head 3 - Synergy**: A spatial overlap measure indicating whether the players occupy complementary zones.

This multi-task learning setup allows the network to learn a richer representation by optimizing multiple related objectives simultaneously.

### Stage 11: Final Score Computation

The final compatibility score for two players is computed by passing their learned embeddings through the Learned Compatibility Scorer network. The result is a single floating-point value between 0 and 1, where higher values indicate greater compatibility.

The output is interpretable: a score above 0.8 suggests elite-level partnership potential, scores between 0.5-0.7 suggest moderate compatibility, and scores below 0.3 suggest poor compatibility.

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
    compatibility_model copy.py
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
python "compatibility_model copy.py" --playerA 5503 --playerB 4320
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
python "compatibility_model copy.py" \
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
python "compatibility_model copy.py" --playerA 5503 --playerB 4320 --epochs 2 --batch-size 16
```

This completes in roughly 1-2 minutes and gives a quick compatibility estimate. Useful for verifying the pipeline works before running full analyses.

**Scenario 2: Accurate Analysis (Recommended)**

For accurate results with reasonable runtime:

```bash
python "compatibility_model copy.py" --playerA 5503 --playerB 4320 --epochs 5 --batch-size 32
```

This typically completes in 3-5 minutes and provides reliable compatibility scores. This is the recommended setting for most use cases.

**Scenario 3: High-Quality Analysis (Thorough)**

For the most accurate embeddings with extended training:

```bash
python "compatibility_model copy.py" --playerA 5503 --playerB 4320 --epochs 10 --batch-size 64
```

This may take 10-20 minutes but produces the highest quality embeddings. Use when computational resources are not a constraint.

**Scenario 4: Sparse Player Filtering**

If your dataset contains many players with very few events (substitutes, youth players, etc.), filter them:

```bash
python "compatibility_model copy.py" --playerA 5503 --playerB 4320 --min-events-player 50
```

This removes any player with fewer than 50 events, improving data quality by focusing on players with substantial playing time.

**Scenario 5: High-Resolution Expected Threat**

For more granular threat calculations using a finer grid:

```bash
python "compatibility_model copy.py" --playerA 5503 --playerB 4320 --grid-x 16 --grid-y 12
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

## Technical Methodology and Design Decisions

This section explains the core concepts and design choices that make this approach effective for player compatibility scoring.

### Why Graph-Based Representation?

Football inherently involves relational dynamics. A player's value is not purely individual statistics, but rather how they influence space, teammate movement, and defensive pressure. Traditional statistical approaches miss these network effects. By representing each match moment as a graph where players are nodes and interactions are edges, the model captures:

**Spatial Context**: Where players typically receive the ball and what they do with it
**Implicit Connections**: Patterns of who passes to whom and when
**Style Matching**: Whether a long-ball passer has appropriate target players who excel at receiving long passes
**Temporal Continuity**: How actions build on each other through possession chains

### The Markov Chain xT Model

Expected Threat values are calculated using a Markov Chain approach rather than empirical look-ups. This allows the model to work with any dataset and handles zone transitions mathematically. The model runs iteratively until convergence, solving for the fixed point where xT values reflect both immediate shooting threats and future pass opportunities. This is more theoretically sound than simple averaging.

### Embedding Concatenation vs. Other Fusion Methods

When combining two player embeddings, the system uses concatenation [h_A || h_B] rather than other approaches like dot products or cosine similarity. Here's why:

Dot products measure angle similarity but lose magnitude information. Cosine similarity normalizes to unit vectors, eliminating information about player intensity or influence scale. Concatenation preserves all information from both embeddings and allows the downstream neural network to discover any relevant interaction patterns. The MLP can learn which combinations of features are complementary.

### Multi-Task Learning Benefits

The compatibility scorer uses auxiliary tasks (pass quality, threat potential, synergy) beyond the main compatibility output. This approach:

- Regularizes the main task, preventing overfitting
- Provides interpretable intermediate outputs for debugging
- Allows the shared trunk to learn richer representations by solving multiple related problems
- Improves generalization by sharing feature extraction across tasks

### Handling Sparse Players

Some players (substitutes, youth squad, long-term injured) have very few events. The `--min-events-player` option filters these out before training. Sparse players produce noisy features and unreliable embeddings. Filtering improves embedding quality for regular starters.

## Understanding Graph Construction Details

Each pass event generates a graph containing valuable contextual information. The graph includes:

- All 22 on-pitch players as nodes
- Edges connecting the passer to all other on-pitch players
- Node features encoding each player's style and position
- Edge features encoding spatial relationships and threat information

This creates a large training set where even a 50-match sample generates thousands of graphs. The GNN learns player representations by observing patterns across all these match moments.

## Citation and Acknowledgments

This project uses event data provided by StatsBomb, a leading football analytics company. StatsBomb's open data initiative makes detailed match analysis accessible to the broader football community.

The graph neural network architecture is based on Graph Convolutional Networks (Kipf & Welling, 2017), implemented using PyTorch and PyTorch Geometric.

## License

This project is open-source and available for educational and research purposes. The code is provided without warranty. The underlying match data is provided by StatsBomb under their Open Data license. Please refer to StatsBomb's terms when using the data.

For questions, issues, or contributions, please open an issue on the GitHub repository.
