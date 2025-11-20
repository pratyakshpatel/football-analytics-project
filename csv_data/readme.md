# StatsBomb Event Data - CSV Format

This directory contains converted StatsBomb soccer event data in CSV format. The data has been processed from the original JSON format and split into multiple files for manageability.

## File Overview

### Core Data Files

#### `events_batch_0.csv` through `events_batch_4.csv`
Main event data files containing detailed play-by-play information for all matches. The events are split across 5 batch files to keep individual files at reasonable sizes.

**Key columns:**
- `match_id` - Unique identifier for the match
- `id` - Unique event identifier
- `index` - Event sequence number within the match
- `period` - Match period (1-5, where 5 is extra time)
- `timestamp` - Event time in HH:MM:SS.mmm format
- `minute`, `second` - Broken down time components
- `type_id`, `type_name` - Event classification (Pass, Shot, Tackle, Carry, Dribble, Foul, etc.)
- `player_id`, `player_name` - Who performed the action
- `team_id`, `team_name` - Player's team
- `position_id`, `position_name` - Player's position on field
- `possession` - Possession sequence number
- `location_x`, `location_y` - Event start location (0-120 on x-axis, 0-80 on y-axis)

**Pass-specific columns:**
- `pass_recipient_id` - Target player for the pass
- `pass_end_location_x`, `pass_end_location_y` - Where the pass ended
- `pass_type_name` - Type of pass (Short, Long, Through Ball, etc.)
- `pass_length`, `pass_angle` - Distance and angle of pass
- `pass_height_id`, `pass_height_name` - Height of pass (Ground, Low, High)
- `pass_body_part_id`, `pass_body_part_name` - Body part used (Foot, Head, etc.)
- `pass_outcome` - Success indicator

**Shot-specific columns:**
- `shot_end_location_x`, `shot_end_location_y` - Shot endpoint/goal location
- `shot_type_name` - Shot classification
- `shot_outcome` - Result (Goal, Saved, Wayward, etc.)

**Context columns:**
- `under_pressure` - Boolean indicating if player was under defensive pressure
- `possession_team` - Team with possession
- `play_pattern` - Context of play (Open Play, From Free Kick, From Corner, etc.)

#### `lineups.csv`
Player roster information for all matches.

**Key columns:**
- `match_id` - Match identifier
- `team_id`, `team_name` - Team information
- `player_id` - Player identifier
- `player_name` - Full player name
- `player_nickname` - Common nickname
- `jersey_number` - Squad number
- `country_id`, `country_name` - Player nationality
- `position_id`, `position_name` - Playing position (Goalkeeper, Center Back, Full Back, Center Midfield, etc.)
- `height_cm` - Player height
- `weight_kg` - Player weight
- `positions_played` - List of positions played in squad (may differ from starting position)

#### `matches.csv`
Match metadata and results.

**Key columns:**
- `match_id` - Match identifier
- `competition_id`, `competition_name` - Competition (e.g., La Liga)
- `season_id` - Season year
- `team_id`, `team_name` - Teams involved (home and away columns)
- `match_date` - Date of match
- `kick_off` - Match start time
- `status` - Match status (Available, Scheduled, etc.)
- `home_score`, `away_score` - Final score
- `duration` - Match duration in minutes
- `referee` - Match referee

#### `competitions.csv`
Competition metadata.

**Key columns:**
- `competition_id` - Unique competition identifier
- `competition_name` - Name of competition (La Liga, etc.)
- `country_name` - Country where competition takes place
- `competition_gender` - M for men's, W for women's

#### `conversion_summary.csv`
Metadata about the data conversion process.

**Typical columns:**
- Batch number
- File conversion status
- Number of events processed
- Date of conversion

#### `competitions.csv`
Reference data for all competitions in the dataset.

## Data Statistics

**Events dataset:**
- Total events: 173,623+
- Batch files: 5
- Time period: Multiple seasons across European football

**Coverage:**
- Multiple clubs from top European leagues
- Detailed event tracking including passes, shots, tackles, interceptions, and more
- Player positioning and movement data
- Pressure context for decision-making analysis

## Data Dictionary - Event Types

Common event types found in `type_name`:
- `Pass` - Ball passed between players
- `Carry` - Player carries ball (dribble)
- `Dribble` - Player attempts dribble with ball at feet
- `Shot` - Attempt to score
- `Tackle` - Defensive action
- `Interception` - Gaining possession through interception
- `Foul` - Rule violation
- `Ball Recovery` - Gaining possession after ball loose
- `Clearance` - Defensive clearing action
- `Throw In` - Throw-in restart
- `Goalkeeper` - Goalkeeper action
- `Half Start` - Period start marker
- `Half End` - Period end marker

## Data Quality Notes

- **Pitch dimensions:** Normalized to 120 × 80 (some sources may vary)
- **Player IDs:** Consistent across matches and seasons
- **Team IDs:** Unique per organization
- **Missing data:** Some optional event attributes may be null (e.g., pass_outcome for unsuccessful passes)
- **Pressure indicator:** Boolean flag; True indicates player acting under opponent pressure

## Usage Examples

### Load all events
```python
import pandas as pd
import glob

dfs = [pd.read_csv(f) for f in sorted(glob.glob('events_batch_*.csv'))]
events = pd.concat(dfs, ignore_index=True)
```

### Filter by team
```python
real_madrid = events[events['team_name'] == 'Real Madrid']
```

### Analyze passes
```python
passes = events[events['type_name'] == 'Pass']
pass_accuracy = (passes['pass_recipient_id'].notna().sum() / len(passes)) * 100
```

### Get player statistics
```python
player_data = events[events['player_id'] == 5207]  # Specific player
player_passes = len(player_data[player_data['type_name'] == 'Pass'])
```

### Pressure analysis
```python
pressured_passes = events[(events['type_name'] == 'Pass') & (events['under_pressure'] == True)]
pressure_pass_acc = pressured_passes['pass_recipient_id'].notna().sum() / len(pressured_passes)
```

## Conversion Notes

- Data converted from StatsBomb's native JSON format
- Location coordinates normalized to standard pitch dimensions
- All timestamps preserved in original format
- Missing values handled gracefully (NaN for numeric, null for categorical)



## Recommended Reading Order

1. Start with `competitions.csv` to understand tournament structure
2. Review `matches.csv` for match-level context
3. Consult `lineups.csv` for player information
4. Analyze `events_batch_*.csv` for detailed play-by-play data

## Notes for Analysis

- Combine events with lineups using `player_id` for enriched analysis
- Match events to matches using `match_id` for context
- Use `possession` field to group related events into sequences
- Filter by `period` to analyze specific match phases
- Leverage `under_pressure` and `play_pattern` for contextual understanding

