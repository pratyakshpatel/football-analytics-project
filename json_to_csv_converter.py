import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime

class StatsBombCSVConverter:
    def __init__(self, data_dir='./open-data/data', output_dir='./csv_data'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_competitions(self):
        """Convert competitions.json to CSV"""
        print("Converting competitions data...")
        competitions_file = self.data_dir / 'competitions.json'
        
        with open(competitions_file, 'r') as f:
            competitions = json.load(f)
        
        df = pd.DataFrame(competitions)
        output_file = self.output_dir / 'competitions.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        return df
    
    def convert_matches(self):
        """Convert all matches JSON files to CSV"""
        print("Converting matches data...")
        matches_dir = self.data_dir / 'matches'
        all_matches = []
        
        for comp_dir in matches_dir.iterdir():
            if comp_dir.is_dir():
                for season_file in comp_dir.glob('*.json'):
                    with open(season_file, 'r') as f:
                        matches = json.load(f)
                    
                    for match in matches:
                        # Flatten nested structures
                        flattened_match = self._flatten_match_data(match)
                        all_matches.append(flattened_match)
        
        df = pd.DataFrame(all_matches)
        output_file = self.output_dir / 'matches.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file} with {len(df)} matches")
        return df
    
    def _flatten_match_data(self, match):
        """Flatten nested match data structure"""
        flattened = {
            'match_id': match.get('match_id'),
            'match_date': match.get('match_date'),
            'kick_off': match.get('kick_off'),
            'competition_id': match.get('competition', {}).get('competition_id'),
            'competition_name': match.get('competition', {}).get('competition_name'),
            'season_id': match.get('season', {}).get('season_id'),
            'season_name': match.get('season', {}).get('season_name'),
            'home_team_id': match.get('home_team', {}).get('home_team_id'),
            'home_team_name': match.get('home_team', {}).get('home_team_name'),
            'away_team_id': match.get('away_team', {}).get('away_team_id'),
            'away_team_name': match.get('away_team', {}).get('away_team_name'),
            'home_score': match.get('home_score'),
            'away_score': match.get('away_score'),
            'match_status': match.get('match_status'),
            'match_week': match.get('match_week'),
            'stadium_id': match.get('stadium', {}).get('id') if match.get('stadium') else None,
            'stadium_name': match.get('stadium', {}).get('name') if match.get('stadium') else None,
            'referee_id': match.get('referee', {}).get('id') if match.get('referee') else None,
            'referee_name': match.get('referee', {}).get('name') if match.get('referee') else None,
        }
        return flattened
    
    def convert_lineups(self):
        """Convert all lineup JSON files to CSV"""
        print("Converting lineups data...")
        lineups_dir = self.data_dir / 'lineups'
        all_lineups = []
        
        for lineup_file in lineups_dir.glob('*.json'):
            match_id = lineup_file.stem
            
            with open(lineup_file, 'r') as f:
                lineups = json.load(f)
            
            for team_lineup in lineups:
                team_id = team_lineup['team_id']
                team_name = team_lineup['team_name']
                
                for player in team_lineup['lineup']:
                    flattened_player = self._flatten_lineup_data(player, match_id, team_id, team_name)
                    all_lineups.append(flattened_player)
        
        df = pd.DataFrame(all_lineups)
        output_file = self.output_dir / 'lineups.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file} with {len(df)} player-match records")
        return df
    
    def _flatten_lineup_data(self, player, match_id, team_id, team_name):
        """Flatten nested player lineup data"""
        flattened = {
            'match_id': match_id,
            'team_id': team_id,
            'team_name': team_name,
            'player_id': player.get('player_id'),
            'player_name': player.get('player_name'),
            'player_nickname': player.get('player_nickname'),
            'jersey_number': player.get('jersey_number'),
            'country_id': player.get('country', {}).get('id') if player.get('country') else None,
            'country_name': player.get('country', {}).get('name') if player.get('country') else None,
        }
        
        # Add position information
        for i, position in enumerate(player.get('positions', [])):
            flattened[f'position_{i+1}_id'] = position.get('position_id')
            flattened[f'position_{i+1}_name'] = position.get('position')
            flattened[f'position_{i+1}_from'] = position.get('from')
            flattened[f'position_{i+1}_to'] = position.get('to')
            flattened[f'position_{i+1}_from_period'] = position.get('from_period')
            flattened[f'position_{i+1}_to_period'] = position.get('to_period')
        
        return flattened
    
    def convert_events(self, max_files=None):
        """Convert event JSON files to CSV (can be memory intensive)"""
        print("Converting events data...")
        events_dir = self.data_dir / 'events'
        
        # Get list of event files
        event_files = list(events_dir.glob('*.json'))
        if max_files:
            event_files = event_files[:max_files]
            print(f"Processing first {max_files} event files...")
        
        # Process in batches to manage memory
        batch_size = 10
        batch_num = 0
        
        for i in range(0, len(event_files), batch_size):
            batch_files = event_files[i:i+batch_size]
            all_events = []
            
            print(f"Processing batch {batch_num + 1} ({len(batch_files)} files)...")
            
            for event_file in batch_files:
                match_id = event_file.stem
                
                with open(event_file, 'r') as f:
                    events = json.load(f)
                
                for event in events:
                    flattened_event = self._flatten_event_data(event, match_id)
                    all_events.append(flattened_event)
            
            # Save batch
            if all_events:
                df = pd.DataFrame(all_events)
                output_file = self.output_dir / f'events_batch_{batch_num}.csv'
                df.to_csv(output_file, index=False)
                print(f"Saved: {output_file} with {len(df)} events")
            
            batch_num += 1
        
        print("Events conversion completed in batches")
    
    def _flatten_event_data(self, event, match_id):
        """Flatten nested event data structure"""
        flattened = {
            'match_id': match_id,
            'id': event.get('id'),
            'index': event.get('index'),
            'period': event.get('period'),
            'timestamp': event.get('timestamp'),
            'minute': event.get('minute'),
            'second': event.get('second'),
            'type_id': event.get('type', {}).get('id') if event.get('type') else None,
            'type_name': event.get('type', {}).get('name') if event.get('type') else None,
            'possession': event.get('possession'),
            'possession_team_id': event.get('possession_team', {}).get('id') if event.get('possession_team') else None,
            'possession_team_name': event.get('possession_team', {}).get('name') if event.get('possession_team') else None,
            'play_pattern_id': event.get('play_pattern', {}).get('id') if event.get('play_pattern') else None,
            'play_pattern_name': event.get('play_pattern', {}).get('name') if event.get('play_pattern') else None,
            'team_id': event.get('team', {}).get('id') if event.get('team') else None,
            'team_name': event.get('team', {}).get('name') if event.get('team') else None,
            'player_id': event.get('player', {}).get('id') if event.get('player') else None,
            'player_name': event.get('player', {}).get('name') if event.get('player') else None,
            'position_id': event.get('position', {}).get('id') if event.get('position') else None,
            'position_name': event.get('position', {}).get('name') if event.get('position') else None,
            'location_x': event.get('location', [None, None])[0] if event.get('location') else None,
            'location_y': event.get('location', [None, None])[1] if event.get('location') and len(event['location']) > 1 else None,
            'duration': event.get('duration'),
            'under_pressure': event.get('under_pressure', False),
            'off_camera': event.get('off_camera', False),
            'out': event.get('out', False),
        }
        
        # Add pass-specific data
        if event.get('pass'):
            pass_data = event['pass']
            flattened.update({
                'pass_recipient_id': pass_data.get('recipient', {}).get('id') if pass_data.get('recipient') else None,
                'pass_recipient_name': pass_data.get('recipient', {}).get('name') if pass_data.get('recipient') else None,
                'pass_length': pass_data.get('length'),
                'pass_angle': pass_data.get('angle'),
                'pass_height_id': pass_data.get('height', {}).get('id') if pass_data.get('height') else None,
                'pass_height_name': pass_data.get('height', {}).get('name') if pass_data.get('height') else None,
                'pass_end_location_x': pass_data.get('end_location', [None, None])[0] if pass_data.get('end_location') else None,
                'pass_end_location_y': pass_data.get('end_location', [None, None])[1] if pass_data.get('end_location') and len(pass_data['end_location']) > 1 else None,
                'pass_body_part_id': pass_data.get('body_part', {}).get('id') if pass_data.get('body_part') else None,
                'pass_body_part_name': pass_data.get('body_part', {}).get('name') if pass_data.get('body_part') else None,
                'pass_type_id': pass_data.get('type', {}).get('id') if pass_data.get('type') else None,
                'pass_type_name': pass_data.get('type', {}).get('name') if pass_data.get('type') else None,
                'pass_outcome_id': pass_data.get('outcome', {}).get('id') if pass_data.get('outcome') else None,
                'pass_outcome_name': pass_data.get('outcome', {}).get('name') if pass_data.get('outcome') else None,
            })
        
        # Add shot-specific data
        if event.get('shot'):
            shot_data = event['shot']
            flattened.update({
                'shot_statsbomb_xg': shot_data.get('statsbomb_xg'),
                'shot_end_location_x': shot_data.get('end_location', [None, None, None])[0] if shot_data.get('end_location') else None,
                'shot_end_location_y': shot_data.get('end_location', [None, None, None])[1] if shot_data.get('end_location') and len(shot_data['end_location']) > 1 else None,
                'shot_end_location_z': shot_data.get('end_location', [None, None, None])[2] if shot_data.get('end_location') and len(shot_data['end_location']) > 2 else None,
                'shot_technique_id': shot_data.get('technique', {}).get('id') if shot_data.get('technique') else None,
                'shot_technique_name': shot_data.get('technique', {}).get('name') if shot_data.get('technique') else None,
                'shot_outcome_id': shot_data.get('outcome', {}).get('id') if shot_data.get('outcome') else None,
                'shot_outcome_name': shot_data.get('outcome', {}).get('name') if shot_data.get('outcome') else None,
                'shot_type_id': shot_data.get('type', {}).get('id') if shot_data.get('type') else None,
                'shot_type_name': shot_data.get('type', {}).get('name') if shot_data.get('type') else None,
                'shot_body_part_id': shot_data.get('body_part', {}).get('id') if shot_data.get('body_part') else None,
                'shot_body_part_name': shot_data.get('body_part', {}).get('name') if shot_data.get('body_part') else None,
            })
        
        return flattened
    
    def convert_all(self, max_event_files=50):
        """Convert all data types to CSV"""
        print("Starting StatsBomb data conversion to CSV...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Convert each data type
        competitions_df = self.convert_competitions()
        matches_df = self.convert_matches()
        lineups_df = self.convert_lineups()
        self.convert_events(max_files=max_event_files)
        
        # Create summary
        summary = {
            'competitions': len(competitions_df),
            'matches': len(matches_df),
            'player_match_records': len(lineups_df),
            'event_files_processed': min(max_event_files, len(list((self.data_dir / 'events').glob('*.json')))),
        }
        
        print("\nConversion Summary:")
        print("-" * 20)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / 'conversion_summary.csv', index=False)
        
        print(f"\nAll CSV files saved to: {self.output_dir}")
        return summary

def main():
    """Main function to run the conversion"""
    converter = StatsBombCSVConverter()
    
    # Convert all data (limit events to first 50 files for demo)
    summary = converter.convert_all(max_event_files=50)
    
    print("\nCSV conversion completed!")
    print("You can now open these files in Excel, Google Sheets, or any data analysis tool.")

if __name__ == "__main__":
    main()
