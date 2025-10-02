#!/usr/bin/env python3
"""
Simple script to run StatsBomb JSON to CSV conversion
Usage: python run_conversion.py
"""

from json_to_csv_converter import StatsBombCSVConverter
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Convert StatsBomb JSON data to CSV format')
    parser.add_argument('--data-dir', default='./open-data/data', 
                       help='Path to StatsBomb data directory')
    parser.add_argument('--output-dir', default='./csv_data', 
                       help='Output directory for CSV files')
    parser.add_argument('--max-events', type=int, default=50,
                       help='Maximum number of event files to process (default: 50)')
    parser.add_argument('--events-only', action='store_true',
                       help='Convert only events data')
    parser.add_argument('--no-events', action='store_true',
                       help='Skip events conversion (faster)')
    
    args = parser.parse_args()
    
    print("StatsBomb JSON to CSV Converter")
    print("=" * 40)
    
    converter = StatsBombCSVConverter(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    try:
        if args.events_only:
            print("Converting events data only...")
            converter.convert_events(max_files=args.max_events)
        elif args.no_events:
            print("Converting competitions, matches, and lineups (skipping events)...")
            converter.convert_competitions()
            converter.convert_matches()
            converter.convert_lineups()
        else:
            print("Converting all data types...")
            converter.convert_all(max_event_files=args.max_events)
        
        print("\n✅ Conversion completed successfully!")
        print(f"📁 Check output directory: {args.output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data directory not found: {e}")
        print("Make sure you've cloned the StatsBomb data first:")
        print("git clone https://github.com/statsbomb/open-data.git")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
