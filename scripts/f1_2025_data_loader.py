import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json

class F1_2025_DataLoader:
    def __init__(self):
        self.data_2025 = {}
        self.current_form = {}
        self.circuit_performance = {}
        self.team_lineups_2025 = {}
        
    def load_2025_data(self):
        """Load all 2025 data from CSV files"""
        print("\nLoading 2025 F1 data from provided CSV files...")
        
        # URLs for the 2025 data files
        urls = {
            'race_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/F1_2025_RaceResults-GgPOrRk5x4ixmpmBGk5RpSFLgmifLp.csv',
            'qualifying_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/F1_2025_QualifyingResults-GbiEN83ilOqXKMJewg9M0rVDTMH4kb.csv',
            'sprint_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/F1_2025_SprintResults-MPn8mZ37cdEwop0cs1TPIgOIdVr7RH.csv',
            'sprint_qualifying': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/F1_2025_SprintQualifyingResults-YXDDumm6YT6JMtDfSfcxB3H1lmmbPy.csv'
        }
        
        try:
            # Load race results
            print("Loading race results...")
            self.data_2025['race_results'] = pd.read_csv(urls['race_results'])
            print(f"Loaded {len(self.data_2025['race_results'])} race result records")
            
            # Load qualifying results
            print("Loading qualifying results...")
            self.data_2025['qualifying_results'] = pd.read_csv(urls['qualifying_results'])
            print(f"Loaded {len(self.data_2025['qualifying_results'])} qualifying records")
            
            # Load sprint results
            print("Loading sprint results...")
            self.data_2025['sprint_results'] = pd.read_csv(urls['sprint_results'])
            print(f"Loaded {len(self.data_2025['sprint_results'])} sprint result records")
            
            # Load sprint qualifying
            print("Loading sprint qualifying results...")
            self.data_2025['sprint_qualifying'] = pd.read_csv(urls['sprint_qualifying'])
            print(f"Loaded {len(self.data_2025['sprint_qualifying'])} sprint qualifying records")
            
            print("2025 F1 data loaded successfully!")
            
        except Exception as e:
            print(f"Error loading 2025 data: {e}")
            # Fallback to sample data
            self._create_sample_data()
            
        return self.data_2025
    
    def _create_sample_data(self):
        """Create sample data if CSV loading fails"""
        print("Creating sample 2025 data...")
        
        # Sample race results
        self.data_2025['race_results'] = pd.DataFrame({
            'Track': ['Bahrain', 'Saudi Arabia', 'Australia'] * 20,
            'Position': list(range(1, 21)) * 3,
            'No': [1, 4, 16, 44, 63, 12, 14, 18, 22, 6, 87, 31, 10, 7, 23, 55, 27, 5, 30, 81] * 3,
            'Driver': ['Max Verstappen', 'Lando Norris', 'Charles Leclerc', 'Lewis Hamilton', 
                      'George Russell', 'Andrea Kimi Antonelli', 'Fernando Alonso', 'Lance Stroll',
                      'Yuki Tsunoda', 'Isack Hadjar', 'Oliver Bearman', 'Esteban Ocon',
                      'Pierre Gasly', 'Jack Doohan', 'Alex Albon', 'Carlos Sainz',
                      'Nico Hulkenberg', 'Gabriel Bortoleto', 'Liam Lawson', 'Oscar Piastri'] * 3,
            'Team': ['Red Bull', 'McLaren', 'Ferrari', 'Ferrari', 'Mercedes', 'Mercedes',
                    'Aston Martin', 'Aston Martin', 'RB', 'RB', 'Haas', 'Haas',
                    'Alpine', 'Alpine', 'Williams', 'Williams', 'Kick Sauber', 'Kick Sauber',
                    'Red Bull', 'McLaren'] * 3,
            'Points': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 3
        })
        
        print("Sample 2025 data created.")
    
    def analyze_2025_season(self):
        """Analyze the 2025 season data"""
        print("\nAnalyzing 2025 F1 season...")
        
        if 'race_results' not in self.data_2025:
            print("No race results data available for analysis")
            return {}
        
        race_results = self.data_2025['race_results']
        
        # Driver standings analysis
        driver_standings = race_results.groupby('Driver').agg({
            'Points': 'sum',
            'Position': 'mean',
            'Track': 'count'
        }).reset_index()
        
        driver_standings.rename(columns={
            'Points': 'Total_Points',
            'Position': 'Avg_Position', 
            'Track': 'Races'
        }, inplace=True)
        
        driver_standings = driver_standings.sort_values('Total_Points', ascending=False)
        driver_standings_dict = driver_standings.set_index('Driver').to_dict('index')
        
        # Team standings analysis
        team_standings = race_results.groupby('Team').agg({
            'Points': 'sum',
            'Position': 'mean',
            'Track': 'count'
        }).reset_index()
        
        team_standings.rename(columns={
            'Points': 'Total_Points',
            'Position': 'Avg_Position',
            'Track': 'Races'
        }, inplace=True)
        
        team_standings = team_standings.sort_values('Total_Points', ascending=False)
        team_standings_dict = team_standings.set_index('Team').to_dict('index')
        
        season_analysis = {
            'driver_standings': driver_standings_dict,
            'team_standings': team_standings_dict,
            'total_races': len(race_results['Track'].unique()),
            'total_drivers': len(race_results['Driver'].unique())
        }
        
        print("2025 Season Analysis Complete.")
        print(f"Analyzed {season_analysis['total_races']} races with {season_analysis['total_drivers']} drivers")
        
        return season_analysis
    
    def get_2025_team_lineups(self):
        """Extract CORRECT 2025 team lineups with driver numbers"""
        print("\nExtracting CORRECT 2025 team lineups...")
        
        # CORRECT 2025 team lineups with accurate driver numbers
        self.team_lineups_2025 = {
            "Red Bull": [
                {"name": "Max Verstappen", "number": 1},
                {"name": "Liam Lawson", "number": 30}
            ],
            "Ferrari": [
                {"name": "Charles Leclerc", "number": 16},
                {"name": "Lewis Hamilton", "number": 44}
            ],
            "McLaren": [
                {"name": "Lando Norris", "number": 4},
                {"name": "Oscar Piastri", "number": 81}
            ],
            "Mercedes": [
                {"name": "George Russell", "number": 63},
                {"name": "Andrea Kimi Antonelli", "number": 12}
            ],
            "Aston Martin": [
                {"name": "Fernando Alonso", "number": 14},
                {"name": "Lance Stroll", "number": 18}
            ],
            "RB": [
                {"name": "Yuki Tsunoda", "number": 22},
                {"name": "Isack Hadjar", "number": 6}
            ],
            "Haas": [
                {"name": "Oliver Bearman", "number": 87},
                {"name": "Esteban Ocon", "number": 31}
            ],
            "Alpine": [
                {"name": "Pierre Gasly", "number": 10},
                {"name": "Jack Doohan", "number": 7}
            ],
            "Williams": [
                {"name": "Alex Albon", "number": 23},
                {"name": "Carlos Sainz", "number": 55}
            ],
            "Kick Sauber": [
                {"name": "Nico Hulkenberg", "number": 27},
                {"name": "Gabriel Bortoleto", "number": 5}
            ]
        }
        
        print("CORRECT 2025 Team Lineups:")
        for team, drivers in self.team_lineups_2025.items():
            driver_list = [f"#{d['number']} {d['name']}" for d in drivers]
            print(f"  {team}: {', '.join(driver_list)}")
            
        # Highlight major changes
        print("\n🔥 MAJOR 2025 CHANGES:")
        print("  • Lewis Hamilton (#44): Mercedes → Ferrari")
        print("  • Carlos Sainz (#55): Ferrari → Williams") 
        print("  • Liam Lawson (#30): Replaces Perez at Red Bull")
        print("  • Andrea Kimi Antonelli (#12): Rookie at Mercedes")
        print("  • Isack Hadjar (#6): Rookie at RB")
        print("  • Jack Doohan (#7): Rookie at Alpine")
        print("  • Gabriel Bortoleto (#5): Rookie at Kick Sauber")
        print("  • Oliver Bearman (#87): Full-time at Haas")
        print("  • Esteban Ocon (#31): Alpine → Haas")
        
        return self.team_lineups_2025
    
    def get_current_form(self):
        """Calculate current driver form based on 2025 race data"""
        print("\nCalculating current driver form from 2025 data...")
        
        if 'race_results' not in self.data_2025:
            return self._get_sample_form()
        
        race_results = self.data_2025['race_results']
        
        # Calculate form for each driver
        driver_form = {}
        
        for driver in race_results['Driver'].unique():
            driver_data = race_results[race_results['Driver'] == driver]
            
            total_points = driver_data['Points'].sum()
            avg_position = driver_data['Position'].mean()
            recent_positions = driver_data['Position'].tail(5).tolist()
            
            driver_form[driver] = {
                "total_points": int(total_points),
                "avg_position": round(avg_position, 2),
                "recent_form": recent_positions
            }
        
        print(f"Calculated form for {len(driver_form)} drivers")
        return driver_form
    
    def _get_sample_form(self):
        """Sample driver form data"""
        return {
            "Max Verstappen": {"total_points": 393, "avg_position": 1.8, "recent_form": [1, 1, 2, 1, 1]},
            "Lando Norris": {"total_points": 331, "avg_position": 2.4, "recent_form": [2, 1, 3, 2, 1]},
            "Charles Leclerc": {"total_points": 307, "avg_position": 2.8, "recent_form": [3, 2, 1, 4, 2]},
            "Oscar Piastri": {"total_points": 279, "avg_position": 3.2, "recent_form": [4, 3, 2, 3, 3]},
            "Lewis Hamilton": {"total_points": 244, "avg_position": 3.8, "recent_form": [5, 4, 3, 5, 4]},
            "George Russell": {"total_points": 192, "avg_position": 4.5, "recent_form": [6, 5, 4, 6, 5]},
            "Carlos Sainz": {"total_points": 174, "avg_position": 5.2, "recent_form": [7, 6, 5, 7, 6]},
            "Liam Lawson": {"total_points": 152, "avg_position": 6.1, "recent_form": [8, 7, 6, 8, 7]},
        }
    
    def get_circuit_performance(self):
        """Extract circuit performance data from 2025 results"""
        print("\nExtracting circuit performance data...")
        
        if 'race_results' not in self.data_2025:
            return self._get_sample_circuit_performance()
        
        race_results = self.data_2025['race_results']
        
        # Get winners by track
        circuit_winners = {}
        for track in race_results['Track'].unique():
            track_data = race_results[race_results['Track'] == track]
            winners = track_data[track_data['Position'] == 1]
            if not winners.empty:
                winner = winners.iloc[0]
                circuit_winners[track] = {
                    'winner': winner['Driver'],
                    'team': winner['Team']
                }
        
        print(f"Extracted performance data for {len(circuit_winners)} circuits")
        return circuit_winners
    
    def _get_sample_circuit_performance(self):
        """Sample circuit performance data"""
        return {
            "Bahrain": {"winner": "Max Verstappen", "team": "Red Bull"},
            "Saudi Arabia": {"winner": "Lando Norris", "team": "McLaren"},
            "Australia": {"winner": "Charles Leclerc", "team": "Ferrari"}
        }

def main():
    """Main function to load and analyze 2025 F1 data"""
    loader = F1_2025_DataLoader()
    
    # Load all 2025 data
    data_2025 = loader.load_2025_data()
    
    # Analyze the season
    season_analysis = loader.analyze_2025_season()
    
    # Get team lineups
    team_lineups = loader.get_2025_team_lineups()
    
    # Get driver form
    driver_form = loader.get_current_form()
    
    # Get circuit performance
    circuit_performance = loader.get_circuit_performance()
    
    print("\n" + "="*60)
    print("2025 F1 Data Loading Complete!")
    print("="*60)
    
    return {
        'data': data_2025,
        'analysis': season_analysis,
        'lineups': team_lineups,
        'form': driver_form,
        'circuits': circuit_performance
    }

if __name__ == "__main__":
    result = main()
