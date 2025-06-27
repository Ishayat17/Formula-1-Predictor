import pandas as pd
import numpy as np
import json
from datetime import datetime
from f1_2025_data_loader import F1_2025_DataLoader

class F1PredictorUpdater2025:
    def __init__(self):
        self.loader = F1_2025_DataLoader()
        self.current_season_data = {}
        self.updated_driver_ratings = {}
        self.updated_team_performance = {}
        
    def load_and_process_2025_data(self):
        """Load and process all 2025 data"""
        print("Loading and processing 2025 F1 data for predictor update...")
        
        # Load 2025 data
        dataframes_2025 = self.loader.load_2025_data()
        season_analysis = self.loader.analyze_2025_season()
        team_lineups = self.loader.get_2025_team_lineups()
        driver_form = self.loader.get_current_form()
        circuit_performance = self.loader.get_circuit_performance()
        
        self.current_season_data = {
            'dataframes': dataframes_2025,
            'analysis': season_analysis,
            'team_lineups': team_lineups,
            'driver_form': driver_form,
            'circuit_performance': circuit_performance
        }
        
        return self.current_season_data
    
    def update_driver_ratings(self):
        """Update driver ratings based on 2025 performance"""
        print("Updating driver ratings based on 2025 performance...")
        
        driver_form = self.current_season_data['driver_form']
        
        # Calculate updated ratings
        for driver, stats in driver_form.items():
            # Base rating calculation
            points_rating = min(100, stats['total_points'] / 10)  # Scale points to 0-100
            position_rating = max(0, 100 - (stats['avg_position'] - 1) * 5)  # Better position = higher rating
            
            # Recent form bonus/penalty
            recent_form = stats['recent_form']
            form_rating = 0
            for pos in recent_form:
                if pos <= 3:
                    form_rating += 20
                elif pos <= 10:
                    form_rating += 10
                elif pos <= 15:
                    form_rating += 5
            form_rating = form_rating / len(recent_form) if recent_form else 0
            
            # Combined rating
            overall_rating = (points_rating * 0.4 + position_rating * 0.4 + form_rating * 0.2)
            
            self.updated_driver_ratings[driver] = {
                'overall_rating': round(overall_rating, 2),
                'points_rating': round(points_rating, 2),
                'position_rating': round(position_rating, 2),
                'form_rating': round(form_rating, 2),
                'current_points': stats['total_points'],
                'avg_position': stats['avg_position'],
                'recent_form': recent_form
            }
        
        # Sort by overall rating
        self.updated_driver_ratings = dict(sorted(
            self.updated_driver_ratings.items(), 
            key=lambda x: x[1]['overall_rating'], 
            reverse=True
        ))
        
        print("Top 10 Driver Ratings for 2025:")
        for i, (driver, rating) in enumerate(list(self.updated_driver_ratings.items())[:10], 1):
            print(f"  {i}. {driver}: {rating['overall_rating']}/100 ({rating['current_points']} pts)")
        
        return self.updated_driver_ratings
    
    def update_team_performance(self):
        """Update team performance metrics"""
        print("Updating team performance metrics...")
        
        if 'team_standings' not in self.current_season_data['analysis']:
            return {}
        
        team_standings = self.current_season_data['analysis']['team_standings']
        
        for team, stats in team_standings.items():
            # Calculate team performance rating
            points_rating = min(100, stats['Total_Points'] / 20)  # Scale team points
            position_rating = max(0, 100 - (stats['Avg_Position'] - 1) * 3)
            
            # Team reliability (based on race completion)
            reliability_rating = min(100, (stats['Races'] / stats['Races']) * 100)  # Simplified
            
            overall_rating = (points_rating * 0.5 + position_rating * 0.3 + reliability_rating * 0.2)
            
            self.updated_team_performance[team] = {
                'overall_rating': round(overall_rating, 2),
                'points_rating': round(points_rating, 2),
                'position_rating': round(position_rating, 2),
                'reliability_rating': round(reliability_rating, 2),
                'total_points': stats['Total_Points'],
                'avg_position': stats['Avg_Position'],
                'races_completed': stats['Races']
            }
        
        # Sort by overall rating
        self.updated_team_performance = dict(sorted(
            self.updated_team_performance.items(),
            key=lambda x: x[1]['overall_rating'],
            reverse=True
        ))
        
        print("Team Performance Rankings for 2025:")
        for i, (team, rating) in enumerate(self.updated_team_performance.items(), 1):
            print(f"  {i}. {team}: {rating['overall_rating']}/100 ({rating['total_points']} pts)")
        
        return self.updated_team_performance
    
    def generate_2025_predictions_data(self):
        """Generate updated prediction data structure"""
        print("Generating 2025 prediction data structure...")
        
        # Updated driver data with 2025 information
        updated_drivers = []
        
        team_lineups = self.current_season_data['team_lineups']
        
        # Create driver entries with 2025 data
        for driver, rating_data in self.updated_driver_ratings.items():
            # Find driver's team
            driver_team = None
            for team, drivers in team_lineups.items():
                if driver in drivers:
                    driver_team = team
                    break
            
            if driver_team:
                # Map to consistent driver ID (simplified)
                driver_id = driver.lower().replace(' ', '_').replace('.', '')
                
                updated_drivers.append({
                    'driverId': driver_id,
                    'name': driver,
                    'team': driver_team,
                    'championshipPoints': rating_data['current_points'],
                    'avgPosition': rating_data['avg_position'],
                    'recentForm': rating_data['recent_form'][-5:] if len(rating_data['recent_form']) >= 5 else rating_data['recent_form'],
                    'overallRating': rating_data['overall_rating'],
                    'reliability': min(95, 70 + (rating_data['form_rating'] / 5)),  # Estimated reliability
                    'experience': 5.0,  # Default experience, would need historical data
                    'season': 2025
                })
        
        # Updated team factors based on 2025 performance
        updated_team_factors = {}
        for team, performance in self.updated_team_performance.items():
            # Convert performance rating to factor (1.0 = average, >1.0 = above average)
            factor = 0.8 + (performance['overall_rating'] / 100) * 0.4
            updated_team_factors[team] = round(factor, 3)
        
        prediction_data = {
            'drivers_2025': updated_drivers,
            'team_factors_2025': updated_team_factors,
            'circuit_winners_2025': self.current_season_data['circuit_performance'],
            'last_updated': datetime.now().isoformat(),
            'season': 2025
        }
        
        return prediction_data
    
    def save_updated_predictor_data(self):
        """Save all updated data for the predictor"""
        print("Saving updated predictor data...")
        
        # Generate prediction data
        prediction_data = self.generate_2025_predictions_data()
        
        # Save to files
        with open('f1_2025_predictor_data.json', 'w') as f:
            json.dump(prediction_data, f, indent=2, default=str)
        
        with open('f1_2025_driver_ratings.json', 'w') as f:
            json.dump(self.updated_driver_ratings, f, indent=2, default=str)
        
        with open('f1_2025_team_performance.json', 'w') as f:
            json.dump(self.updated_team_performance, f, indent=2, default=str)
        
        print("Updated predictor data saved successfully!")
        
        return prediction_data
    
    def run_full_update(self):
        """Run the complete update process"""
        print("Starting F1 Predictor 2025 Update Process...")
        print("="*60)
        
        # Load and process 2025 data
        self.load_and_process_2025_data()
        
        # Update ratings and performance
        self.update_driver_ratings()
        self.update_team_performance()
        
        # Save updated data
        prediction_data = self.save_updated_predictor_data()
        
        print("\n" + "="*60)
        print("F1 Predictor successfully updated with 2025 data!")
        print("="*60)
        
        return prediction_data

def main():
    """Main function to update the F1 predictor with 2025 data"""
    updater = F1PredictorUpdater2025()
    prediction_data = updater.run_full_update()
    
    print("\nUpdate Summary:")
    print(f"- Updated {len(prediction_data['drivers_2025'])} drivers")
    print(f"- Updated {len(prediction_data['team_factors_2025'])} teams")
    print(f"- Processed {len(prediction_data['circuit_winners_2025'])} circuits")
    print(f"- Data current as of: {prediction_data['last_updated']}")
    
    return prediction_data

if __name__ == "__main__":
    prediction_data = main()
from f1_2025_data_loader import F1_2025_DataLoader

class F1_2025_DataLoader:
    def __init__(self):
        self.data_2025 = {}
        self.current_form = {}
        self.circuit_performance = {}
        
    def load_2025_data(self):
        """Load all 2025 data from various sources"""
        print("\nLoading 2025 F1 data...")
        
        # Placeholder for loading dataframes (CSV, API, etc.)
        self.data_2025['races'] = pd.DataFrame({
            'raceId': [1, 2, 3], 'year': [2025, 2025, 2025], 
            'name': ['Race 1', 'Race 2', 'Race 3']
        })
        self.data_2025['results'] = pd.DataFrame({
            'resultId': [1, 2, 3], 'raceId': [1, 1, 2], 
            'driverId': [10, 20, 10], 'positionOrder': [1, 2, 1], 'points': [25, 18, 25]
        })
        self.data_2025['drivers'] = pd.DataFrame({
            'driverId': [10, 20, 30], 'driverRef': ['driver1', 'driver2', 'driver3'], 
            'nationality': ['Italian', 'British', 'German']
        })
        self.data_2025['constructors'] = pd.DataFrame({
            'constructorId': [1, 2, 3], 'name': ['Ferrari', 'Mercedes', 'Red Bull']
        })
        self.data_2025['driver_standings'] = pd.DataFrame({
            'driverStandingsId': [1, 2, 3], 'raceId': [3, 3, 3], 
            'driverId': [10, 20, 30], 'points': [75, 50, 25], 'position': [1, 2, 3]
        })
        self.data_2025['constructor_standings'] = pd.DataFrame({
            'constructorStandingsId': [1, 2, 3], 'raceId': [3, 3, 3], 
            'constructorId': [1, 2, 3], 'points': [150, 100, 50], 'position': [1, 2, 3]
        })
        
        print("2025 F1 data loaded successfully (placeholder data).")
        return self.data_2025
    
    def analyze_2025_season(self):
        """Analyze the 2025 season data"""
        print("\nAnalyzing 2025 F1 season...")
        
        # Example analysis: Team Standings
        constructor_standings = self.data_2025['constructor_standings']
        constructors = self.data_2025['constructors']
        
        team_standings = pd.merge(
            constructor_standings, constructors, on='constructorId', how='left'
        )
        team_standings = team_standings.groupby('name').agg(
            {'points': 'sum', 'position': 'mean'}
        ).reset_index()
        team_standings.rename(
            columns={'name': 'Team', 'points': 'Total_Points', 'position': 'Avg_Position'}, 
            inplace=True
        )
        team_standings['Avg_Position'] = team_standings['Avg_Position'].round(2)
        team_standings = team_standings.sort_values(by='Total_Points', ascending=False)
        team_standings['Races'] = 3  # Assuming 3 races for now
        team_standings = team_standings[['Team', 'Total_Points', 'Avg_Position', 'Races']]
        
        # Convert to dictionary for easier access
        team_standings_dict = team_standings.set_index('Team').to_dict('index')
        
        season_analysis = {
            'team_standings': team_standings_dict
        }
        
        print("2025 Season Analysis Complete.")
        return season_analysis
    
    def get_2025_team_lineups(self):
        """Extract 2025 team lineups with correct driver transfers"""
        print("\nExtracting 2025 team lineups with driver transfers...")
        
        # Updated 2025 team lineups with correct driver transfers
        team_lineups_2025 = {
            "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],  # Hamilton's big move
            "Red Bull Racing Honda RBPT": ["Max Verstappen", "Sergio Perez", "Yuki Tsunoda", "Liam Lawson"],
            "McLaren Mercedes": ["Lando Norris", "Oscar Piastri"],
            "Mercedes": ["George Russell", "Kimi Antonelli"],  # New young driver after Hamilton
            "Williams Mercedes": ["Carlos Sainz", "Alexander Albon", "Franco Colapinto"],  # Sainz's move
            "Aston Martin Aramco Mercedes": ["Fernando Alonso", "Lance Stroll"],
            "Alpine Renault": ["Pierre Gasly", "Esteban Ocon"],
            "Haas Ferrari": ["Nico Hulkenberg", "Kevin Magnussen"],
            "Kick Sauber Ferrari": ["Valtteri Bottas", "Guanyu Zhou"]
        }
        
        print("2025 Team Lineups (Updated with Transfers):")
        for team, drivers in team_lineups_2025.items():
            print(f"  {team}: {', '.join(drivers)}")
            
        # Highlight major transfers
        print("\n🔥 MAJOR 2025 TRANSFERS:")
        print("  • Lewis Hamilton: Mercedes → Ferrari")
        print("  • Carlos Sainz: Ferrari → Williams Mercedes") 
        print("  • Kimi Antonelli: New driver → Mercedes")
        
        return team_lineups_2025
    
    def get_current_form(self):
        """Simulate current driver form based on recent races"""
        print("\nSimulating current driver form...")
        
        # Example driver form (replace with actual data/calculations)
        driver_form = {
            "Max Verstappen": {
                "total_points": 75,
                "avg_position": 1.33,
                "recent_form": [1, 2, 1]
            },
            "Charles Leclerc": {
                "total_points": 50,
                "avg_position": 2.0,
                "recent_form": [2, 3, 2]
            },
            "Sergio Perez": {
                "total_points": 40,
                "avg_position": 3.0,
                "recent_form": [3, 4, 3]
            },
            "George Russell": {
                "total_points": 30,
                "avg_position": 4.0,
                "recent_form": [4, 5, 4]
            },
            "Lando Norris": {
                "total_points": 25,
                "avg_position": 5.0,
                "recent_form": [5, 6, 5]
            },
            "Lewis Hamilton": {
                "total_points": 20,
                "avg_position": 6.0,
                "recent_form": [6, 7, 6]
            },
            "Oscar Piastri": {
                "total_points": 15,
                "avg_position": 7.0,
                "recent_form": [7, 8, 7]
            },
            "Fernando Alonso": {
                "total_points": 10,
                "avg_position": 8.0,
                "recent_form": [8, 9, 8]
            },
            "Esteban Ocon": {
                "total_points": 5,
                "avg_position": 9.0,
                "recent_form": [9, 10, 9]
            },
            "Pierre Gasly": {
                "total_points": 2,
                "avg_position": 10.0,
                "recent_form": [10, 11, 10]
            }
        }
        
        print("Current driver form simulated.")
        return driver_form
    
    def get_circuit_performance(self):
        """Simulate circuit performance data"""
        print("\nSimulating circuit performance data...")
        
        # Example circuit performance (replace with actual data/calculations)
        circuit_performance = {
            "Melbourne": "Red Bull Racing Honda RBPT",
            "Shanghai": "Ferrari",
            "Miami": "McLaren Mercedes"
        }
        
        print("Circuit performance data simulated.")
        return circuit_performance
