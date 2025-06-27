import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class AdvancedF1DataLoader:
    def __init__(self):
        # Updated data URLs with your new comprehensive dataset
        self.data_urls = {
            'sprint_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sprint_results-HecUwmna0Ga6KNTqLlP1dXXvOvgVgh.csv',
            'status': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/status-6C6u6QFgkgwoenfck85uqafyASuLOv.csv',
            'circuits': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/circuits-Xk7rlMpjaDg6eQgF05436JLdxm7HP5.csv',
            'pit_stops': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/pit_stops-wFxjzwVAiv5kNaVq6IzYftaLKl1tdl.csv',
            'constructors': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/constructors-0hulyHaqqzQbk4e3rtOQUfTJpep5sy.csv',
            'constructor_standings': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/constructor_standings-9l3y7oP5Zn3NK3O6WgQWkDxVElTVfe.csv',
            'drivers': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/drivers-z4PEkaeo8llAmAm4OW4afLGt8N70C5.csv',
            'constructor_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/constructor_results-yXfucYGuCuJY91SEvUJ4tJeMLhDP1j.csv',
            'driver_standings': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/driver_standings-qE2KbIDThJHv90uecSfrKVCgmImNLV.csv',
            # Previous data sources
            'seasons': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/seasons-hs9GJLLWKJFhEhMfr1AOU9HKY8oeyK.csv',
            'races': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/races-QUJ9dlQpgRzhjobaUk0eblDAoPpsrH.csv',
            'qualifying': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/qualifying-Fl7YfWJFvtszwzdIOsp3msmQEujBud.csv',
            'results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/results-P6NQQAMhGzgbfM8kZqHewfGHkRnJxC.csv'
        }
        self.dataframes = {}
        self.f1_api_base = "http://ergast.com/api/f1"
        
    def load_csv_data(self):
        """Load all F1 datasets from URLs"""
        print("Loading comprehensive F1 datasets...")
        
        for name, url in self.data_urls.items():
            try:
                print(f"Loading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Read CSV data
                df = pd.read_csv(StringIO(response.text))
                self.dataframes[name] = df
                print(f"✓ {name}: {len(df)} records loaded")
                
            except Exception as e:
                print(f"✗ Error loading {name}: {str(e)}")
                continue
        
        print(f"\nSuccessfully loaded {len(self.dataframes)} datasets")
        return self.dataframes
    
    def fetch_live_f1_data(self):
        """Fetch live F1 data from Ergast API"""
        print("Fetching live F1 data from Ergast API...")
        
        try:
            # Get current season schedule
            current_year = datetime.now().year
            schedule_url = f"{self.f1_api_base}/{current_year}.json"
            
            response = requests.get(schedule_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                races = data['MRData']['RaceTable']['Races']
                
                # Process upcoming races
                upcoming_races = []
                current_date = datetime.now()
                
                for race in races:
                    race_date = datetime.strptime(race['date'], '%Y-%m-%d')
                    if race_date >= current_date:
                        upcoming_races.append({
                            'round': race['round'],
                            'name': race['raceName'],
                            'date': race['date'],
                            'time': race.get('time', ''),
                            'circuit': race['Circuit']['circuitName'],
                            'location': race['Circuit']['Location']['locality'],
                            'country': race['Circuit']['Location']['country'],
                            'circuit_id': race['Circuit']['circuitId']
                        })
                
                print(f"✓ Found {len(upcoming_races)} upcoming races")
                return upcoming_races
            
        except Exception as e:
            print(f"✗ Error fetching live F1 data: {str(e)}")
            
        return []
    
    def get_current_standings(self):
        """Get current driver and constructor standings"""
        print("Fetching current F1 standings...")
        
        try:
            current_year = datetime.now().year
            
            # Driver standings
            driver_url = f"{self.f1_api_base}/{current_year}/driverStandings.json"
            driver_response = requests.get(driver_url, timeout=10)
            
            # Constructor standings  
            constructor_url = f"{self.f1_api_base}/{current_year}/constructorStandings.json"
            constructor_response = requests.get(constructor_url, timeout=10)
            
            standings = {}
            
            if driver_response.status_code == 200:
                data = driver_response.json()
                driver_standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
                standings['drivers'] = [
                    {
                        'position': int(driver['position']),
                        'driver_id': driver['Driver']['driverId'],
                        'name': f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}",
                        'team': driver['Constructors'][0]['name'],
                        'points': float(driver['points']),
                        'wins': int(driver['wins'])
                    }
                    for driver in driver_standings
                ]
            
            if constructor_response.status_code == 200:
                data = constructor_response.json()
                constructor_standings = data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
                standings['constructors'] = [
                    {
                        'position': int(constructor['position']),
                        'constructor_id': constructor['Constructor']['constructorId'],
                        'name': constructor['Constructor']['name'],
                        'points': float(constructor['points']),
                        'wins': int(constructor['wins'])
                    }
                    for constructor in constructor_standings
                ]
            
            print(f"✓ Current standings loaded")
            return standings
            
        except Exception as e:
            print(f"✗ Error fetching standings: {str(e)}")
            return {}
    
    def analyze_data_completeness(self):
        """Analyze completeness and quality of loaded data"""
        print("\n" + "="*80)
        print("COMPREHENSIVE F1 DATA ANALYSIS")
        print("="*80)
        
        total_records = 0
        data_quality = {}
        
        for name, df in self.dataframes.items():
            records = len(df)
            total_records += records
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            
            data_quality[name] = {
                'records': records,
                'columns': len(df.columns),
                'missing_percentage': round(missing_pct, 2),
                'data_types': df.dtypes.value_counts().to_dict(),
                'date_range': self._get_date_range(df)
            }
            
            print(f"\n{name.upper()}:")
            print(f"  Records: {records:,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Missing Data: {missing_pct:.1f}%")
            print(f"  Date Range: {data_quality[name]['date_range']}")
        
        print(f"\nTOTAL RECORDS: {total_records:,}")
        print(f"DATASETS: {len(self.dataframes)}")
        
        return data_quality
    
    def _get_date_range(self, df):
        """Extract date range from dataframe"""
        date_columns = ['date', 'dob', 'year']
        for col in date_columns:
            if col in df.columns:
                try:
                    if col == 'year':
                        years = pd.to_numeric(df[col], errors='coerce').dropna()
                        if not years.empty:
                            return f"{int(years.min())} - {int(years.max())}"
                    else:
                        dates = pd.to_datetime(df[col], errors='coerce').dropna()
                        if not dates.empty:
                            return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                except:
                    continue
        return "Unknown"
    
    def create_master_dataset(self):
        """Create comprehensive master dataset with all information"""
        print("Creating master F1 dataset...")
        
        # Start with race results as base
        if 'results' not in self.dataframes:
            print("Error: Results data not available")
            return None
            
        master_df = self.dataframes['results'].copy()
        print(f"Base results: {len(master_df)} records")
        
        # Merge with races information
        if 'races' in self.dataframes:
            races_df = self.dataframes['races'].copy()
            master_df = master_df.merge(races_df, on='raceId', how='left', suffixes=('', '_race'))
            print(f"After races merge: {len(master_df)} records")
        
        # Merge with drivers information
        if 'drivers' in self.dataframes:
            drivers_df = self.dataframes['drivers'].copy()
            master_df = master_df.merge(drivers_df, on='driverId', how='left', suffixes=('', '_driver'))
            print(f"After drivers merge: {len(master_df)} records")
        
        # Merge with constructors information
        if 'constructors' in self.dataframes:
            constructors_df = self.dataframes['constructors'].copy()
            master_df = master_df.merge(constructors_df, on='constructorId', how='left', suffixes=('', '_constructor'))
            print(f"After constructors merge: {len(master_df)} records")
        
        # Merge with circuits information
        if 'circuits' in self.dataframes:
            circuits_df = self.dataframes['circuits'].copy()
            master_df = master_df.merge(circuits_df, on='circuitId', how='left', suffixes=('', '_circuit'))
            print(f"After circuits merge: {len(master_df)} records")
        
        # Merge with qualifying data
        if 'qualifying' in self.dataframes:
            qualifying_df = self.dataframes['qualifying'].copy()
            # Aggregate qualifying data per driver per race
            quali_agg = qualifying_df.groupby(['raceId', 'driverId']).agg({
                'position': 'first',
                'q1': 'first',
                'q2': 'first',
                'q3': 'first'
            }).reset_index()
            quali_agg.rename(columns={'position': 'quali_position'}, inplace=True)
            master_df = master_df.merge(quali_agg, on=['raceId', 'driverId'], how='left', suffixes=('', '_quali'))
            print(f"After qualifying merge: {len(master_df)} records")
        
        # Merge with pit stops data
        if 'pit_stops' in self.dataframes:
            pit_stops_df = self.dataframes['pit_stops'].copy()
            # Aggregate pit stop statistics
            pit_agg = pit_stops_df.groupby(['raceId', 'driverId']).agg({
                'stop': 'count',
                'duration': ['mean', 'min', 'max'],
                'milliseconds': ['mean', 'sum']
            }).reset_index()
            
            # Flatten column names
            pit_agg.columns = ['raceId', 'driverId', 'pit_stops_count', 'avg_pit_duration', 
                              'min_pit_duration', 'max_pit_duration', 'avg_pit_ms', 'total_pit_ms']
            
            master_df = master_df.merge(pit_agg, on=['raceId', 'driverId'], how='left', suffixes=('', '_pit'))
            print(f"After pit stops merge: {len(master_df)} records")
        
        # Merge with driver standings (for championship context)
        if 'driver_standings' in self.dataframes:
            standings_df = self.dataframes['driver_standings'].copy()
            standings_df.rename(columns={
                'position': 'championship_position',
                'points': 'championship_points',
                'wins': 'championship_wins'
            }, inplace=True)
            master_df = master_df.merge(
                standings_df[['raceId', 'driverId', 'championship_position', 'championship_points', 'championship_wins']], 
                on=['raceId', 'driverId'], how='left', suffixes=('', '_standings')
            )
            print(f"After driver standings merge: {len(master_df)} records")
        
        # Merge with constructor standings
        if 'constructor_standings' in self.dataframes:
            const_standings_df = self.dataframes['constructor_standings'].copy()
            const_standings_df.rename(columns={
                'position': 'constructor_championship_position',
                'points': 'constructor_championship_points',
                'wins': 'constructor_championship_wins'
            }, inplace=True)
            master_df = master_df.merge(
                const_standings_df[['raceId', 'constructorId', 'constructor_championship_position', 
                                   'constructor_championship_points', 'constructor_championship_wins']], 
                on=['raceId', 'constructorId'], how='left', suffixes=('', '_const_standings')
            )
            print(f"After constructor standings merge: {len(master_df)} records")
        
        # Add sprint results if available
        if 'sprint_results' in self.dataframes:
            sprint_df = self.dataframes['sprint_results'].copy()
            sprint_agg = sprint_df.groupby(['raceId', 'driverId']).agg({
                'position': 'first',
                'points': 'first',
                'grid': 'first'
            }).reset_index()
            sprint_agg.rename(columns={
                'position': 'sprint_position',
                'points': 'sprint_points',
                'grid': 'sprint_grid'
            }, inplace=True)
            master_df = master_df.merge(sprint_agg, on=['raceId', 'driverId'], how='left', suffixes=('', '_sprint'))
            print(f"After sprint results merge: {len(master_df)} records")
        
        print(f"Master dataset created with {len(master_df)} records and {len(master_df.columns)} columns")
        return master_df
    
    def load_all_data(self):
        """Load all data sources and create comprehensive dataset"""
        print("Starting comprehensive F1 data loading...")
        
        # Load CSV data
        self.load_csv_data()
        
        # Fetch live data
        upcoming_races = self.fetch_live_f1_data()
        current_standings = self.get_current_standings()
        
        # Analyze data quality
        data_quality = self.analyze_data_completeness()
        
        # Create master dataset
        master_df = self.create_master_dataset()
        
        return {
            'dataframes': self.dataframes,
            'master_dataset': master_df,
            'upcoming_races': upcoming_races,
            'current_standings': current_standings,
            'data_quality': data_quality
        }

def main():
    """Main function to load all F1 data"""
    loader = AdvancedF1DataLoader()
    all_data = loader.load_all_data()
    
    print("\n" + "="*80)
    print("F1 DATA LOADING COMPLETED")
    print("="*80)
    print(f"Datasets loaded: {len(all_data['dataframes'])}")
    print(f"Master dataset records: {len(all_data['master_dataset']) if all_data['master_dataset'] is not None else 0}")
    print(f"Upcoming races: {len(all_data['upcoming_races'])}")
    print(f"Current standings available: {'Yes' if all_data['current_standings'] else 'No'}")
    
    return all_data

if __name__ == "__main__":
    all_data = main()
