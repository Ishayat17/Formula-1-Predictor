import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class F1DataLoader:
    def __init__(self):
        self.data_urls = {
            'sprint_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sprint_results-Y4Ic4CTL6hQnjb0aUY6Gsjq5zE1W2V.csv',
            'seasons': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/seasons-hs9GJLLWKJFhEhMfr1AOU9HKY8oeyK.csv',
            'races': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/races-QUJ9dlQpgRzhjobaUk0eblDAoPpsrH.csv',
            'qualifying': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/qualifying-Fl7YfWJFvtszwzdIOsp3msmQEujBud.csv',
            'results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/results-P6NQQAMhGzgbfM8kZqHewfGHkRnJxC.csv',
            'circuits': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/circuits-YS2b8VUceqiMjFOMWv2RVBEj57PCxB.csv',
            'status': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/status-aPr4Bd28dJlhdLcGMWO93Jf9CnE1UZ.csv',
            'constructor_results': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/constructor_results-Zyf36AZ5XdxBm6z5XwPbduiz4d4YGk.csv',
            'pit_stops': 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/pit_stops-DNz7vdGmA7jQGh5eP2diUXZVAn9UDp.csv'
        }
        self.dataframes = {}
    
    def load_data(self):
        """Load all F1 datasets from URLs"""
        print("Loading F1 datasets...")
        
        for name, url in self.data_urls.items():
            try:
                print(f"Loading {name}...")
                response = requests.get(url)
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
    
    def explore_data(self):
        """Explore the structure of loaded datasets"""
        print("\n" + "="*60)
        print("F1 DATA EXPLORATION")
        print("="*60)
        
        for name, df in self.dataframes.items():
            print(f"\n{name.upper()}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:")
            print(df.head(2))
            print("-" * 40)
    
    def get_data_summary(self):
        """Get summary statistics for all datasets"""
        summary = {}
        for name, df in self.dataframes.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'data_types': df.dtypes.value_counts().to_dict()
            }
        return summary

def main():
    """Main function to load and explore F1 data"""
    loader = F1DataLoader()
    
    # Load all datasets
    dataframes = loader.load_data()
    
    # Explore data structure
    loader.explore_data()
    
    # Get summary
    summary = loader.get_data_summary()
    print("\nDATA SUMMARY:")
    for name, stats in summary.items():
        print(f"{name}: {stats['rows']} rows, {stats['columns']} cols, {stats['missing_values']} missing")
    
    return dataframes

if __name__ == "__main__":
    dataframes = main()
