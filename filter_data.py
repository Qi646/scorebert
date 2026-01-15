import pandas as pd

def filter_pdmx(csv_path, output_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Initial count: {len(df)}")
    
    # Filtering criteria based on Phase 1.2 + Stricter Quality
    # 1. Instrumentation: n_tracks in [1, 2] (Piano Solo or Grand Staff)
    # 2. Length: 30s <= duration <= 300s
    # 3. Density: n_notes >= 50
    # 4. Valid subset: subset:all_valid == True
    # 5. Deduplicated & Rated: subset:rated_deduplicated == True
    # 6. High Rating: rating >= 4.0
    
    filtered_df = df[
        (df['n_tracks'].isin([1, 2])) & 
        (df['song_length.seconds'] >= 30) & 
        (df['song_length.seconds'] <= 300) & 
        (df['n_notes'] >= 50) & 
        (df['subset:all_valid'] == True) &
        (df['subset:rated_deduplicated'] == True) &
        (df['rating'] >= 4.0)
    ]
    
    print(f"Filtered count: {len(filtered_df)}")
    
    # Save the 'mxl' paths
    mxl_paths = filtered_df['mxl'].tolist()
    
    # Clean up paths (they start with ./mxl/ but the zip might have different structure)
    # Based on the CSV, they look like ./mxl/1/11/Qm...mxl
    
    with open(output_path, 'w') as f:
        for path in mxl_paths:
            # We want just the part relative to the 'mxl' root in the tarball if possible
            # But for now let's save exactly what's in the CSV
            f.write(path + '\n')
            
    print(f"Saved {len(mxl_paths)} paths to {output_path}")

if __name__ == "__main__":
    filter_pdmx('data/PDMX.csv', 'data/filtered_mxl_paths.txt')
