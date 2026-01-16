import os
import glob
import torch
from tqdm import tqdm
from data_processor import ScoreProcessor
from typing import List, Dict, Any, Optional

def worker_process(file_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Worker function for multiprocessing.
    """
    processor = ScoreProcessor()
    inputs, labels = processor.process_file(file_path)
    
    if inputs is None or labels is None:
        return None
        
    # Convert to Tensors immediately to save space/time later
    try:
        # Inputs
        # MIDI pitch is technically int, but embedding layers often take Long
        input_pitch = torch.tensor([x['pitch'] for x in inputs], dtype=torch.long) 
        input_duration = torch.tensor([x['duration'] for x in inputs], dtype=torch.float)
        # Onset is kept for debugging/inference, though not strictly input to BERT (pos encoding handles order)
        input_onset = torch.tensor([x['onset'] for x in inputs], dtype=torch.float) 
        
        # Labels
        label_grid = torch.tensor([x['grid_label'] for x in labels], dtype=torch.long)
        label_spelling = torch.tensor([x['spelling_label'] for x in labels], dtype=torch.long)
        label_staff = torch.tensor([x['staff_label'] for x in labels], dtype=torch.long)
        label_tuplet = torch.tensor([x['tuplet_label'] for x in labels], dtype=torch.long)
        
        return {
            'input_pitch': input_pitch,
            'input_duration': input_duration,
            'input_onset': input_onset,
            'label_grid': label_grid,
            'label_spelling': label_spelling,
            'label_staff': label_staff,
            'label_tuplet': label_tuplet
        }
    except Exception:
        return None

def preprocess_all(data_dir: str, output_path: str):
    """
    Runs ScoreProcessor on all found .mxl files and saves the processed dataset to disk.
    """
    files = glob.glob(os.path.join(data_dir, "**/*.mxl"), recursive=True)
    print(f"Found {len(files)} files to preprocess.")
    
    processed_data: List[Dict[str, torch.Tensor]] = []
    
    from multiprocessing import Pool, cpu_count
    
    # Use roughly 75% of cores to leave some for system responsiveness
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Starting processing with {num_processes} workers...")

    with Pool(processes=num_processes) as pool:
        # imap_unordered is faster if order doesn't matter (it doesn't for training)
        for result in tqdm(pool.imap_unordered(worker_process, files), total=len(files)):
            if result is not None:
                processed_data.append(result)
            
    print(f"Successfully processed {len(processed_data)} files.")
    print(f"Saving to {output_path}...")
    torch.save(processed_data, output_path)
    print("Done.")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    preprocess_all("mxl", "data/processed_dataset.pt")
