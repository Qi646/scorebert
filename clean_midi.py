import argparse
import torch
import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from score_bert import ScoreBERT
from data_processor import ScoreProcessor
from reconstructor import ScoreReconstructor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/scorebert_best.pt"
SEQ_LEN = 512

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    
    model = ScoreBERT().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def predict(model: ScoreBERT, input_events: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Runs inference on the input events using sliding windows.
    """
    # Convert to tensors
    pitches = torch.tensor([x['pitch'] for x in input_events], dtype=torch.long)
    durations = torch.tensor([x['duration'] for x in input_events], dtype=torch.float)
    onsets = [x['onset'] for x in input_events] # Keep for reconstruction
    
    num_events = len(pitches)
    predictions = []
    
    # Simple non-overlapping windows for inference (simplification)
    # Ideally we'd averaging overlapping logits
    for start in tqdm(range(0, num_events, SEQ_LEN), desc="Inference"):
        end = min(start + SEQ_LEN, num_events)
        
        chunk_pitch = pitches[start:end].to(DEVICE).unsqueeze(0) # (1, Seq)
        chunk_dur = durations[start:end].to(DEVICE).unsqueeze(0) # (1, Seq)
        
        # Pad if necessary
        if chunk_pitch.size(1) < SEQ_LEN:
            pad_len = SEQ_LEN - chunk_pitch.size(1)
            chunk_pitch = torch.cat([chunk_pitch, torch.full((1, pad_len), 128, device=DEVICE)], dim=1)
            chunk_dur = torch.cat([chunk_dur, torch.zeros((1, pad_len), device=DEVICE)], dim=1)
            
        with torch.no_grad():
            # No padding mask needed for inference on the valid part, but technically the padded part should be masked
            # However, ScoreBERT ignores padding in loss, here we just ignore outputs
            r_logits, s_logits, st_logits, t_logits = model(chunk_pitch, chunk_dur)
            
        # Argmax
        r_preds = torch.argmax(r_logits, dim=-1).cpu().numpy()[0]
        s_preds = torch.argmax(s_logits, dim=-1).cpu().numpy()[0]
        st_preds = torch.argmax(st_logits, dim=-1).cpu().numpy()[0]
        t_preds = torch.argmax(t_logits, dim=-1).cpu().numpy()[0]
        
        # Collect valid predictions
        valid_len = end - start
        for i in range(valid_len):
            predictions.append({
                'pitch_midi': int(input_events[start+i]['pitch']),
                'onset': onsets[start+i],
                'duration': float(durations[start+i]),
                'grid_label': int(r_preds[i]),
                'spelling_label': int(s_preds[i]),
                'staff_label': int(st_preds[i]),
                'tuplet_label': int(t_preds[i])
            })
            
    return predictions

def main():
    parser = argparse.ArgumentParser(description="ScoreBERT: MIDI to MusicXML Converter")
    parser.add_argument("--input", "-i", required=True, help="Path to input MIDI file")
    parser.add_argument("--output", "-o", required=True, help="Path to output MusicXML file")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    print(f"Processing {args.input}...")
    
    # 1. Load Data
    processor = ScoreProcessor()
    input_events = processor.process_midi_file(args.input)
    if not input_events:
        print("Failed to process MIDI file.")
        return
        
    print(f"Extracted {len(input_events)} events.")
    
    # 2. Load Model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(e)
        print("Please run training first.")
        return

    # 3. Predict
    labeled_events = predict(model, input_events)
    
    # 4. Reconstruct
    print("Reconstructing score...")
    reconstructor = ScoreReconstructor()
    score = reconstructor.reconstruct(labeled_events)
    
    # 5. Save
    score.write('musicxml', args.output)
    print(f"Successfully saved to {args.output}")

if __name__ == "__main__":
    main()
