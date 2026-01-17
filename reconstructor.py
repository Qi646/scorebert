import music21
import torch
import numpy as np
from typing import List, Dict, Any, Optional

class ScoreReconstructor:
    """
    Translates model predictions (Rhythm, Spelling, Staff, Tuplet) 
    back into a valid music21 Score.
    """
    
    TICKS_PER_BEAT: int = 48
    
    def __init__(self):
        """Initialize the reconstructor."""
        pass

    def _get_pitch(self, midi_val: int, spelling_label: int) -> music21.pitch.Pitch:
        """
        Converts MIDI value and spelling label to a music21 Pitch.
        0: Flat, 1: Natural, 2: Sharp
        """
        p = music21.pitch.Pitch(midi_val)
        
        # music21 pitch class mapping
        # We need to force the accidental based on the label
        # This handles cases like 61 -> C# or Db
        
        if spelling_label == 0: # Flat
            # Get the flat version of the pitch
            # E.g. 61 -> Db
            p.accidental = music21.pitch.Accidental('flat')
        elif spelling_label == 2: # Sharp
            # E.g. 61 -> C#
            p.accidental = music21.pitch.Accidental('sharp')
        else: # Natural
            # Note: For pitches that are naturally sharp/flat (like F#), 
            # music21 handles defaults, but our label 1 means "Natural relative to scale".
            # For simplicity, we ensure no accidental if label is 1
            p.accidental = None
            
        return p

    def reconstruct(self, note_events: List[Dict[str, Any]]) -> music21.stream.Score:
        """
        Reconstructs a music21 Score from a list of note dictionaries.
        
        Each note dict should contain:
        - pitch_midi (int)
        - onset (original float, used to find beat number)
        - grid_label (int 0-47)
        - spelling_label (int 0-2)
        - staff_label (int 0-1)
        - tuplet_label (int 0-3)
        """
        score = music21.stream.Score()
        
        # Create two parts: Treble and Bass
        part_treble = music21.stream.Part(id='Treble')
        part_bass = music21.stream.Part(id='Bass')
        
        # We'll use a Voice for each to handle polyphony easily during reconstruction
        # but music21 can handle overlapping notes in a Part by automatically
        # creating voices or using offsets.
        
        for event in note_events:
            # 1. Quantize Onset
            # Current Beat = floor of the noisy/original onset
            # We assume the model is given enough context to know which beat it's in,
            # or we derive it from the running sequence of IOIs.
            # For now, we take the floor of the onset as the Beat Number.
            beat_number = int(np.floor(event['onset']))
            grid_pos = event['grid_label'] / self.TICKS_PER_BEAT
            quantized_onset = beat_number + grid_pos
            
            # 2. Spelling
            p = self._get_pitch(int(event['pitch_midi']), event['spelling_label'])
            
            # 3. Note Object
            n = music21.note.Note(p)
            n.offset = quantized_onset
            
            # Use original duration but ideally we'd quantize this too.
            # Phase 5 focus is mostly on onset and structure.
            n.duration.quarterLength = float(event['duration'])
            
            # 4. Tuplet Handling (Phase 5.1)
            # 1: Triplet (3:2)
            if event['tuplet_label'] == 1:
                t = music21.duration.Tuplet(3, 2, 'quarter')
                n.duration.appendTuplet(t)
            elif event['tuplet_label'] == 2:
                t = music21.duration.Tuplet(5, 4, 'quarter')
                n.duration.appendTuplet(t)
            elif event['tuplet_label'] == 3:
                t = music21.duration.Tuplet(7, 8, 'quarter')
                n.duration.appendTuplet(t)

            # 5. Assign to Part
            if event['staff_label'] == 0:
                part_treble.insert(quantized_onset, n)
            else:
                part_bass.insert(quantized_onset, n)

        # 6. Post-processing
        # Automatic grouping and cleaning
        for p in [part_treble, part_bass]:
            try:
                p.makeMeasures(inPlace=True) # Must create measures first
                p.makeBeams(inPlace=True)
            except Exception as e:
                print(f"Warning: Automatic beaming failed for part {p.id}. Saving without beams. Error: {e}")
            score.insert(0, p)
            
        return score

if __name__ == "__main__":
    # Test reconstruction using Ground Truth from a real file
    from data_processor import ScoreProcessor
    import glob
    import os
    
    files = glob.glob("mxl/**/*.mxl", recursive=True)
    if not files:
        print("No files found.")
    else:
        proc = ScoreProcessor()
        recon = ScoreReconstructor()
        
        test_file = files[0]
        print(f"Testing reconstruction on {test_file}...")
        
        _, labels = proc.process_file(test_file)
        if labels:
            # Reconstruct using the labels (Perfect Prediction)
            out_score = recon.reconstruct(labels)
            out_path = "reconstruction_test.musicxml"
            out_score.write('musicxml', out_path)
            print(f"Successfully reconstructed score to {out_path}")
