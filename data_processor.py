import music21
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class ScoreProcessor:
    """
    Handles the extraction of ground truth labels from MusicXML
    and the generation of corrupted (noisy) MIDI-like inputs.
    
    Attributes:
        TICKS_PER_BEAT (int): Resolution of the grid for quantization (default: 48).
    """
    
    TICKS_PER_BEAT: int = 48
    
    def __init__(self):
        """
        Initialize the ScoreProcessor.
        """
        pass
        
    def process_file(self, file_path: str) -> Tuple[Optional[List[Dict[str, float]]], Optional[List[Dict[str, Any]]]]:
        """
        Main entry point for processing a single MusicXML file.
        
        Args:
            file_path: Path to the MusicXML file.
            
        Returns:
            A tuple containing (inputs, labels).
            - inputs: List of dictionaries representing the "corrupted" MIDI events.
            - labels: List of dictionaries representing the ground truth labels.
            Returns (None, None) if processing fails.
        """
        try:
            # forceSource=True ensures we parse the XML fresh and don't rely on cached pickles
            score = music21.converter.parse(file_path, format='musicxml', forceSource=True)
            labels = self.extract_labels(score)
            inputs = self.corrupt_score(labels)
            return inputs, labels
        except Exception as e:
            # In a production pipeline, we might want to log this to a file
            # print(f"Error processing {file_path}: {e}")
            return None, None

    def _get_spelling_label(self, note_obj: music21.note.Note) -> int:
        """
        Helper to determine the spelling label for a note.
        0: Flat, 1: Natural, 2: Sharp
        """
        acc = note_obj.pitch.accidental
        if acc is None:
            return 1 # Natural
        if acc.alter < 0:
            return 0 # Flat
        if acc.alter > 0:
            return 2 # Sharp
        return 1

    def _get_tuplet_label(self, tuplets: music21.duration.Tuplet) -> int:
        """
        Helper to determine the tuplet label.
        0: None, 1: Triplet, 2: Quintuplet, 3: Septuplet
        """
        if not tuplets:
            return 0
        
        t = tuplets[0]
        actual = t.numberNotesActual
        if actual == 3:
            return 1
        elif actual == 5:
            return 2
        elif actual == 7:
            return 3
        return 0

    def extract_labels(self, score: music21.stream.Score) -> List[Dict[str, Any]]:
        """
        Extracts Ground Truth labels from the score.
        
        Args:
            score: The parsed music21 Score object.
            
        Returns:
            A sorted list of dictionaries, where each dictionary represents a note event
            and its associated ground truth labels (Grid, Spelling, Staff, Tuplet).
        """
        note_events: List[Dict[str, Any]] = []
        
        # Iterate through parts to assign Staff ID
        for part_idx, part in enumerate(score.parts):
            # Flatten part to get a stream of notes/chords with absolute offsets
            flat_part = part.flatten()
            
            # Filter for notes and chords
            for element in flat_part.getElementsByClass(['Note', 'Chord']):
                
                sub_notes: List[music21.note.Note] = []
                offset: float = float(element.offset)
                duration: float = float(element.duration.quarterLength)
                tuplets = element.duration.tuplets
                
                if isinstance(element, music21.chord.Chord):
                    sub_notes = list(element.notes)
                elif isinstance(element, music21.note.Note):
                    sub_notes = [element]

                for n in sub_notes:
                    # 1. Grid Label (0-47)
                    beat_position = offset % 1.0
                    grid_label = int(round(beat_position * self.TICKS_PER_BEAT)) % self.TICKS_PER_BEAT
                    
                    # 2. Spelling Label
                    spelling_label = self._get_spelling_label(n)
                        
                    # 3. Staff Label
                    staff_label: int = min(part_idx, 1) 
                    
                    # 4. Tuplet Label
                    tuplet_label = self._get_tuplet_label(tuplets)
                            
                    note_data = {
                        'pitch_midi': int(n.pitch.midi),
                        'onset': offset,
                        'duration': duration,
                        'grid_label': grid_label,
                        'spelling_label': spelling_label,
                        'staff_label': staff_label,
                        'tuplet_label': tuplet_label,
                        'original_name': n.pitch.nameWithOctave
                    }
                    note_events.append(note_data)
                    
        # Sort by onset time, then pitch
        note_events.sort(key=lambda x: (x['onset'], x['pitch_midi']))
        return note_events

    def corrupt_score(self, ground_truth_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Applies destructive transformations to create Model Inputs.
        
        Args:
            ground_truth_data: The clean list of note events.
            
        Returns:
            A list of "corrupted" note events containing only pitch, onset, and duration.
        """
        corrupted_events: List[Dict[str, float]] = []
        
        for event in ground_truth_data:
            pitch = event['pitch_midi']
            
            # Timing Noise (Gaussian)
            noise_onset = np.random.normal(0, 0.05)
            new_onset = max(0.0, event['onset'] + noise_onset)
            
            # Duration Scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            new_duration = max(0.1, event['duration'] * scale_factor)
            
            input_event = {
                'pitch': float(pitch),
                'onset': float(new_onset),
                'duration': float(new_duration)
            }
            corrupted_events.append(input_event)
        
        # DO NOT RE-SORT. The i-th corrupted event must correspond to the i-th label.
        # The model uses Positional Encoding, not absolute onset, to learn order.
        
        return corrupted_events

if __name__ == "__main__":
    import glob
    import sys
    
    files = glob.glob("mxl/**/*.mxl", recursive=True)
    if not files:
        print("No .mxl files found to test.")
        sys.exit(0)
        
    test_file = files[0]
    print(f"Testing ScoreProcessor on {test_file}...")
    
    processor = ScoreProcessor()
    inputs, labels = processor.process_file(test_file)
    
    if inputs and labels:
        print(f"Success! Extracted {len(labels)} events.")
        print("Sample Label (First 3):")
        for i in range(min(3, len(labels))):
            print(labels[i])
            
        print("\nSample Input (First 3 - Corrupted):")
        for i in range(min(3, len(inputs))):
            print(inputs[i])
    else:
        print("Processing failed.")