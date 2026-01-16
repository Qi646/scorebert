import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any

class MusicXMLDataset(Dataset):
    """
    PyTorch Dataset that loads pre-processed MusicXML data (tensors)
    and slices them into fixed-length windows for training.
    """
    
    def __init__(
        self, 
        data_path: str, 
        seq_len: int = 512, 
        stride: int = 384, # 75% overlap (512 * 0.75 = 384)
        pad_token_pitch: int = 128
    ):
        """
        Args:
            data_path: Path to the .pt file containing the list of processed songs.
            seq_len: The fixed sequence length for the model input.
            stride: The step size for the sliding window.
            pad_token_pitch: The token index used for padding pitch (128).
        """
        print(f"Loading dataset from {data_path}...")
        self.data: List[Dict[str, torch.Tensor]] = torch.load(data_path)
        self.seq_len = seq_len
        self.stride = stride
        self.pad_token_pitch = pad_token_pitch
        
        # Build the index map: global_idx -> (song_idx, start_offset)
        self.indices: List[Tuple[int, int]] = []
        
        for song_idx, song in enumerate(self.data):
            num_tokens = len(song['input_pitch'])
            
            # If song is shorter than seq_len, it's just one window (padded later)
            if num_tokens <= seq_len:
                self.indices.append((song_idx, 0))
            else:
                # Sliding window
                for start in range(0, num_tokens, stride):
                    # If the last window is too short, we still take it and pad it
                    # (Unless it's completely redundant, but stride logic handles that)
                    if start < num_tokens:
                        self.indices.append((song_idx, start))
                        
        print(f"Dataset loaded: {len(self.data)} songs turned into {len(self.indices)} training windows.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        song_idx, start = self.indices[idx]
        song_data = self.data[song_idx]
        
        end = start + self.seq_len
        actual_len = len(song_data['input_pitch'])
        
        # Slice the tensors
        # We need to handle the case where end > actual_len (Padding needed)
        
        # Helper to slice and pad
        def slice_and_pad(tensor: torch.Tensor, pad_value: float) -> torch.Tensor:
            sliced = tensor[start:min(end, actual_len)]
            if len(sliced) < self.seq_len:
                pad_size = self.seq_len - len(sliced)
                # Create padding tensor
                padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                return torch.cat([sliced, padding])
            return sliced

        # Create padding mask (True where it is padding)
        # 0 for real data, 1 for padding
        sliced_len = min(end, actual_len) - start
        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        if sliced_len < self.seq_len:
            padding_mask[sliced_len:] = True

        batch = {
            'input_pitch': slice_and_pad(song_data['input_pitch'], self.pad_token_pitch),
            'input_duration': slice_and_pad(song_data['input_duration'], 0.0), # Pad duration with 0
            # 'input_onset': slice_and_pad(song_data['input_onset'], 0.0), # Not strictly needed for model
            
            'label_grid': slice_and_pad(song_data['label_grid'], -100), # -100 is PyTorch Ignore Index
            'label_spelling': slice_and_pad(song_data['label_spelling'], -100),
            'label_staff': slice_and_pad(song_data['label_staff'], -100),
            'label_tuplet': slice_and_pad(song_data['label_tuplet'], -100),
            
            'padding_mask': padding_mask
        }
        
        return batch
