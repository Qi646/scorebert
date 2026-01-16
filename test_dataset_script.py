from dataset import MusicXMLDataset
import torch

def test_dataset():
    ds = MusicXMLDataset("data/processed_dataset.pt")
    print(f"Total windows: {len(ds)}")
    
    sample = ds[0]
    print("Sample keys:", sample.keys())
    print("Input Pitch shape:", sample['input_pitch'].shape)
    print("Padding Mask shape:", sample['padding_mask'].shape)
    
    # Check padding logic
    # Find a window that likely has padding (the last window of the first song)
    # We don't know exactly which index, but let's check a few
    print("Checking for padded batch...")
    for i in range(min(100, len(ds))):
        s = ds[i]
        if s['padding_mask'].any():
            print(f"Found padded batch at index {i}")
            print("Mask sum:", s['padding_mask'].sum().item())
            print("Last pitch:", s['input_pitch'][-1].item())
            break

if __name__ == "__main__":
    test_dataset()
