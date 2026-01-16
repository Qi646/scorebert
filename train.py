import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import MusicXMLDataset
from score_bert import ScoreBERT
from typing import Dict, Any

# Configuration
BATCH_SIZE = 32 # Adjust based on VRAM
LEARNING_RATE = 1e-4
EPOCHS = 10
DATA_PATH = "data/processed_dataset.pt"
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Load Data
    full_dataset = MusicXMLDataset(DATA_PATH)
    
    # Split Train/Val
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {train_size} samples, Validating on {val_size} samples.")
    print(f"Device: {DEVICE}")

    # 2. Model
    model = ScoreBERT().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Loss Functions
    # ignore_index=-100 matches the padding value we set in Dataset
    criterion_rhythm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_spelling = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_staff = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_tuplet = nn.CrossEntropyLoss(ignore_index=-100)
    
    weights = {
        'rhythm': 1.0,
        'staff': 0.8,
        'spelling': 0.5,
        'tuplet': 0.5
    }

    best_val_loss = float('inf')

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in loop:
            # Move data to device
            pitch = batch['input_pitch'].to(DEVICE)
            duration = batch['input_duration'].to(DEVICE)
            mask = batch['padding_mask'].to(DEVICE)
            
            lbl_grid = batch['label_grid'].to(DEVICE)
            lbl_spelling = batch['label_spelling'].to(DEVICE)
            lbl_staff = batch['label_staff'].to(DEVICE)
            lbl_tuplet = batch['label_tuplet'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            # Note: src_key_padding_mask expects True for padding
            r_out, s_out, st_out, t_out = model(pitch, duration, src_key_padding_mask=mask)
            
            # Flatten outputs for Loss: (Batch * Seq, Classes)
            loss_r = criterion_rhythm(r_out.view(-1, 48), lbl_grid.view(-1))
            loss_s = criterion_spelling(s_out.view(-1, 3), lbl_spelling.view(-1))
            loss_st = criterion_staff(st_out.view(-1, 2), lbl_staff.view(-1))
            loss_t = criterion_tuplet(t_out.view(-1, 4), lbl_tuplet.view(-1))
            
            # Weighted Sum
            total_loss = (loss_r * weights['rhythm'] + 
                          loss_s * weights['spelling'] + 
                          loss_st * weights['staff'] + 
                          loss_t * weights['tuplet'])
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pitch = batch['input_pitch'].to(DEVICE)
                duration = batch['input_duration'].to(DEVICE)
                mask = batch['padding_mask'].to(DEVICE)
                
                lbl_grid = batch['label_grid'].to(DEVICE)
                lbl_spelling = batch['label_spelling'].to(DEVICE)
                lbl_staff = batch['label_staff'].to(DEVICE)
                lbl_tuplet = batch['label_tuplet'].to(DEVICE)
                
                r_out, s_out, st_out, t_out = model(pitch, duration, src_key_padding_mask=mask)
                
                loss_r = criterion_rhythm(r_out.view(-1, 48), lbl_grid.view(-1))
                loss_s = criterion_spelling(s_out.view(-1, 3), lbl_spelling.view(-1))
                loss_st = criterion_staff(st_out.view(-1, 2), lbl_staff.view(-1))
                loss_t = criterion_tuplet(t_out.view(-1, 4), lbl_tuplet.view(-1))
                
                total_loss = (loss_r * weights['rhythm'] + 
                              loss_s * weights['spelling'] + 
                              loss_st * weights['staff'] + 
                              loss_t * weights['tuplet'])
                
                val_loss += total_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Results: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_DIR, "scorebert_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    train()
