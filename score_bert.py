import math
import torch
import torch.nn as nn
from typing import Tuple

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects sequence order information into the embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to prevent it from being a learnable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, d_model)
            
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model) with PE added.
        """
        # Slice the PE tensor to matching sequence length
        return x + self.pe[:, :x.size(1), :]

class ScoreBERT(nn.Module):
    """
    Encoder-Only Transformer for Music Transcription (MIDI -> MusicXML Labels).
    
    Architecture:
        - Pitch Embedding (Learnable)
        - Duration Embedding (Linear Projection)
        - Positional Encoding
        - Transformer Encoder
        - Multi-Task Output Heads
    """
    
    def __init__(
        self, 
        d_model: int = 256, 
        nhead: int = 4, 
        num_layers: int = 6, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 128 MIDI pitches + 1 Padding token
        self.pitch_embedding = nn.Embedding(129, d_model)
        
        # Duration is continuous, so we project it to d_model
        # To avoid feature scale issues, we have a small net for duration
        self.duration_embedder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # We will now CONCATENATE embeddings and project down
        # This is more robust than simple addition.
        self.embedding_combiner = nn.Linear(d_model * 2, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Multi-Task Heads ---
        self.head_rhythm = nn.Linear(d_model, 48)
        self.head_spelling = nn.Linear(d_model, 3)
        self.head_staff = nn.Linear(d_model, 2)
        self.head_tuplet = nn.Linear(d_model, 4)
        
        self.d_model = d_model

    def forward(self, pitch: torch.Tensor, duration: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        """
        # 1. Embeddings
        pitch_emb = self.pitch_embedding(pitch)
        
        # Normalize and embed duration
        norm_duration = torch.log1p(duration).unsqueeze(-1)
        dur_emb = self.duration_embedder(norm_duration)
        
        # Concatenate and project
        combined_emb = torch.cat([pitch_emb, dur_emb], dim=-1)
        x = self.embedding_combiner(combined_emb)
        
        # 2. Positional Encoding & Normalization
        x = self.pos_encoder(x)
        x = self.layer_norm(x) # LayerNorm is crucial for stability
        x = self.dropout(x)
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 4. Heads
        rhythm_out = self.head_rhythm(x)
        spelling_out = self.head_spelling(x)
        staff_out = self.head_staff(x)
        tuplet_out = self.head_tuplet(x)
        
        return rhythm_out, spelling_out, staff_out, tuplet_out

if __name__ == "__main__":
    # Verification Step (Phase 3.4)
    print("Verifying ScoreBERT architecture...")
    model = ScoreBERT()
    
    # Dummy Input (Batch=2, Seq=100)
    dummy_pitch = torch.randint(0, 128, (2, 100))
    dummy_duration = torch.rand(2, 100)
    
    r, s, st, t = model(dummy_pitch, dummy_duration)
    
    print(f"Rhythm Output Shape: {r.shape} (Expected: [2, 100, 48])")
    print(f"Spelling Output Shape: {s.shape} (Expected: [2, 100, 3])")
    print(f"Staff Output Shape: {st.shape} (Expected: [2, 100, 2])")
    print(f"Tuplet Output Shape: {t.shape} (Expected: [2, 100, 4])")
    
    if (r.shape == (2, 100, 48) and 
        s.shape == (2, 100, 3) and 
        st.shape == (2, 100, 2) and 
        t.shape == (2, 100, 4)):
        print("Verification Successful!")
    else:
        print("Verification Failed!")
