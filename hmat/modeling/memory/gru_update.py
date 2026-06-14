import torch
import torch.nn as nn


class GRUUpdate(nn.Module):
    """
    GRU-based recurrent memory update mechanism.

    Updates object features over time using Gated Recurrent Units,
    smoothing out single-frame noise and maintaining long-term object identity.

    Based on the formulation in the paper:
    z_t = σ(W_z[o_t, m_{t-1}])  -- reset gate
    r_t = σ(W_r[o_t, m_{t-1}])  -- update gate
    m~_t = tanh(W_h[o_t, r_t ⊙ m_{t-1}]) -- candidate hidden state
    m_t = (1 - z_t) ⊙ m_{t-1} + z_t ⊙ m~_t -- new memory
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # GRU cell for memory update
        self.gru_cell = nn.GRUCell(d_model, d_model)

        # Alternative: Manual GRU implementation for better interpretability
        self.reset_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.candidate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )

        self.use_manual_gru = False  # Set to True to use manual implementation

    def forward(self, old_memory, new_features):
        """
        Update memory embeddings with new observations.

        Args:
            old_memory: [B, N_objects, D] - previous memory embeddings
            new_features: [B, N_objects, D] - current frame features (from decoder)

        Returns:
            updated_memory: [B, N_objects, D] - updated memory for next frame
        """
        if self.use_manual_gru:
            return self._manual_gru(old_memory, new_features)
        else:
            return self._pytorch_gru(old_memory, new_features)

    def _pytorch_gru(self, old_memory, new_features):
        """Use PyTorch's built-in GRU cell."""
        B, N, D = old_memory.shape

        # Reshape for GRU cell: [B*N, D]
        old_mem_flat = old_memory.reshape(B * N, D)
        new_feat_flat = new_features.reshape(B * N, D)

        # Update
        updated = self.gru_cell(new_feat_flat, old_mem_flat)

        # Reshape back
        updated = updated.reshape(B, N, D)

        return updated

    def _manual_gru(self, old_memory, new_features):
        """Manual GRU implementation for interpretability."""
        B, N, D = old_memory.shape

        # Concatenate input and hidden state
        combined = torch.cat([new_features, old_memory], dim=-1)

        # Reset gate
        reset = self.reset_gate(combined)  # [B, N, D]

        # Update gate
        update = self.update_gate(combined)  # [B, N, D]

        # Candidate hidden state
        combined_reset = torch.cat(
            [new_features, reset * old_memory], dim=-1
        )
        candidate = self.candidate(combined_reset)  # [B, N, D]

        # Final update
        updated = (1 - update) * old_memory + update * candidate

        return updated

    def get_gate_statistics(self, old_memory, new_features):
        """
        For analysis: get statistics of GRU gates to understand update behavior.
        """
        combined = torch.cat([new_features, old_memory], dim=-1)
        reset = self.reset_gate(combined)
        update = self.update_gate(combined)

        return {
            'reset_gate_mean': reset.mean().item(),
            'update_gate_mean': update.mean().item(),
            'reset_gate_std': reset.std().item(),
            'update_gate_std': update.std().item(),
        }
