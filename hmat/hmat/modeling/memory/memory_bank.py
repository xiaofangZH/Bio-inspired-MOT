import torch
import torch.nn as nn
from .gru_update import GRUUpdate


class MemoryBank(nn.Module):
    """
    Memory Bank for tracking object features over time.

    Maintains a dynamic pool of object embeddings:
    - Stores active track embeddings
    - Updates via GRU mechanism
    - Ages and prunes long-lived idle tracks
    - Creates new tracks from high-confidence detections
    """

    def __init__(self, hidden_dim=256, max_age=30, init_history=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_age = max_age
        self.init_history = init_history  # Frames to confirm new track

        # GRU update mechanism
        self.gru = GRUUpdate(hidden_dim)

        # Track storage
        self.tracks = {}  # {track_id: {'embed': tensor, 'bbox': tensor, 'age': int, 'score': float, 'hits': int}}
        self.next_track_id = 0

        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_finished = 0

    def reset(self):
        """Reset memory bank for new video sequence."""
        self.tracks = {}
        self.next_track_id = 0

    def get_track_queries(self):
        """
        Get embeddings of currently active tracks for decoder input.

        Returns:
            track_embeds: [N_tracks, D] embeddings of active tracks
            track_ids: [N_tracks] corresponding track IDs
            track_boxes: [N_tracks, 4] bounding boxes (cx, cy, w, h) normalized
        """
        if not self.tracks:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32), [], torch.empty((0, 4), dtype=torch.float32)

        track_ids = sorted(self.tracks.keys())
        track_embeds = torch.stack([
            self.tracks[tid]['embed'] for tid in track_ids
        ])
        track_boxes = torch.stack([
            self.tracks[tid]['bbox'] for tid in track_ids
        ])

        return track_embeds, track_ids, track_boxes

    def get_memory_values(self):
        """
        Get all active track embeddings for memory cross-attention.

        Returns:
            memory: [1, N_tracks, D] or None if no tracks
        """
        track_embeds, _, _ = self.get_track_queries()

        if track_embeds.numel() == 0:
            return None

        return track_embeds.unsqueeze(0)  # [1, N_tracks, D]

    def update(self, track_embeds, track_preds, detect_embeds, detect_preds,
               matcher=None, targets=None):
        """
        Update memory bank with new observations.

        Args:
            track_embeds: [B, N_track, D] embeddings for existing tracks
            track_preds: dict with predictions for existing tracks
            detect_embeds: [B, N_detect, D] embeddings for new detections
            detect_preds: dict with predictions for new detections
            matcher: Optional matcher for associating detections to existing tracks
            targets: Optional GT annotations for supervised update
        """
        B = track_embeds.shape[0] if track_embeds.numel() > 0 else 1

        if B != 1:
            raise NotImplementedError("Only batch_size=1 supported currently")

        # 1. Update existing tracks via GRU
        if track_embeds.shape[1] > 0:
            self._update_existing_tracks(track_embeds, track_preds)

        # 2. Create new tracks from high-confidence detections
        if detect_embeds.shape[1] > 0:
            self._create_new_tracks(detect_embeds, detect_preds)

        # 3. Age and prune old tracks
        self._age_and_prune_tracks()

    def _update_existing_tracks(self, track_embeds, track_preds):
        """Update embeddings of existing tracks using GRU."""
        track_ids = sorted(self.tracks.keys())

        if len(track_ids) == 0:
            return

        # Get old embeddings
        old_embeds = torch.stack([
            self.tracks[tid]['embed'] for tid in track_ids
        ]).unsqueeze(0)  # [1, N_track, D]

        # Update via GRU
        updated_embeds = self.gru(old_embeds, track_embeds)  # [1, N_track, D]

        # Get confidences from predictions
        logits = track_preds.get(
            'pred_logits', torch.zeros((1, len(track_ids), 2)))
        # Confidence of object class
        scores = torch.softmax(logits, dim=-1)[:, :, 0]

        # Update each track (detach so next frame backward does not go through this graph)
        for idx, track_id in enumerate(track_ids):
            self.tracks[track_id]['embed'] = updated_embeds[0, idx].detach()
            self.tracks[track_id]['score'] = scores[0, idx].item()
            self.tracks[track_id]['hits'] += 1

    def _create_new_tracks(self, detect_embeds, detect_preds):
        """Create new tracks from high-confidence detections."""
        # Get confidence scores
        logits = detect_preds.get(
            'pred_logits', torch.zeros((1, detect_embeds.shape[1], 2)))
        scores = torch.softmax(logits, dim=-1)[:, :, 0]  # [1, N_detect]

        # Threshold for track creation (降低以匹配当前模型输出 ~0.29)
        conf_threshold = 0.4  # 提高阈值，与检测阈值一致，避免创建低质量tracks
        high_conf_idx = (scores[0] > conf_threshold).nonzero(as_tuple=True)[0]

        for idx in high_conf_idx:
            # Create new track (detach to avoid double backward across frames)
            self.tracks[self.next_track_id] = {
                'embed': detect_embeds[0, idx].detach(),
                'age': 0,
                'score': scores[0, idx].item(),
                'hits': 1,  # Need N hits to confirm
                'bbox': detect_preds.get('pred_boxes', torch.zeros((1, detect_embeds.shape[1], 4)))[0, idx]
            }
            self.next_track_id += 1
            self.total_tracks_created += 1

    def _age_and_prune_tracks(self):
        """Remove old tracks that haven't been matched recently."""
        tracks_to_remove = []

        for track_id, track_info in self.tracks.items():
            track_info['age'] += 1

            # Remove if too old
            if track_info['age'] > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.total_tracks_finished += 1

    def get_statistics(self):
        """Get memory bank statistics for monitoring."""
        return {
            'num_active_tracks': len(self.tracks),
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_finished': self.total_tracks_finished,
            'avg_track_age': sum(t['age'] for t in self.tracks.values()) / max(1, len(self.tracks)),
            'avg_track_score': sum(t['score'] for t in self.tracks.values()) / max(1, len(self.tracks)),
        }
