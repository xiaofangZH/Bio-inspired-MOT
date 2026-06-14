import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCropper(nn.Module):
    """
    Crops high-resolution regions of interest and extracts local features.

    Part of the Foveal Focus stage in hierarchical encoding.
    """

    def __init__(self, hidden_dim=256, crop_size=224, num_patches_per_roi=49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.crop_size = crop_size
        self.num_patches_per_roi = num_patches_per_roi

    def forward(self, images, roi_indices, backbone):
        """
        Extract features from cropped ROI regions.

        Args:
            images: [B, C, H, W] original images
            roi_indices: [B, K] token indices indicating ROI centers
            backbone: DINOv3Wrapper for feature extraction

        Returns:
            local_feats: [B, K*N_patches, D] concatenated local features
        """
        B, C, H, W = images.shape

        # Simple implementation: for each ROI index, crop and process
        local_feats_list = []

        for b in range(B):
            batch_local_feats = []

            for roi_idx in roi_indices[b]:
                roi_idx = roi_idx.item()

                # Map token index to spatial location
                # Assuming tokens are in grid order
                num_tokens_h = int((H / 16) ** 0.5 * 2)  # Approximate
                num_tokens_w = int((W / 16) ** 0.5 * 2)

                if num_tokens_h == 0 or num_tokens_w == 0:
                    # Fallback: use center crop
                    y_center = H // 2
                    x_center = W // 2
                else:
                    y_token = (roi_idx // num_tokens_w)
                    x_token = roi_idx % num_tokens_w
                    y_center = int((y_token + 0.5) * H / num_tokens_h)
                    x_center = int((x_token + 0.5) * W / num_tokens_w)

                # Crop region around center
                crop_h = min(self.crop_size, H)
                crop_w = min(self.crop_size, W)

                y1 = max(0, y_center - crop_h // 2)
                y2 = min(H, y1 + crop_h)
                x1 = max(0, x_center - crop_w // 2)
                x2 = min(W, x1 + crop_w)

                # Handle edge cases
                if x2 - x1 < crop_w:
                    if x1 == 0:
                        x2 = min(W, x1 + crop_w)
                    else:
                        x1 = max(0, x2 - crop_w)

                if y2 - y1 < crop_h:
                    if y1 == 0:
                        y2 = min(H, y1 + crop_h)
                    else:
                        y1 = max(0, y2 - crop_h)

                # Extract crop
                crop = images[b:b+1, :, y1:y2, x1:x2]

                # Resize to standard size
                if crop.shape[-2:] != (self.crop_size, self.crop_size):
                    crop = F.interpolate(
                        crop,
                        size=(self.crop_size, self.crop_size),
                        mode='bilinear',
                        align_corners=False
                    )

                # Extract features
                feats, _ = backbone(crop)  # [1, N, D]
                batch_local_feats.append(feats[0])

            if batch_local_feats:
                batch_feats = torch.cat(batch_local_feats, dim=0)  # [K*N, D]
                local_feats_list.append(batch_feats)
            else:
                # No ROIs: return empty
                local_feats_list.append(torch.zeros(
                    (0, self.hidden_dim), device=images.device))

        # Pad to same length and stack
        max_len = max(f.shape[0]
                      for f in local_feats_list) if local_feats_list else 0
        if max_len == 0:
            return torch.zeros((B, 0, self.hidden_dim), device=images.device)

        padded = []
        for feats in local_feats_list:
            if feats.shape[0] < max_len:
                pad = torch.zeros(
                    (max_len - feats.shape[0], self.hidden_dim), device=feats.device)
                feats = torch.cat([feats, pad], dim=0)
            padded.append(feats)

        local_feats = torch.stack(padded, dim=0)  # [B, max_len, D]

        return local_feats
