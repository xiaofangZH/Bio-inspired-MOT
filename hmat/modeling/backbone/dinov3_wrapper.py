import torch
import torch.nn as nn
import os
from pathlib import Path
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class DINOv3Wrapper(nn.Module):
    """
    Simplified DINOv3-ViT-L/16 Wrapper that directly loads weights
    without requiring the dinov3 package.
    """

    def __init__(self, output_dim=256, weights_path=None, use_checkpoint=True, max_blocks=None):
        super().__init__()
        self.output_dim = output_dim
        self.use_checkpoint = use_checkpoint
        # 推理时仅用前 max_blocks 层以提升速度，None 表示用全部层
        self.max_blocks = max_blocks
        self.weights_path = self._resolve_weights_path(weights_path)

        # Load DINOv3 weights directly
        self.vit = self._load_dinov3_weights(self.weights_path)

        # Infer vit type for forwarding
        if hasattr(self.vit, 'forward_features'):
            # timm models have forward_features but may also have get_intermediate_layers
            # Check if the get_intermediate_layers signature supports return_class_token
            self._vit_type = 'timm'
            if hasattr(self.vit, 'get_intermediate_layers'):
                import inspect
                try:
                    sig = inspect.signature(self.vit.get_intermediate_layers)
                    if 'return_class_token' in sig.parameters:
                        self._vit_type = 'dinov3'
                except (ValueError, TypeError):
                    pass
        elif hasattr(self.vit, 'get_intermediate_layers') or hasattr(self.vit, 'prepare_tokens_with_masks'):
            self._vit_type = 'dinov3'
        elif hasattr(self.vit, 'prepare_tokens'):
            self._vit_type = 'dinov3'
        else:
            self._vit_type = 'placeholder'

        # Infer output dimension from actual loaded model (vitb=768, vitl=1024, ...)
        self.vit_dim = self._infer_vit_dim(self.vit)

        # Projection layer to match hidden_dim
        self.proj = nn.Linear(self.vit_dim, output_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)

        # 预创建通道投影层 (在 _tokens_from_image 中使用，避免每次 forward 创建 nn.Linear)
        self._channel_proj = None  # lazy init, created on first use

        # Multi-scale: per-level projections for fusing intermediate layers
        self._level_projs = None   # lazy init

        # Expose patch_size for downstream modules (ViT-B/16 → 16, ViT-L/16 → 16)
        self.patch_size = getattr(self.vit, 'patch_size', 16)

        # Pre-cache effective blocks list (avoid slicing on every forward)
        self._effective_blocks = None

        # Store layer-wise freeze state for progressive unfreezing
        # (no-op: parameters initialized as trainable by default)

    def _resolve_weights_path(self, weights_path):
        """Resolve user-provided or default local DINOv3 checkpoint path."""
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[3]

        if weights_path:
            candidate = Path(str(weights_path)).expanduser()
            search_candidates = []
            if candidate.is_absolute():
                search_candidates.append(candidate)
            else:
                # Try both cwd-relative and project-root-relative.
                search_candidates.append((Path.cwd() / candidate).resolve())
                search_candidates.append((project_root / candidate).resolve())

            for p in search_candidates:
                if p.exists():
                    print(f"Using DINOv3 weights: {p}")
                    return str(p)

            print(f"Warning: configured DINOv3 weights not found: {weights_path}")

        # Auto-discover common local checkpoints under project root.
        default_candidates = [
            project_root / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            project_root / 'dinov3_vitl16_pretrain_lvd1689m-8aa47e7c.pth',
        ]
        for p in default_candidates:
            if p.exists():
                print(f"Auto-discovered DINOv3 weights: {p}")
                return str(p)

        return None

    def _infer_hub_model_name(self, weights_path):
        """Infer local torch.hub entrypoint from checkpoint file name."""
        if not weights_path:
            return 'dinov3_vitl16'
        name = Path(weights_path).name.lower()
        if 'vitb16' in name:
            return 'dinov3_vitb16'
        if 'vitl16' in name:
            return 'dinov3_vitl16'
        if 'vits16' in name:
            return 'dinov3_vits16'
        return 'dinov3_vitl16'

    def _infer_vit_dim(self, vit_model):
        """Infer token embedding dim from loaded backbone."""
        if hasattr(vit_model, 'embed_dim'):
            return int(vit_model.embed_dim)
        if hasattr(vit_model, 'num_features'):
            return int(vit_model.num_features)
        if hasattr(vit_model, 'pos_embed') and getattr(vit_model, 'pos_embed') is not None:
            return int(vit_model.pos_embed.shape[-1])
        if hasattr(vit_model, 'norm') and hasattr(vit_model.norm, 'normalized_shape'):
            shape = vit_model.norm.normalized_shape
            if shape:
                return int(shape[-1])
        print("Warning: could not infer vit_dim from model; fallback to 1024")
        return 1024

    def _load_dinov3_weights(self, weights_path):
        """
        Load DINOv3 weights via local torch.hub (uses `hubconf.py` at repo root).
        Falls back to timm if local hub loading fails.
        """
        print("Loading DINOv3-ViT-L/16 backbone via local torch.hub...")

        # Prefer the bundled dinov3 hub under hmat/dinov3 if present
        this_file = Path(__file__).resolve()
        repo_candidates = [this_file.parents[2] / 'dinov3' / 'dinov3', this_file.parents[2] / 'dinov3', this_file.parents[3]]
        hub_model_name = self._infer_hub_model_name(weights_path)

        for repo_dir in repo_candidates:
            repo_dir = str(repo_dir)
            if os.path.exists(repo_dir) and os.path.exists(os.path.join(repo_dir, 'hubconf.py')):
                try:
                    if weights_path and os.path.exists(weights_path):
                        model = torch.hub.load(repo_dir, hub_model_name, source='local', weights=weights_path)
                        print(f"✓ Loaded {hub_model_name} from local checkpoint: {weights_path}")
                    else:
                        # Avoid network download in offline env when no explicit local weights were found.
                        model = torch.hub.load(repo_dir, hub_model_name, source='local', pretrained=False)
                        print(f"Warning: no local DINOv3 weights found, initialized {hub_model_name} with pretrained=False")
                    print(f"✓ Loaded DINOv3 from local hubconf at: {repo_dir}")
                    return model
                except Exception as e:
                    print(f"Local hub load failed at {repo_dir}: {e}")
                    # try next candidate
                    continue

        print("Falling back to timm or placeholder ViT...")

        try:
            import timm
            # Use timm ViT as fallback
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            print("✓ Loaded ViT-B/16 from timm as fallback")

            if weights_path and os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded additional weights into timm model")
                except Exception as e2:
                    print(f"Warning: could not load checkpoint into timm model: {e2}")

            return model
        except Exception:
            print("timm not available; creating lightweight placeholder ViT")

            class SimpleViT(nn.Module):
                def __init__(self, embed_dim=1024, num_patches=576):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.num_patches = num_patches
                    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
                    self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
                    self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=16, dim_feedforward=4096, batch_first=True)
                    self.norm = nn.LayerNorm(embed_dim)

                def forward(self, x):
                    B = x.shape[0]
                    cls = self.cls_token.expand(B, -1, -1)
                    x = torch.cat([cls, x], dim=1)
                    x = x + self.pos_embed[:, :x.shape[1]]
                    x = self.transformer(x)
                    x = self.norm(x)
                    return x

            model = SimpleViT()
            if weights_path and os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    if 'model' in checkpoint:
                        checkpoint = checkpoint['model']
                    model.load_state_dict(checkpoint, strict=False)
                    print("✓ Loaded weights into placeholder ViT (partial)")
                except Exception as e3:
                    print(f"Warning: could not load checkpoint into placeholder: {e3}")

            return model

    def _get_level_projs(self, num_levels, device):
        """Get or create per-level projection layers for multi-scale fusion."""
        if self._level_projs is None:
            self._level_projs = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.vit_dim),
                    nn.Linear(self.vit_dim, self.output_dim),
                ).to(device)
                for _ in range(num_levels)
            ])
        return self._level_projs

    def get_multiscale_features(self, x, num_levels=4):
        """
        提取 DINOv3 多层特征并融合。

        从 ViT 的不同深度提取 features:
        - Level 0: early  (~25% depth)
        - Level 1: mid    (~50% depth)
        - Level 2: late   (~75% depth)
        - Level 3: final  (last layer)

        每层通过独立的 LayerNorm + Linear 投影到 output_dim，然后平均融合。

        Returns:
            features: [B, N, output_dim]
            mask:     [B, N]
        """
        B, C, H, W = x.shape

        if self._vit_type == 'dinov3' and hasattr(self.vit, 'get_intermediate_layers'):
            n_blocks = len(self._get_blocks())
            level_indices = [
                max(0, n_blocks // 4),
                max(0, n_blocks // 2),
                max(0, 3 * n_blocks // 4),
                n_blocks - 1,
            ]
            level_indices = sorted(set(level_indices))
            n = min(num_levels, len(level_indices))

            outputs = self.vit.get_intermediate_layers(
                x, n=level_indices, reshape=False,
                return_class_token=False, norm=True
            )

        elif self._vit_type == 'dinov3' and hasattr(self.vit, 'prepare_tokens_with_masks'):
            tokens_result = self.vit.prepare_tokens_with_masks(x, masks=None)
            if isinstance(tokens_result, tuple):
                tokens, hw = tokens_result
            else:
                tokens, hw = tokens_result, None

            blocks = self._get_blocks()
            n_blocks = len(blocks)
            level_indices = sorted(set([
                max(0, n_blocks // 4),
                max(0, n_blocks // 2),
                max(0, 3 * n_blocks // 4),
                n_blocks - 1,
            ]))[:num_levels]

            rope_sincos = None
            if hw is not None and hasattr(self.vit, 'rope_embed'):
                rope_sincos = self.vit.rope_embed(*hw)

            outputs = []
            for i, blk in enumerate(blocks):
                if self.use_checkpoint and self.training:
                    if rope_sincos is not None:
                        tokens = torch_checkpoint(blk, tokens, rope_sincos=rope_sincos, use_reentrant=False)
                    else:
                        tokens = torch_checkpoint(blk, tokens, use_reentrant=False)
                else:
                    if rope_sincos is not None:
                        tokens = blk(tokens, rope_sincos=rope_sincos)
                    else:
                        tokens = blk(tokens)
                if i in level_indices and len(outputs) < num_levels:
                    n_prefix = getattr(self.vit, 'num_prefix_tokens', 1)
                    level_feat = self.vit.norm(tokens[:, n_prefix:])
                    outputs.append(level_feat)

        elif self._vit_type == 'dinov3':
            tokens = self.vit.prepare_tokens(x)
            blocks = self._get_blocks()
            n_blocks = len(blocks)
            level_indices = sorted(set([
                max(0, n_blocks // 4),
                max(0, n_blocks // 2),
                max(0, 3 * n_blocks // 4),
                n_blocks - 1,
            ]))[:num_levels]

            outputs = []
            for i, blk in enumerate(blocks):
                if self.use_checkpoint and self.training:
                    tokens = torch_checkpoint(blk, tokens, use_reentrant=False)
                else:
                    tokens = blk(tokens)
                if i in level_indices and len(outputs) < num_levels:
                    outputs.append(self.vit.norm(tokens))

        else:
            vit_out = self._tokens_from_image(x)
            features = self.proj(vit_out)
            features = self.norm(features)
            N = features.shape[1]
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            return features, mask

        num_actual = len(outputs)
        level_projs = self._get_level_projs(num_actual, x.device)

        fused = None
        for i, feat in enumerate(outputs):
            projected = level_projs[i](feat)
            if fused is None:
                fused = projected
            else:
                fused = fused + projected

        fused = fused / num_actual
        fused = self.norm(fused)

        N = fused.shape[1]
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        return fused, mask

    # ===== 渐进式冻结/解冻 (三步训练用) =====

    def freeze_all_backbone(self):
        """冻结 Vit backbone + proj + norm 的所有参数 (Phase 1 前5 epoch)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n=2):
        """解冻 Vit backbone 的最后 N 个 transformer blocks (Phase 1 后10 epoch).
        
        Args:
            n: 解冻的 block 数量 (默认 2)
        """
        blocks = self._get_blocks()
        num_blocks = len(blocks)
        n = min(n, num_blocks)
        
        # 先确保所有块都冻结
        for blk in blocks:
            for param in blk.parameters():
                param.requires_grad = False
        
        # 解冻最后 N 个 block
        for blk in blocks[-n:]:
            for param in blk.parameters():
                param.requires_grad = True
        
        # proj 和 norm 解冻（用于特征投影）
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True

    def unfreeze_all_backbone(self):
        """全面解冻 backbone + proj + norm (Phase 2+)."""
        for param in self.vit.parameters():
            param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True

    def get_frozen_status(self):
        """Return frozen status of backbone."""
        frozen_params = sum(1 for p in self.vit.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.vit.parameters())
        return {"frozen": frozen_params, "total": total_params, "ratio": frozen_params / total_params if total_params > 0 else 0}

    def _get_blocks(self):
        """Return cached effective blocks (truncated by max_blocks)."""
        if self._effective_blocks is None:
            blocks = getattr(self.vit, 'blocks', [])
            if self.max_blocks is not None and self.max_blocks < len(blocks):
                self._effective_blocks = list(blocks)[:self.max_blocks]
            else:
                self._effective_blocks = blocks
        return self._effective_blocks

    def forward(self, x):
        """Forward pass through backbone. Handles different vit implementations.

        Returns features as [B, N, D] and mask [B, N].
        """
        B, C, H, W = x.shape

        if self._vit_type == 'dinov3':
            try:
                if hasattr(self.vit, 'get_intermediate_layers'):
                    outputs = self.vit.get_intermediate_layers(
                        x, n=1, reshape=False,
                        return_class_token=False, norm=True
                    )
                    vit_out = outputs[0]
                elif hasattr(self.vit, 'prepare_tokens_with_masks'):
                    tokens_result = self.vit.prepare_tokens_with_masks(x, masks=None)
                    if isinstance(tokens_result, tuple):
                        tokens, hw = tokens_result
                    else:
                        tokens, hw = tokens_result, None
                    blocks = self._get_blocks()
                    rope_sincos = None
                    if hw is not None and hasattr(self.vit, 'rope_embed'):
                        rope_sincos = self.vit.rope_embed(*hw)
                    for blk in blocks:
                        if self.use_checkpoint and self.training:
                            if rope_sincos is not None:
                                tokens = torch_checkpoint(blk, tokens, rope_sincos=rope_sincos, use_reentrant=False)
                            else:
                                tokens = torch_checkpoint(blk, tokens, use_reentrant=False)
                        else:
                            if rope_sincos is not None:
                                tokens = blk(tokens, rope_sincos=rope_sincos)
                            else:
                                tokens = blk(tokens)
                    n_prefix = getattr(self.vit, 'num_prefix_tokens', 1)
                    vit_out = self.vit.norm(tokens[:, n_prefix:])
                else:
                    tokens = self.vit.prepare_tokens(x)
                    blocks = self._get_blocks()
                    for blk in blocks:
                        if self.use_checkpoint and self.training:
                            tokens = torch_checkpoint(blk, tokens, use_reentrant=False)
                        else:
                            tokens = blk(tokens)
                    vit_out = self.vit.norm(tokens)
            except Exception:
                vit_out = self._tokens_from_image(x)

        elif self._vit_type == 'timm':
            # timm ViT accepts images; use forward_features if available
            try:
                vit_out = self.vit.forward_features(x)
                # Ensure seq dim exists: if returned [B, D] (cls-only), make [B,1,D]
                if vit_out.ndim == 2:
                    vit_out = vit_out.unsqueeze(1)
            except Exception:
                vit_out = self._tokens_from_image(x)

        else:
            vit_out = self._tokens_from_image(x)

        # vit_out should be [B, N, D]
        if vit_out.ndim == 2:
            vit_out = vit_out.unsqueeze(1)

        # Project to output dimension
        features = self.proj(vit_out)
        features = self.norm(features)

        N = features.shape[1]
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        return features, mask

    def _tokens_from_image(self, x, patch_size=16):
        """Create simple patch tokens via adaptive pooling."""
        B, C, H, W = x.shape
        num_patches_h = max(1, H // patch_size)
        num_patches_w = max(1, W // patch_size)
        num_patches = num_patches_h * num_patches_w

        features = torch.nn.functional.adaptive_avg_pool2d(x, (num_patches_h, num_patches_w))
        features = features.permute(0, 2, 3, 1).contiguous().view(B, num_patches, C)

        # If channel dim != vit_dim, use pre-created projection layer (avoid per-forward allocation)
        if C != self.vit_dim:
            if self._channel_proj is None or self._channel_proj.in_features != C:
                self._channel_proj = nn.Linear(C, self.vit_dim).to(x.device)
            vit_out = self._channel_proj(features)
        else:
            vit_out = features

        return vit_out

    def _get_pos_embed(self, num_patches, dim, device):
        """Generate positional embeddings."""
        # Simple sinusoidal positional embeddings
        pos = torch.arange(num_patches, dtype=torch.float32,
                           device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) *
                             -(torch.log(torch.tensor(10000.0)) / dim))
        pos_embed = torch.zeros(num_patches, dim, device=device)
        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        if dim % 2 == 1:
            pos_embed[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pos_embed[:, 1::2] = torch.cos(pos * div_term)
        return pos_embed
