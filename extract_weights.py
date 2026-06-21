import torch

ckpt = torch.load(
    '/root/MOT项目/HAMT/results/hamt_full_20260620_014619/checkpoints/epoch_010.pth',
    map_location='cpu', weights_only=False
)
data = {
    'model_state_dict': ckpt['model_state_dict'],
    'epoch': ckpt.get('epoch', 0),
    'global_step': ckpt.get('global_step', 0),
    'stage_name': ckpt.get('stage_name', 'stage2'),
}
torch.save(data, '/root/MOT项目/HAMT/best_stage2_init.pth')
print(f"Saved. epoch={data['epoch']}, step={data['global_step']}, stage={data['stage_name']}")
print(f"Keys: {len(data['model_state_dict'])} params")
