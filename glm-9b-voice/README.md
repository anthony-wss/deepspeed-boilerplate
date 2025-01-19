# Fully finetune GLM-4-voice

```bash
python build_dataset.py
deepspeed train.py
```

## Common Issues

1. Recomputed values for the following tensors have different metadata than during the forward pass.

```
[rank0]: torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.
[rank0]: tensor at position 4:
[rank0]: saved metadata: {'shape': torch.Size([4096]), 'dtype': torch.float16, 'device': device(type='cuda', index=0)}
[rank0]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float16, 'device': device(type='cuda', index=0)}
[rank0]: tensor at position 36:
[rank0]: saved metadata: {'shape': torch.Size([4096]), 'dtype': torch.float16, 'device': device(type='cuda', index=0)}
[rank0]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float16, 'device': device(type='cuda', index=0)}
```

Solution: Disable `gradient_checkpointing`

2. Loss is 0.0

```
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.01}                                                                              
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.02}                                                                              
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.03}                                                                              
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.04}                                                                              
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.05}                                                                              
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5e-05, 'epoch': 0.06}
```

Solution: change `fp16` to `bf16`. Since GLM-4-voice is trained with bf16?
