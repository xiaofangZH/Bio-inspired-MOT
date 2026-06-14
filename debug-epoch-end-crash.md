# Debug Session: epoch-end-crash

## Status: [OPEN]

## Symptom
Training process is silently killed (SIGKILL) near the end of Epoch 1 (~81800/81887 frames), after ~40K+ successful training steps. No Python exception, no CUDA error, no dmesg OOM message. Reproduced 4 times consistently at the same point.

## Hypotheses

| ID | Hypothesis | Observation Point | Falsification Condition |
|----|-----------|-------------------|------------------------|
| **A** | MemoryBank accumulates too many track embeddings over 81K+ frames, causing GPU OOM at epoch end | MemoryBank size & GPU memory before crash | If memory bank has few entries or GPU memory is stable |
| **B** | Epoch-end checkpoint save memory spike triggers OOM Kill | System/GPU memory at save point | If crash happens far from save checkpoint timing |
| **C** | Dataloader last batch has corrupted data causing silent crash | Which sequence/sequence_idx is being processed at crash | If crash happens mid-sequence, not at sequence boundary |
| **D** | Gradient accumulation flush at epoch boundary causes memory spike | Memory before/after grad flush | If accum_step % grad_accum == 0 at crash (no flush needed) |
| **E** | Python/system memory fragmentation triggers allocation failure | Python RSS/memory before crash | If Python memory usage is moderate and stable |

## Instrumentation Plan
1. Debug Server on port 7777
2. Instrument: epoch progress, memory bank size, GPU/CPU memory, checkpoint timing, sequence boundaries
3. Log to `trae-debug-log-epoch-end-crash.ndjson`
