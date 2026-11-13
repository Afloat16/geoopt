# Technical review of LeRobot #5352

Target: https://github.com/huggingface/lerobot/pull/5352

Reviewed commit: `a2293442b3ebb8950f1e70d4840f2f75e0f7cf4a`. Review action: COMMENT.

## Summary

Reviewed the load-key normalization, its registration before compiling the U-Net, and the two added regression tests at a2293442b3eb. I also traced the existing single-file and streamed-shard paths in PreTrainedPolicy._load_checkpoint.

The four save/load setting combinations exercise the intended cross-configuration behavior. I have one follow-up on the collision guarantee: the new test covers both spellings in one safetensors file, while the shared loader invokes the pre-hook separately for each shard. I left a concrete case on the guard below.

This is a source review; I did not rerun the test suite or validate EMA resume, PEFT, accelerator execution, or a physical robot.

## Inline comment

File: `src/lerobot/policies/diffusion/modeling_diffusion.py`. Right-side line: 217.

Could the collision regression also cover the two spellings landing in different checkpoint shards?

PreTrainedPolicy._load_checkpoint loads each file into a fresh state dictionary and calls model.load_state_dict(state, strict=False) separately (pretrained.py:337–342). From that control flow, a shard containing `diffusion.unet.<weight>` loads the original value; a later shard containing `diffusion.unet._orig_mod.<weight>` sees no `target` in its own dictionary, renames it here, and loads a second value into the same parameter. Neither load reports an unexpected key, so the final strict check can accept the checkpoint. Reversing the shards changes which value wins.

The current single-file test correctly catches its case, but does not exercise this streamed path. A small forced-sharding fixture with distinct values for the two names would protect the stated no-overwrite guarantee; alternatively, please make the single-file scope explicit. This finding is from tracing the code, not an executed reproducer.
