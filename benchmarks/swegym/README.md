# SWE-Gym Benchmark - building Docker images

This directory contains implementation for building custom agent server Docker images for SWE-Gym. The primary purpose is to use GitHub workflows for building these images fast and using them to train LLMs as SWE agents.

## Dataset

- **Source**: [Paper](https://arxiv.org/abs/2412.21139)
- **Dataset**: 
  - `SWE-Gym/SWE-Gym` - Full dataset
- **Splits**: `train`

## Usage

### Build Docker Images

You need to build Docker images for the SWE-Gym instances. Each instance requires a specific environment setup based on the repository and issue. **Note that this will consume atleast 5-6TB of disk space. Considering setting `--n-limit` to a smaller value if required**

```bash
uv run python -m benchmarks.swegym.build_images \
  --dataset SWE-Gym/SWE-Gym \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

### Running rollouts

This is not supported yet for SWE-Gym because the primary purpose of this directory is fast and smooth creation of Docker images.

