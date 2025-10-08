# OpenHands Benchmarks Migration

⚠️ **Migration in Progress**: We are currently migrating the benchmarks infrastructure from [OpenHands](https://github.com/OpenHands/OpenHands/) to work with the [OpenHands Agent SDK](https://github.com/OpenHands/agent-sdk).

## Current Status

This repository contains benchmark evaluation tools that are being migrated to use the OpenHands Agent SDK directly, removing the middleware runtime dependencies.

### What's Working

- ✅ **SWE-Bench Local Evaluation**: Working with `run_infer.sh` script
- ✅ **Direct SDK Integration**: No more runtime middleware dependencies
- ✅ **Local Docker Mode**: Full local evaluation support

### What's Not Working (Yet)

- ❌ **SWE-Gym**: Migration pending
- ❌ **SWT-Bench**: Migration pending  
- ❌ **Remote Runtime**: Migration pending
- ❌ **Interactive Modes**: Migration pending

## Migration Details

We are transitioning from:
- **Before**: `run_infer.py` → `utils/shared.py` → OpenHands Runtime → Agent SDK
- **After**: `run_infer.py` → Agent SDK (direct integration)

This simplifies the architecture and removes the middleware layer, making the benchmarks more maintainable and easier to understand.

## Quick Start

For SWE-Bench evaluation (the only currently working benchmark):

```bash
# Navigate to SWE-Bench directory
cd swe_bench

# Run evaluation with 10 instances
./scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 10
```

For detailed instructions, see [swe_bench/README.md](./swe_bench/README.md).

## Repository Structure

```
benchmarks/
├── swe_bench/          # SWE-Bench evaluation (✅ Working)
│   ├── run_infer.py    # Direct SDK integration
│   └── scripts/
│       └── run_infer.sh
├── utils/              # Legacy middleware (being phased out)
└── README.md           # This file
```

## Contributing

During this migration period, please:
1. Only use the SWE-Bench evaluation functionality
2. Report any issues with the current working implementation
3. Avoid using non-working features until migration is complete

## Links

- **Original OpenHands**: https://github.com/OpenHands/OpenHands/
- **Agent SDK**: https://github.com/OpenHands/agent-sdk
- **SWE-Bench**: https://www.swebench.com/