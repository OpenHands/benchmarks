# OpenHands SWT-Bench


## Prerequisites

Before running any benchmarks, you need to set up the environment see main README.md

### 1. Run SWT-Bench Evaluation
```bash
# Run evaluation with your configured LLM
uv run swtbench-infer .llm_config/sonnet-4.json --critic pass
```

### 2. Selecting Specific Instances

You can run evaluation on a specific subset of instances using the `--select` option:

1. Create a text file with one instance ID per line:

**instances.txt:**
```
django__django-11333
astropy__astropy-12345
requests__requests-5555
```

2. Run evaluation with the selection file:
```bash
python -m benchmarks.swe_bench.run_infer \
    --agent-cls CodeActAgent \
    --llm-config llm_config.toml \
    --max-iterations 30 \
    --select instances.txt \
    --eval-output-dir ./evaluation_results \
    --max-attempts 3 \
    --critic finish_with_patch
```

This will only evaluate the instances listed in the file.

## Links

- **Original OpenHands**: https://github.com/All-Hands-AI/OpenHands/
- **Agent SDK**: https://github.com/All-Hands-AI/agent-sdk
- **SWT-Bench**: https://www.swtbench.com/
