# Per-Instance Resource Factor Mappings

This directory contains JSON files that map instance IDs to their base resource factors.

## Purpose

Different benchmark instances may have varying resource requirements (e.g., memory, CPU). Some instances with large codebases or heavy dependencies require more resources to successfully complete evaluation. By specifying per-instance resource factors, we can:

1. **Avoid wasted retries**: Start resource-intensive instances with higher resources instead of failing with default resources
2. **Improve evaluation efficiency**: Reduce the number of retry attempts needed
3. **Better resource allocation**: Allocate resources based on actual instance needs

## File Naming Convention

Files should be named after the dataset they apply to, with slashes replaced by double underscores:

- Dataset: `princeton-nlp/SWE-bench_Lite` → File: `princeton-nlp__SWE-bench_Lite.json`
- Dataset: `princeton-nlp/SWE-bench_Verified` → File: `princeton-nlp__SWE-bench_Verified.json`

## File Format

Each JSON file maps instance IDs to integer resource factors (≥ 1):

```json
{
  "instance-id-1": 1,
  "instance-id-2": 2,
  "instance-id-3": 4
}
```

### Resource Factor Values

- `1`: Default resources (1x CPU/memory allocation)
- `2`: Double resources (2x CPU/memory allocation)
- `4`: Quadruple resources (4x CPU/memory allocation)
- `8`: Maximum initial resources (8x CPU/memory allocation)

**Note**: During retries, the resource factor is exponentially increased:
- First attempt: base factor (from mapping or default)
- After 1st failure: base factor × 2
- After 2nd failure: base factor × 4
- Capped at `max_resource_factor` from evaluation metadata (default: 8)

## Creating Mapping Files

To create a mapping file for a dataset:

1. Identify instances that frequently fail with resource-related errors
2. Determine appropriate base resource factors based on their needs
3. Create a JSON file with the dataset name (replace `/` with `__`)
4. Add mappings for instances that need non-default resources

Only instances requiring non-default resources need to be listed. Missing instances default to the global `base_resource_factor` from evaluation metadata (typically 1).

## Example

For instances known to be resource-intensive in SWE-bench Lite:

```json
{
  "django__django-11179": 2,
  "django__django-11333": 2,
  "matplotlib__matplotlib-22835": 4,
  "scikit-learn__scikit-learn-11281": 2
}
```

This ensures these instances start with appropriate resources, reducing failed attempts and speeding up evaluation.
