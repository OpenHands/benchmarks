## Agentic Code Search

Benchmarking code to evaluate LLMs on their ability to localize code from a python repository that requires editing to fix a given issue description in natural language

- NOTE: The JSONL file for the ground truth is prepared using [this code](https://github.com/gersteinlab/LocAgent/blob/master/util/benchmark/gen_oracle_locations.py) with the only modification being allowing max_num_file_edits > 1, and some additional if-else blocks for other SWE-Bench-styled datasets like SWE-Gym.