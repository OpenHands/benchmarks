# OpenAgentSafety Benchmark

OpenAgentSafety is a benchmark for evaluating AI agent safety in realistic workplace scenarios. Agents interact with simulated NPCs (Non-Player Characters) in various professional contexts to test their ability to handle social engineering, manipulation, and other safety-critical situations.

## Features

- **Jinja2 Templating**: Customizable agent prompts using Jinja2 templates
** NPC Interacton**: NPC-agent communication through bash script provided in Dockerfile
- **Flexible Configuration**: Support for different LLM providers and models
- **Safety Evaluation**: Comprehensive testing of agent behavior in challenging scenarios

## Environment Setup

### LLM Configuration

Follow the same LLM configuration approach as other benchmarks by creating a JSON config file in `.llm_config/`:

**`.llm_config/gpt-4o-mini.json`:**
```json
{
  "model": "gpt-4o-mini",
  "api_key": "YOUR_OPENAI_API_KEY_HERE"
}
```

**`.llm_config/claude-sonnet.json`:**
```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://llm-proxy.eval.all-hands.dev",
  "api_key": "YOUR_API_KEY_HERE"
}
```

### Docker and TheAgentCompany Services Setup

OpenAgentSafety requires TheAgentCompany infrastructure (GitLab, ownCloud, RocketChat, etc.) for realistic NPC interactions:

```bash
# 1. Start Docker daemon
sudo dockerd > /tmp/docker.log 2>&1 &
sleep 5

# 2. Verify Docker is running
sudo docker version

# 3. Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# 4. Set up TheAgentCompany services (~30GB download, 15-30 min setup)
cd /workspace/project
curl -fsSL https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/download/setup-script-20241208/setup.sh | sh
```

**Requirements:**
- ~30GB disk space
- Sufficient system resources for multiple services
- Docker and docker-compose installed

### NPC Environment Variables

OpenAgentSafety requires additional environment variables for NPC tool interactions:

```bash
# NPC LLM Configuration (for character interactions)
export NPC_API_KEY="sk-..."        # API key for NPC model
export NPC_BASE_URL="https://api.openai.com/v1"  # Base URL for NPC model
export NPC_MODEL="gpt-4o-mini"     # Model name for NPCs
```

## Usage

### Quick Start

1. **Configure your LLM** following the examples above
2. **Start Docker and set up services** (see Docker setup section above)
3. **Set NPC environment variables** (see NPC Environment Variables section above)
4. **Run evaluation**:
   ```bash
   cd benchmarks
   uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
     --dataset mgulavani/openagentsafety_full_updated_v3 \
     --split train \
     --output-dir ./results \
     --num-workers 1 \
     --n-limit 2 \
     --critic pass
   ```

### Basic Command

```bash
uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
  --dataset mgulavani/openagentsafety_full_updated_v3 \
  --split train \
  --output-dir ./results \
  --num-workers 1 \
  --n-limit 2 \
  --critic pass
```

### Command Line Arguments

- `--dataset`: HuggingFace dataset identifier
- `--split`: Dataset split to use (train/test/validation)
- `--output-dir`: Directory to save evaluation results
- `--num-workers`: Number of parallel workers
- `--n-limit`: Limit number of instances to evaluate
- `--critic`: Critic mode (pass/fail evaluation)
- `--prompt-path`: Custom prompt template (optional)
- `--select`: Path to text file containing specific instance IDs to evaluate

### Custom Prompt Templates

The benchmark supports custom Jinja2 templates for agent instructions:

```bash
# Use a custom prompt template
uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
  --dataset mgulavani/openagentsafety_full_updated_v3 \
  --split train \
  --output-dir ./results \
  --prompt-path custom_template.j2
```

Available template choices are automatically detected from the `prompts/` directory.

### Selecting Specific Instances

You can run evaluation on a specific subset of instances using the `--select` option:

1. Create a text file with one instance ID per line:

**instances.txt:**
```
safety-audit
safety-compliance
safety-credentials
```

2. Run evaluation with the selection file:
```bash
uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
  --dataset mgulavani/openagentsafety_full_updated_v3 \
  --split train \
  --output-dir ./results \
  --select instances.txt \
  --critic pass
```

This will only evaluate the instances listed in the file.

## Prompt Templates

### Template Structure

Templates are stored in `benchmarks/openagentsafety/prompts/` and use Jinja2 syntax:

- `default.j2`: Default agent instruction template
- Custom templates can be added and selected via `--prompt-path`

### Template Variables

Templates have access to the following variables:

```jinja2
{{ instance.problem_statement }}    # Main task description
{{ instance.environment }}          # Environment details
{{ instance.npcs }}                 # Number of NPCs
{{ instance.agent_profiles }}       # List of NPC profiles
```

### Example Template Usage

```jinja2
{{ instance.problem_statement }}

## Environment
{{ instance.environment }}

## Available NPCs
{% if instance.npcs > 0 %}
You can interact with the following characters:
{% for profile in instance.agent_profiles %}
- {{ profile.name }}
{% endfor %}
{% endif %}
```

### NPC Configuration

NPCs are configured per-task with:
- Character profiles (age, occupation, background)
- Social goals and motivations
- Strategy hints for realistic behavior

## Architecture

### Main Components

1. **run_infer.py**: Main evaluation script with Jinja2 template support
2. **prompts/**: Jinja2 template directory
3. **Dockerfile**: Container setup for isolated evaluation

## Development

### Adding New Templates

1. Create a new `.j2` file in `benchmarks/openagentsafety/prompts/`
2. Use Jinja2 syntax with available instance variables
3. Test with `--prompt-path your_template.j2`

### Template Validation

Templates are automatically validated and available choices are shown in CLI help:

```bash
uv run openagentsafety-infer --help
# Shows: --prompt-path {default.j2} (choices detected from prompts/ directory)
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure all required API keys and URLs are set
2. **Template Not Found**: Check that custom templates exist in the `prompts/` directory
3. **NPC Configuration**: Verify NPC API credentials are correctly configured

### Debug Mode

Add verbose logging to troubleshoot issues:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uv run openagentsafety-infer ...
```