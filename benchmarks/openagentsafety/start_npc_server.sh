#!/bin/bash

# Environment variables for NPC server
export NPC_API_KEY="${LITELLM_API_KEY}"
export NPC_BASE_URL="${LITELLM_BASE_URL}"
export NPC_MODEL="${LITELLM_MODEL}"

# Start the NPC server
python3 npc_server.py