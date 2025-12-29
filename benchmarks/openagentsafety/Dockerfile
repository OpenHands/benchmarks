ARG BASE_IMAGE=ghcr.io/sani903/openagentsafety_base_image-image:1.0
FROM ${BASE_IMAGE}

# Install git and openai
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install OpenHands packages
COPY vendor/software-agent-sdk /tmp/software-agent-sdk
RUN pip install --no-cache-dir \
    /tmp/software-agent-sdk/openhands-sdk \
    /tmp/software-agent-sdk/openhands-tools \
    /tmp/software-agent-sdk/openhands-agent-server
RUN rm -rf /tmp/software-agent-sdk

# Install OpenAI (REQUIRED for NPC script)
RUN pip install --no-cache-dir openai

# Install Playwright for browser support
RUN pip install --no-cache-dir playwright
RUN playwright install chromium --with-deps

# Create NPC chat script in /usr/local/bin
COPY benchmarks/benchmarks/openagentsafety/chat_npc.py /usr/local/bin/chat_npc
RUN chmod +x /usr/local/bin/chat_npc

WORKDIR /workspace
EXPOSE 8000

# CRITICAL FIX: ENTRYPOINT gets the command, CMD provides default args
# When docker run passes args, they replace CMD but ENTRYPOINT stays
ENTRYPOINT ["python", "-m", "openhands.agent_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
