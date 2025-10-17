FROM ghcr.io/sani903/openagentsafety_base_image-image:1.0

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install OpenHands packages
RUN pip install --no-cache-dir \
    "git+https://github.com/All-Hands-AI/agent-sdk.git@main#subdirectory=openhands/sdk"

RUN pip install --no-cache-dir \
    "git+https://github.com/All-Hands-AI/agent-sdk.git@main#subdirectory=openhands/tools"

RUN pip install --no-cache-dir \
    "git+https://github.com/All-Hands-AI/agent-sdk.git@main#subdirectory=openhands/agent_server"

    # Install Playwright for browser support
RUN pip install --no-cache-dir playwright

# Install Chromium browser with dependencies
RUN playwright install chromium --with-deps

# Copy custom tools into the image
COPY custom_tools /opt/custom_tools
ENV PYTHONPATH="/opt:${PYTHONPATH}"

# Create Python entry point that registers tools then starts server
RUN cat > /opt/start_server.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

# Add custom tools to path
sys.path.insert(0, '/opt')

# Register custom tools
try:
    from custom_tools.register import register_custom_tools
    register_custom_tools()
    print("Custom tools registered successfully")
except Exception as e:
    print(f"Warning: Failed to register custom tools: {e}")

# Now start the agent server
from openhands.agent_server.__main__ import main
if __name__ == '__main__':
    main()
EOF

RUN chmod +x /opt/start_server.py

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["python3", "/opt/start_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]