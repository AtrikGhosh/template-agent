FROM registry.access.redhat.com/ubi9/python-312:latest

# --------------------------------------------------------------------------------------------------
# set the working directory to /app
# --------------------------------------------------------------------------------------------------

WORKDIR /app

# --------------------------------------------------------------------------------------------------
# Copy manifest files and install python packages
# --------------------------------------------------------------------------------------------------

USER root
COPY pyproject.toml /app/pyproject.toml
RUN pip install uv
RUN uv venv
RUN source /app/.venv/bin/activate
RUN uv pip install -r pyproject.toml

# Create cache directory for embedding model
RUN mkdir -p /app/.cache/huggingface && \
    chown -R default:root /app/.cache && \
    chmod -R 775 /app/.cache

# Set cache environment variables for embedding model
ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub

RUN chown -R default:root /app/.cache && \
    find /app/.cache -type d -exec chmod 775 {} \; && \
    find /app/.cache -type f -exec chmod 664 {} \;

USER default

# --------------------------------------------------------------------------------------------------
# copy source code and files
# --------------------------------------------------------------------------------------------------

COPY template_agent /app/template_agent

# --------------------------------------------------------------------------------------------------
# Set PYTHONPATH to include /app
# --------------------------------------------------------------------------------------------------

ENV PYTHONPATH=/app


# --------------------------------------------------------------------------------------------------
# add entrypoint for the container
# --------------------------------------------------------------------------------------------------

CMD ["/app/.venv/bin/python", "-m", "template_agent.src.main"]
