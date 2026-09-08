# OKX Event-Contract Paper Trading Bot — server image.
# Meant to be run via docker-compose on a server, not locally (see README →
# "Деплой на сервер (Docker)"). Ships the web dashboard on :8000; console
# mode makes no sense in a container with no attached TTY.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so code-only changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Deliberately NOT copying .env / data/ — secrets and
# runtime state come from the compose file's env_file + volume mounts, never
# baked into the image.
COPY run.py .
COPY src/ src/
COPY config.yaml .

RUN mkdir -p /app/data

# Deliberately runs as root (no USER directive): ./data is a bind-mounted
# HOST directory (see docker-compose.yml), so its real ownership/permissions
# are whatever they are ON THE HOST — anything baked into the image (a
# non-root user + chmod at build time) gets completely shadowed by the
# mount and has no effect at runtime. A previous version of this Dockerfile
# switched to a non-root `botuser` and hit exactly that: PermissionError
# writing data/bot.log the moment ./data was owned by someone else on the
# host. Running as root sidesteps the whole class of bind-mount permission
# mismatches without needing an entrypoint script to chown things at
# startup. Fine for a personal/internal dashboard; harden further yourself
# (non-root + entrypoint chown, or a named volume instead of a bind mount)
# if this needs to withstand a less trusted environment.

EXPOSE 8000

# Binds 127.0.0.1 by default (src/config.py); compose sets
# DASHBOARD_WEB_HOST=0.0.0.0 so the port mapping actually works.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/state', timeout=3)" || exit 1

CMD ["python", "run.py"]
