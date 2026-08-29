# idCam

Production document capture for **Raspberry Pi 5** + **Arducam 64MP Hawkeye** autofocus camera.

Continuously monitors the capture area, detects documents, stabilizes, autofocuses, captures, validates, and uploads to a remote server. Designed for 24/7 unattended operation.

## Features

- Explicit deterministic state machine (no callback spaghetti)
- Single active capture transaction (no queue)
- Arducam 64MP / libcamera / Picamera2 integration
- Autofocus with sharpness verification
- Document detection and stability gating
- Atomic capture files with validation
- HTTPS upload with idempotency key
- Structured JSON logging with rotation
- systemd autostart with recovery
- Mock camera mode for development/testing

## Requirements

- Raspberry Pi 5
- Arducam 64MP Hawkeye (CSI, CAM1 port on Pi 5)
- Raspberry Pi OS Bookworm or Trixie (64-bit)
- Arducam driver stack (see Installation)

## Production installation on Raspberry Pi

On a fresh Raspberry Pi OS (**Bookworm or Trixie, 64-bit / arm64**) the intended
first-install flow is:

```bash
git clone https://github.com/shipittofar/idCamV3.git
cd idCamV3
sudo bash ./scripts/install.sh
```

The installer is the bootstrap. It installs OS packages, the `idcam` system user,
application venv, production config template, and systemd unit. **Do not**
pre-create `/etc/idcam`, `/var/lib/idcam`, `/var/log/idcam`, or `/run/idcam`, and
**do not** install Python/camera packages by hand to work around the installer.

The installer walks eight steps:

1. Checking supported OS (Raspberry Pi OS 64-bit, Bookworm/Trixie)
2. Checking package manager
3. Installing system dependencies (`python3-picamera2` / `python3-libcamera` from apt;
   numpy and OpenCV from PyPI wheels — not `libatlas-base-dev`, which Trixie removed)
4. Creating the `idcam` service user and directories
5. Installing the Python application into `/opt/idcam/venv` (`--system-site-packages`
   so Picamera2 comes from Raspberry Pi OS, not pip)
6. Installing configuration and the systemd unit
7. Validating camera/application environment
8. Final installation validation

It creates the `idcam` user/group (not root), writes a first-boot config **only if**
`/etc/idcam/config.yaml` does not already exist, installs
`/etc/systemd/system/idcam.service`, reloads systemd, and enables the unit. It does
**not** start the service until `device_id` and `server.url` are real values (the
shipped template uses `CHANGE_ME` / `https://unconfigured.invalid` and will fail
validation on purpose). Duplicate APT-source warnings on the Pi are ignored; the
installer does not rewrite `/etc/apt`.

If a **required** package has no installation candidate, the installer stops with
that package name and does not continue as if install succeeded.

Site-specific values the installer cannot guess — set these after the first run:

```bash
sudo nano /etc/idcam/config.yaml
```

Set at least:

- `device_id` — unique id for this Pi (not `CHANGE_ME`)
- `server.url` — real HTTPS API base URL (not `example.com` / `unconfigured.invalid`)
- `server.path` — upload path (default `/api/document`)

Set the auth token in the environment file (preferred; never commit secrets):

```bash
sudo nano /etc/idcam/environment
```

```bash
IDCAM_AUTH_TOKEN=your-token-here
```

Optional overrides in the same file: `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`.

Then enable and start:

```bash
sudo systemctl enable --now idcam
```

Verify:

```bash
sudo systemctl status idcam
journalctl -u idcam -f
cat /run/idcam/status.json
```

### Upgrade

Existing config, token, captured data, and logs are preserved.

```bash
cd idCamV3
git pull
sudo bash ./scripts/install.sh
sudo systemctl restart idcam   # if the service was not already running
```

### Uninstall

```bash
sudo ./scripts/uninstall.sh              # keeps /etc/idcam, /var/lib/idcam, /var/log/idcam
sudo ./scripts/uninstall.sh --purge      # type DELETE to also remove config/data/logs
```

## Development (off-device)

```bash
pip install -e ".[dev]"
IDCAM_MOCK_CAMERA=1 idcam --config config.example.yaml
```

## Configuration

See `config.example.yaml`. Key settings:

| Setting | Description |
|---------|-------------|
| `device_id` | Unique device identifier |
| `server.url` | HTTPS API base URL |
| `server.path` | Upload endpoint path |
| `auth.token` | Bearer token (prefer `IDCAM_AUTH_TOKEN` env) |
| `camera.capture_width/height` | Full-res capture (default 9152×6944) |
| `quality.min_sharpness` | Laplacian variance threshold |

Environment overrides: `IDCAM_CONFIG`, `IDCAM_AUTH_TOKEN`, `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`, `IDCAM_ENV`, `IDCAM_MOCK_CAMERA`.

## Live Debug Viewer

Optional development viewer. **Disabled by default** (`debug.enabled: false`). It is not required for capture.

When enabled in `/etc/idcam/config.yaml`, the existing preview pipeline publishes the latest frame to a debug HTTP observer (no second camera):

```yaml
debug:
  enabled: true
  host: 0.0.0.0
  port: 8088
  fps: 5.0
  jpeg_quality: 70
  max_width: 1920
  overlay: true
```

This exposes a live camera image over HTTP **without authentication**. Do not enable on untrusted networks. Restart the service after changing config:

```bash
sudo systemctl restart idcam
hostname -I
```

Then open `http://<PI-IP>:8088/preview/live` (MJPEG: `/preview/live.mjpg`, JSON: `/preview/status`). Auth tokens are never shown.

## User Status

The application writes real-time status to `/run/idcam/status.json`:

```json
{
  "state": "IDLE",
  "message": "Ready — Place document",
  "camera_healthy": true,
  "network_healthy": true
}
```

## Server API

See [API Protocol](docs/API_PROTOCOL.md) for full idempotency and timeout semantics.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
IDC_RUN_HARDWARE_TESTS=1 pytest tests/ -v -m hardware  # on Pi with camera
```

Hardware tests are skipped off-device. Run the full suite after installer changes.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security](docs/SECURITY.md)

## License

Proprietary
