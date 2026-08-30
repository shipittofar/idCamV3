#!/usr/bin/env bash
# idCam production installer for Raspberry Pi OS (Bookworm/Trixie, 64-bit).
# First-install bootstrap: OS check, apt packages, idcam user, venv, config, systemd.
# Does NOT create /run/idcam — systemd RuntimeDirectory=idcam owns that path.
#
# Usage:
#   sudo ./scripts/install.sh
#   sudo ./scripts/install.sh --start
#   sudo IDC_DEVICE_ID=pi-01 IDC_SERVER_URL=https://api.example.net \
#        IDC_AUTH_TOKEN=secret ./scripts/install.sh --start
#   ./scripts/install.sh --staging /tmp/idcam-staging --skip-python

set -euo pipefail

IDC_VERSION="1.0.0"
SERVICE_USER="idcam"
SERVICE_GROUP="idcam"
IDC_INSTALL_STEPS=8

# Required OS packages. Fail the installer if any of these have no candidate
# or cannot be installed. numpy and opencv-python-headless come from PyPI
# wheels (bundled OpenBLAS). Do not install libatlas-base-dev: ATLAS was
# dropped in Debian Trixie and is not a runtime requirement of this project.
IDC_APT_REQUIRED=(
    python3
    python3-pip
    python3-venv
    python3-dev
    python3-libcamera
    python3-picamera2
    v4l-utils
    wget
    curl
)

# Recommended camera-stack extras. Install when this OS has a candidate.
# python3-kms++ is the DRM backend Picamera2 prefers; libcamera-dev is
# headers only. Neither is a substitute for python3-picamera2.
IDC_APT_RECOMMENDED=(
    python3-kms++
    libcamera-dev
)

# Camera CLI tools. Raspberry Pi OS renamed libcamera-apps -> rpicam-apps.
# Install whichever names exist; skip names with no candidate.
IDC_APT_CAMERA_CLI=(
    rpicam-apps
    libcamera-apps
)

STAGING_ROOT=""
START_SERVICE=0
SKIP_APT=0
SKIP_CAMERA=0
SKIP_PYTHON=0
SKIP_SYSTEMD=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"

INSTALL_DIR=""
VENV_DIR=""
CONFIG_DIR=""
STATE_DIR=""
TEMP_DIR=""
LOG_DIR=""
SYSTEMD_DIR=""
CONFIG_FILE=""
ENV_FILE=""
UNIT_FILE=""
DROPIN_DIR=""

usage() {
    cat <<'EOF'
idCam installer

Usage: sudo ./scripts/install.sh [options]

On a fresh Raspberry Pi OS (Bookworm/Trixie, 64-bit) this is the complete
bootstrap. Do not pre-install OS packages by hand.

Options:
  --start           Start/restart the service if configuration validates
  --skip-apt        Skip apt-get
  --skip-camera     Skip Arducam driver installation
  --skip-python     Skip venv/pip (layout-only / tests)
  --skip-systemd    Skip enabling systemd (unit file is still installed)
  --staging DIR     Install under DIR (no root, apt, useradd, or systemctl)
  -h, --help        Show this help

Optional environment variables (applied only when creating new files, and
never overwriting an existing non-empty IDCAM_AUTH_TOKEN):
  IDC_DEVICE_ID     device_id / IDCAM_DEVICE_ID
  IDC_SERVER_URL    server.url / IDCAM_SERVER_URL
  IDC_AUTH_TOKEN    IDCAM_AUTH_TOKEN in /etc/idcam/environment
EOF
}

log() { echo "=== $* ==="; }
warn() { echo "WARNING: $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

step() {
    local n="$1"
    shift
    echo
    echo "[${n}/${IDC_INSTALL_STEPS}] $*"
}

prefix_path() {
    if [[ -n "$STAGING_ROOT" ]]; then
        printf '%s\n' "${STAGING_ROOT}$1"
    else
        printf '%s\n' "$1"
    fi
}

init_paths() {
    INSTALL_DIR="$(prefix_path /opt/idcam)"
    VENV_DIR="${INSTALL_DIR}/venv"
    CONFIG_DIR="$(prefix_path /etc/idcam)"
    STATE_DIR="$(prefix_path /var/lib/idcam)"
    TEMP_DIR="${STATE_DIR}/tmp"
    LOG_DIR="$(prefix_path /var/log/idcam)"
    SYSTEMD_DIR="$(prefix_path /etc/systemd/system)"
    CONFIG_FILE="${CONFIG_DIR}/config.yaml"
    ENV_FILE="${CONFIG_DIR}/environment"
    UNIT_FILE="${SYSTEMD_DIR}/idcam.service"
    DROPIN_DIR="${SYSTEMD_DIR}/idcam.service.d"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --start) START_SERVICE=1; shift ;;
            --skip-apt) SKIP_APT=1; shift ;;
            --skip-camera) SKIP_CAMERA=1; shift ;;
            --skip-python) SKIP_PYTHON=1; shift ;;
            --skip-systemd) SKIP_SYSTEMD=1; shift ;;
            --staging)
                [[ -n "${2:-}" ]] || { echo "ERROR: --staging requires a directory" >&2; exit 2; }
                STAGING_ROOT="$2"
                shift 2
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "ERROR: Unknown option: $1" >&2; usage >&2; exit 2 ;;
        esac
    done

    if [[ -n "$STAGING_ROOT" ]]; then
        mkdir -p "$STAGING_ROOT"
        STAGING_ROOT="$(cd "$STAGING_ROOT" && pwd)"
        SKIP_APT=1
        SKIP_CAMERA=1
        SKIP_SYSTEMD=1
    fi
}

ensure_dir() {
    mkdir -p "$2"
    if ! chmod "$1" "$2" 2>/dev/null; then
        [[ -n "$STAGING_ROOT" ]] || die "Could not chmod $1 $2"
    fi
}

copy_file() {
    cp "$2" "$3"
    if ! chmod "$1" "$3" 2>/dev/null; then
        [[ -n "$STAGING_ROOT" ]] || die "Could not chmod $1 $3"
    fi
}

require_root() {
    if [[ -n "$STAGING_ROOT" ]]; then
        return 0
    fi
    if [[ "${EUID}" -ne 0 ]]; then
        die "Run as root: sudo ./scripts/install.sh"
    fi
}

python_ok() {
    command -v python3 >/dev/null 2>&1 || return 1
    python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

is_raspberry_pi_os() {
    [[ -f /etc/rpi-issue ]] && return 0
    [[ -f /usr/bin/raspi-config ]] && return 0
    if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
        return 0
    fi
    if [[ -e /etc/apt/sources.list.d/raspi.list || -e /etc/apt/sources.list.d/raspi.sources ]]; then
        return 0
    fi
    return 1
}

check_supported_os() {
    step 1 "Checking supported OS"
    if [[ -n "$STAGING_ROOT" ]]; then
        echo "Staging install: OS check skipped"
        return 0
    fi

    local arch
    arch="$(uname -m)"
    case "$arch" in
        aarch64|arm64) echo "Architecture: ${arch}" ;;
        *)
            die "Unsupported architecture '${arch}'. idCam requires Raspberry Pi OS 64-bit (arm64)."
            ;;
    esac

    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "OS: ${PRETTY_NAME:-unknown} (${VERSION_CODENAME:-unknown})"
        case "${VERSION_CODENAME:-}" in
            bookworm|trixie) ;;
            *)
                warn "Untested Debian codename '${VERSION_CODENAME:-unknown}'. Supported: bookworm, trixie."
                ;;
        esac
    else
        warn "/etc/os-release is missing; cannot confirm Debian codename"
    fi

    if ! is_raspberry_pi_os; then
        die "This installer requires Raspberry Pi OS (Bookworm/Trixie, 64-bit).
python3-picamera2 / python3-libcamera are Raspberry Pi packages, not vanilla Debian.
Do not work around this by installing packages by hand."
    fi

    if [[ -f /proc/device-tree/model ]]; then
        echo "Hardware: $(tr -d '\0' < /proc/device-tree/model)"
    fi
}

check_package_manager() {
    step 2 "Checking package manager"
    [[ -f "${PROJECT_DIR}/pyproject.toml" ]] || die "pyproject.toml not found in ${PROJECT_DIR}"
    [[ -f "${DEPLOY_DIR}/idcam.service" ]] || die "Missing ${DEPLOY_DIR}/idcam.service"
    [[ -f "${DEPLOY_DIR}/config.yaml.template" ]] || die "Missing ${DEPLOY_DIR}/config.yaml.template"
    [[ -f "${DEPLOY_DIR}/environment" ]] || die "Missing ${DEPLOY_DIR}/environment"

    if [[ -n "$STAGING_ROOT" || "$SKIP_APT" -eq 1 ]]; then
        echo "apt skipped (--staging or --skip-apt)"
        return 0
    fi
    command -v apt-get >/dev/null || die "apt-get is required on Raspberry Pi OS / Debian"
    command -v apt-cache >/dev/null || die "apt-cache is required on Raspberry Pi OS / Debian"
    echo "apt-get: $(command -v apt-get)"
}

apt_has_candidate() {
    local pkg="$1"
    local candidate=""
    candidate="$(apt-cache policy "$pkg" 2>/dev/null | awk '/^[[:space:]]*Candidate:/ {print $2; exit}')"
    [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

require_apt_candidates() {
    local pkg
    local missing=()
    for pkg in "$@"; do
        if ! apt_has_candidate "$pkg"; then
            missing+=("$pkg")
        fi
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi
    die "Required package(s) have no installation candidate on this OS:

  ${missing[*]}

idCam supports Raspberry Pi OS Bookworm/Trixie (64-bit, arm64).
These packages must come from Raspberry Pi OS apt (python3-picamera2 is not
on vanilla Debian). The installer does not use libatlas-base-dev (removed in
Debian Trixie; numpy/opencv are installed from PyPI wheels).

The installer will not continue and will not install remaining steps.
Fix the OS/repository, then re-run: sudo bash ./scripts/install.sh
Do not install these packages by hand as a substitute for the installer."
}

run_apt_update() {
    local update_log status real_errors
    update_log="$(mktemp)"
    set +e
    apt-get update >"$update_log" 2>&1
    status=$?
    set -e
    cat "$update_log"

    if grep -Eqi 'configured multiple times' "$update_log"; then
        warn "Duplicate APT repository entries detected (environment/configuration issue)."
        warn "The installer will not modify /etc/apt/sources.list or files in /etc/apt/sources.list.d."
        warn "Duplicate-source warnings are ignored; real apt errors are not."
    fi

    if [[ "$status" -eq 0 ]]; then
        rm -f "$update_log"
        return 0
    fi

    real_errors="$(grep -E '^(E:|Error:)' "$update_log" | grep -Eiv 'configured multiple times' || true)"
    if [[ -n "$real_errors" ]]; then
        rm -f "$update_log"
        die "apt-get update failed:

${real_errors}

Fix the package manager (network, signatures, or broken repos) and re-run this installer.
The installer does not rewrite APT sources."
    fi

    if apt-cache policy python3 >/dev/null 2>&1; then
        warn "apt-get update exited ${status} with warnings only; package lists are usable."
        rm -f "$update_log"
        return 0
    fi
    rm -f "$update_log"
    die "apt-get update failed (exit ${status}) and package lists are not usable."
}

install_required_apt_packages() {
    echo "Installing required packages: $*"
    if ! apt-get install -y "$@"; then
        die "Failed to install required package(s): $*

This is a hard failure. The installer will not continue with a partial system.
Fix the apt error above and re-run: sudo bash ./scripts/install.sh
Do not install these packages by hand as a substitute for the installer."
    fi
}

install_optional_apt_packages() {
    local pkg
    for pkg in "$@"; do
        if ! apt_has_candidate "$pkg"; then
            echo "Optional package not available on this OS (skipped): ${pkg}"
            continue
        fi
        echo "Installing optional package: ${pkg}"
        if ! apt-get install -y "$pkg"; then
            warn "Optional package ${pkg} failed to install; continuing because it is not required"
        fi
    done
}

install_apt_packages() {
    if [[ "$SKIP_APT" -eq 1 ]]; then
        echo "Skipping apt package installation"
        return 0
    fi
    export DEBIAN_FRONTEND=noninteractive
    run_apt_update
    require_apt_candidates "${IDC_APT_REQUIRED[@]}"
    install_required_apt_packages "${IDC_APT_REQUIRED[@]}"
    install_optional_apt_packages "${IDC_APT_RECOMMENDED[@]}"
    install_optional_apt_packages "${IDC_APT_CAMERA_CLI[@]}"
}

install_arducam() {
    if [[ "$SKIP_CAMERA" -eq 1 ]]; then
        return 0
    fi
    local bootcfg=""
    if [[ -f /boot/firmware/config.txt ]]; then
        bootcfg=/boot/firmware/config.txt
    elif [[ -f /boot/config.txt ]]; then
        bootcfg=/boot/config.txt
    fi
    if [[ -n "$bootcfg" ]] && grep -q "arducam-64mp" "$bootcfg"; then
        echo "Arducam overlay already present in ${bootcfg}"
        return 0
    fi
    echo "Installing Arducam 64MP packages when the vendor script is reachable"
    echo "After a new overlay install, reboot so dtoverlay=arducam-64mp takes effect."
    if ! wget -O /tmp/install_pivariety_pkgs.sh \
        https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh; then
        warn "Could not download the Arducam vendor script."
        warn "OS libcamera/picamera2 packages are still installed. If the 64MP sensor is not detected, re-run this installer after network to GitHub works."
        return 0
    fi
    chmod +x /tmp/install_pivariety_pkgs.sh
    local pkg
    for pkg in libcamera_dev libcamera_apps 64mp_pi_hawk_eye_kernel_driver; do
        if /tmp/install_pivariety_pkgs.sh -p "$pkg"; then
            echo "Arducam component installed: ${pkg}"
        else
            warn "Arducam vendor script failed for ${pkg} (non-fatal if Raspberry Pi libcamera packages are already installed)"
        fi
    done
}

create_user() {
    if [[ -n "$STAGING_ROOT" ]]; then
        echo "Staging install: not creating system user ${SERVICE_USER}"
        return 0
    fi
    if ! getent group "$SERVICE_GROUP" >/dev/null; then
        groupadd --system "$SERVICE_GROUP"
        echo "Created system group ${SERVICE_GROUP}"
    else
        echo "Group ${SERVICE_GROUP} already exists"
    fi
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
        useradd --system --gid "$SERVICE_GROUP" \
            --home-dir /var/lib/idcam --shell /usr/sbin/nologin \
            --no-create-home "$SERVICE_USER"
        echo "Created system user ${SERVICE_USER}"
    else
        echo "User ${SERVICE_USER} already exists (not recreated)"
    fi
    local extra=""
    for g in video render plugdev; do
        if getent group "$g" >/dev/null; then
            usermod -aG "$g" "$SERVICE_USER"
            extra="${extra} ${g}"
        fi
    done
    if [[ -n "$extra" ]]; then
        echo "Ensured ${SERVICE_USER} is in groups:${extra}"
    fi
}

create_directories() {
    ensure_dir 0755 "$INSTALL_DIR"
    ensure_dir 0750 "$CONFIG_DIR"
    ensure_dir 0750 "$STATE_DIR"
    ensure_dir 0750 "$TEMP_DIR"
    ensure_dir 0750 "$LOG_DIR"
    if [[ -z "$STAGING_ROOT" ]]; then
        chown root:root "$INSTALL_DIR" || die "Could not chown ${INSTALL_DIR}"
        chown root:"$SERVICE_GROUP" "$CONFIG_DIR" || die "Could not chown ${CONFIG_DIR}"
        chown "$SERVICE_USER":"$SERVICE_GROUP" "$STATE_DIR" "$TEMP_DIR" "$LOG_DIR" \
            || die "Could not chown state/log directories"
    fi
    echo "Directories: ${INSTALL_DIR}, ${CONFIG_DIR}, ${STATE_DIR}, ${TEMP_DIR}, ${LOG_DIR}"
    echo "/run/idcam is created by systemd RuntimeDirectory=idcam at service start"
}

quote_shell() {
    python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$1"
}

append_env() {
    local key="$1"
    local value="$2"
    if grep -qE "^${key}=.+" "$ENV_FILE"; then
        echo "Keeping existing ${key}"
        return 0
    fi
    if [[ -f "$ENV_FILE" ]]; then
        grep -vE "^${key}=" "$ENV_FILE" > "${ENV_FILE}.tmp" || true
        mv "${ENV_FILE}.tmp" "$ENV_FILE"
    fi
    printf '%s=%s\n' "$key" "$(quote_shell "$value")" >> "$ENV_FILE"
    chmod 0640 "$ENV_FILE"
    echo "Set ${key} in ${ENV_FILE}"
}

install_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        copy_file 0640 "${DEPLOY_DIR}/config.yaml.template" "$CONFIG_FILE"
        if [[ -n "${IDC_DEVICE_ID:-}" ]]; then
            python3 - "$CONFIG_FILE" "${IDC_DEVICE_ID}" <<'PY'
import sys
from pathlib import Path
path, device_id = Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8").replace('device_id: "CHANGE_ME"', f"device_id: {device_id!r}")
path.write_text(text, encoding="utf-8")
PY
        fi
        if [[ -n "${IDC_SERVER_URL:-}" ]]; then
            python3 - "$CONFIG_FILE" "${IDC_SERVER_URL}" <<'PY'
import sys
from pathlib import Path
path, url = Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8").replace("url: https://unconfigured.invalid", f"url: {url}")
path.write_text(text, encoding="utf-8")
PY
        fi
        echo "Created ${CONFIG_FILE}"
    else
        echo "Keeping existing ${CONFIG_FILE}"
    fi
    copy_file 0640 "${DEPLOY_DIR}/config.yaml.template" "${CONFIG_DIR}/config.yaml.template"

    if [[ ! -f "$ENV_FILE" ]]; then
        copy_file 0640 "${DEPLOY_DIR}/environment" "$ENV_FILE"
        echo "Created ${ENV_FILE}"
    else
        echo "Keeping existing ${ENV_FILE}"
        chmod 0640 "$ENV_FILE"
    fi

    if [[ -n "${IDC_DEVICE_ID:-}" ]]; then
        append_env IDCAM_DEVICE_ID "$IDC_DEVICE_ID"
    fi
    if [[ -n "${IDC_SERVER_URL:-}" ]]; then
        append_env IDCAM_SERVER_URL "$IDC_SERVER_URL"
    fi
    if [[ -n "${IDC_AUTH_TOKEN:-}" ]]; then
        append_env IDCAM_AUTH_TOKEN "$IDC_AUTH_TOKEN"
    fi

    if [[ -z "$STAGING_ROOT" ]]; then
        chown root:"$SERVICE_GROUP" "$CONFIG_FILE" "${CONFIG_DIR}/config.yaml.template" "$ENV_FILE" \
            || die "Could not chown config files"
        chmod 0640 "$CONFIG_FILE" "$ENV_FILE" "${CONFIG_DIR}/config.yaml.template"
        chmod 0750 "$CONFIG_DIR"
    fi
}

install_application() {
    if [[ "$SKIP_PYTHON" -eq 1 ]]; then
        echo "Skipping Python install (--skip-python)"
        return 0
    fi
    python_ok || die "Python 3.11 or newer is required after system package installation"
    echo "Creating venv with --system-site-packages so Picamera2/libcamera come from apt"
    python3 -m venv --system-site-packages "$VENV_DIR"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel
    "${VENV_DIR}/bin/pip" install "${PROJECT_DIR}"
    printf '%s\n' "$IDC_VERSION" > "${INSTALL_DIR}/VERSION"
    if [[ -z "$STAGING_ROOT" ]]; then
        chown -R root:root "$INSTALL_DIR" || die "Could not chown ${INSTALL_DIR}"
        chmod 0755 "$INSTALL_DIR"
    fi
    [[ -x "${VENV_DIR}/bin/idcam" ]] || die "idcam executable was not installed into ${VENV_DIR}/bin/idcam"
    [[ -x "${VENV_DIR}/bin/idcam-admin" ]] || die "idcam-admin was not installed into ${VENV_DIR}/bin/idcam-admin"
}

write_systemd_dropin() {
    local groups=()
    if [[ -n "$STAGING_ROOT" ]]; then
        groups=(video render)
    else
        for g in video render; do
            if getent group "$g" >/dev/null; then
                groups+=("$g")
            fi
        done
    fi
    if [[ ${#groups[@]} -eq 0 ]]; then
        return 0
    fi
    ensure_dir 0755 "$DROPIN_DIR"
    cat > "${DROPIN_DIR}/hardware.conf" <<EOF
[Service]
SupplementaryGroups=${groups[*]}
EOF
    chmod 0644 "${DROPIN_DIR}/hardware.conf"
}

install_systemd_unit() {
    ensure_dir 0755 "$SYSTEMD_DIR"
    copy_file 0644 "${DEPLOY_DIR}/idcam.service" "$UNIT_FILE"
    write_systemd_dropin
    if [[ "$SKIP_SYSTEMD" -eq 1 ]]; then
        echo "Wrote ${UNIT_FILE} (not enabled)"
        return 0
    fi
    systemctl daemon-reload
    systemctl enable idcam.service
    echo "Enabled idcam.service"
}

config_is_valid() {
    if [[ "$SKIP_PYTHON" -eq 1 || ! -x "${VENV_DIR}/bin/idcam-admin" ]]; then
        return 1
    fi
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
    fi
    export IDCAM_ENV=production
    export IDCAM_CONFIG="$CONFIG_FILE"
    "${VENV_DIR}/bin/idcam-admin" --config "$CONFIG_FILE" validate >/dev/null 2>&1
}

validate_camera_environment() {
    step 7 "Validating camera/application environment"
    if [[ "$SKIP_PYTHON" -eq 1 ]]; then
        echo "Python/camera import checks skipped (--skip-python)"
        return 0
    fi
    [[ -d "$VENV_DIR" ]] || die "Python venv is missing: ${VENV_DIR}"
    [[ -x "${VENV_DIR}/bin/python" ]] || die "venv python is missing: ${VENV_DIR}/bin/python"
    [[ -x "${VENV_DIR}/bin/idcam" ]] || die "idcam executable is missing: ${VENV_DIR}/bin/idcam"

    if ! "${VENV_DIR}/bin/python" -c "import picamera2, libcamera" 2>/dev/null; then
        die "Picamera2/libcamera are not importable in ${VENV_DIR}.
The venv is created with --system-site-packages so these must come from apt
packages python3-picamera2 and python3-libcamera (Raspberry Pi OS).
Do not pip install picamera2 as a workaround.
Re-run this installer after the Raspberry Pi camera packages are available."
    fi
    echo "Picamera2 and libcamera import OK (system site-packages)"
}

check_idcam_writable() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        echo "FAIL: ${label} directory is missing: ${path}"
        return 1
    fi
    if [[ -n "$STAGING_ROOT" ]]; then
        return 0
    fi
    if ! sudo -u "$SERVICE_USER" test -w "$path"; then
        echo "FAIL: ${label} is not writable by ${SERVICE_USER}: ${path}"
        return 1
    fi
    echo "${label} writable by ${SERVICE_USER}: ${path}"
}

final_validation() {
    step 8 "Final installation validation"
    local failed=0
    local config_state="unknown"

    [[ -d "$CONFIG_DIR" ]] || { echo "FAIL: ${CONFIG_DIR} missing"; failed=1; }
    [[ -d "$STATE_DIR" ]] || { echo "FAIL: ${STATE_DIR} missing"; failed=1; }
    [[ -d "$TEMP_DIR" ]] || { echo "FAIL: ${TEMP_DIR} missing"; failed=1; }
    [[ -d "$LOG_DIR" ]] || { echo "FAIL: ${LOG_DIR} missing"; failed=1; }
    [[ -f "$CONFIG_FILE" ]] || { echo "FAIL: ${CONFIG_FILE} missing"; failed=1; }
    [[ -f "$ENV_FILE" ]] || { echo "FAIL: ${ENV_FILE} missing"; failed=1; }
    [[ -f "$UNIT_FILE" ]] || { echo "FAIL: ${UNIT_FILE} missing"; failed=1; }
    grep -q "User=idcam" "$UNIT_FILE" || { echo "FAIL: unit does not run as idcam"; failed=1; }
    grep -q "RuntimeDirectory=idcam" "$UNIT_FILE" || { echo "FAIL: unit missing RuntimeDirectory=idcam"; failed=1; }

    if [[ "$SKIP_PYTHON" -eq 0 ]]; then
        [[ -x "${VENV_DIR}/bin/idcam" ]] || { echo "FAIL: ${VENV_DIR}/bin/idcam missing"; failed=1; }
        [[ -x "${VENV_DIR}/bin/idcam-admin" ]] || { echo "FAIL: ${VENV_DIR}/bin/idcam-admin missing"; failed=1; }
    fi

    if [[ -z "$STAGING_ROOT" ]]; then
        check_idcam_writable "$STATE_DIR" "state" || failed=1
        check_idcam_writable "$TEMP_DIR" "temp" || failed=1
        check_idcam_writable "$LOG_DIR" "log" || failed=1
    fi

    if [[ "$SKIP_SYSTEMD" -eq 0 ]] && command -v systemd-analyze >/dev/null 2>&1; then
        if ! systemd-analyze verify "$UNIT_FILE"; then
            echo "FAIL: systemd unit is not syntactically valid: ${UNIT_FILE}"
            failed=1
        else
            echo "systemd unit syntax OK"
        fi
    elif [[ -n "$STAGING_ROOT" || "$SKIP_SYSTEMD" -eq 1 ]]; then
        echo "systemd-analyze skipped (staging or --skip-systemd)"
    fi

    if [[ "$SKIP_PYTHON" -eq 1 || ! -x "${VENV_DIR}/bin/idcam-admin" ]]; then
        config_state="not checked (--skip-python)"
        echo "Config validation skipped"
    elif config_is_valid; then
        config_state="valid"
        echo "Configuration validates"
    else
        config_state="present but not production-ready"
        echo "Configuration exists but is not production-ready (expected until device_id, server.url, and token are set)."
        echo "The service will refuse to start until those values are real (not CHANGE_ME / unconfigured.invalid / example.com)."
    fi

    if [[ "$failed" -ne 0 ]]; then
        die "Installation validation failed. See FAIL lines above. Re-run sudo bash ./scripts/install.sh after fixing them."
    fi
    echo "Validation passed (${config_state})"
}

maybe_start() {
    if [[ "$SKIP_SYSTEMD" -eq 1 ]]; then
        return 0
    fi
    local was_active=0
    if systemctl is-active --quiet idcam.service; then
        was_active=1
    fi
    if [[ "$START_SERVICE" -eq 1 || "$was_active" -eq 1 ]]; then
        if config_is_valid; then
            log "Restarting idcam.service"
            if ! systemctl restart idcam.service; then
                die "idcam.service failed to restart. Inspect: journalctl -u idcam -n 50"
            fi
            if ! systemctl is-active --quiet idcam.service; then
                die "idcam.service is not active after restart. Inspect: journalctl -u idcam -n 50"
            fi
            echo "idcam.service is active"
        else
            warn "Configuration is not valid yet; not starting idcam.service"
            warn "Edit ${CONFIG_FILE} and ${ENV_FILE}, then: sudo systemctl start idcam"
        fi
    elif config_is_valid; then
        echo "Configuration is valid. Start with: sudo systemctl start idcam"
    else
        echo "Service is enabled but not started (configure device_id, server.url, and token first)."
    fi
}

print_summary() {
    local config_note
    if [[ "$SKIP_PYTHON" -eq 0 ]] && config_is_valid; then
        config_note="production configuration validates"
    else
        config_note="site-specific values still required"
    fi

    cat <<EOF

=== idCam ${IDC_VERSION} installation complete ===

Installed:
  Service user/group: ${SERVICE_USER}:${SERVICE_GROUP} (not root)
  Application:        ${VENV_DIR}
  Config:             ${CONFIG_FILE}   (root:${SERVICE_GROUP} 0640; never overwritten on upgrade)
  Environment:        ${ENV_FILE}      (root:${SERVICE_GROUP} 0640)
  State/temp:         ${STATE_DIR} and ${TEMP_DIR}  (${SERVICE_USER}:${SERVICE_GROUP} 0750)
  Logs:               ${LOG_DIR}       (${SERVICE_USER}:${SERVICE_GROUP} 0750)
  Runtime:            /run/idcam via systemd RuntimeDirectory=idcam (mode 0755)
  Unit:               ${UNIT_FILE}

Status: ${config_note}

Remaining site-specific configuration (the installer cannot guess these):
  1. device_id     in ${CONFIG_FILE}   (not CHANGE_ME)
  2. server.url    in ${CONFIG_FILE}   (real HTTPS URL, not example.com / unconfigured.invalid)
  3. IDCAM_AUTH_TOKEN in ${ENV_FILE}

Then start:
  sudo nano ${CONFIG_FILE}
  sudo nano ${ENV_FILE}
  sudo systemctl start idcam
  sudo systemctl status idcam
  journalctl -u idcam -f
  cat /run/idcam/status.json

Upgrade (preserves config, credentials, captured data, and logs):
  cd ${PROJECT_DIR} && git pull && sudo bash ./scripts/install.sh
  sudo systemctl restart idcam     # if it was not already running

Uninstall (keeps config/data/logs unless you pass --purge):
  sudo ./scripts/uninstall.sh
EOF
}

install_main() {
    require_root
    init_paths
    check_supported_os
    check_package_manager

    step 3 "Installing system dependencies"
    install_apt_packages
    install_arducam

    step 4 "Creating idcam service user and directories"
    create_user
    create_directories

    step 5 "Installing Python application"
    install_application

    step 6 "Installing configuration and systemd unit"
    install_config
    install_systemd_unit

    validate_camera_environment
    final_validation
    maybe_start
    print_summary
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    parse_args "$@"
    install_main
fi
