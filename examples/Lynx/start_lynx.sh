#!/usr/bin/env sh
# -------------------------------------------------------------
# T-ELF / TELF launcher (zsh/bash/sh compatible)
# - CDs to a directory that CONTAINS "TELF/" so:
#     streamlit run TELF/applications/Lynx/frontend/main.py
#   works and the app finds TELF/projects correctly.
# - Copies each -p project into TELF/projects/
# - ALSO copies a settings.json that sits NEXT TO THIS SCRIPT
#   into each destination project (backs up existing one).
#
# Usage:
#   ./start_lynx.sh
#   ./start_lynx.sh -p "../Full TELF Pipeline/.../07_SemanticHNMFk"
#   ./start_lynx.sh -p /abs/path/ProjA -p "../rel path/ProjB"
#   ./start_lynx.sh --ssh user@host --port 8501
#   ./start_lynx.sh --no-copy
# -------------------------------------------------------------

set -eu

PORT=8501
SSH_TARGET=""
COPY_PROJECTS=1
PROJECTS=""

# --- parse args (POSIX)
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--project)
      [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      PROJECTS="${PROJECTS}${PROJECTS:+
}$2"
      shift 2
      ;;
    --ssh)
      [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      SSH_TARGET="$2"
      shift 2
      ;;
    --port)
      [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --no-copy)
      COPY_PROJECTS=0
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [-p PATH]... [--ssh USER@HOST] [--port N] [--no-copy]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# ---- resolve script dir robustly (works in zsh/bash/sh)
case "$0" in
  /*) SCRIPT_PATH="$0" ;;
  *)  SCRIPT_PATH="$(pwd)/$0" ;;
esac
SCRIPT_DIR=$(cd "$(dirname -- "$SCRIPT_PATH")" 2>/dev/null && pwd -P)
SETTINGS_SRC="$SCRIPT_DIR/settings.json"

# ---- find a repo root that CONTAINS a TELF/ directory
# Try common layouts:
#   <prefix>/TELF/...
#   <prefix>/T-ELF/TELF/...
find_repo_root() {
  CUR="$SCRIPT_DIR"
  i=0
  while [ $i -le 6 ]; do
    if [ -d "$CUR/TELF" ]; then
      echo "$CUR"; return 0
    fi
    if [ -d "$CUR/T-ELF/TELF" ]; then
      echo "$CUR/T-ELF"; return 0
    fi
    if [ -d "$CUR/../TELF" ]; then
      echo "$(cd "$CUR/.." && pwd -P)"; return 0
    fi
    CUR=$(cd "$CUR/.." 2>/dev/null && pwd -P)
    i=$((i+1))
  done
  echo ""
}

REPO_ROOT="$(find_repo_root)"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: couldn't locate a directory that contains 'TELF/' starting from: $SCRIPT_DIR" >&2
  exit 1
fi

TELF_DIR="$REPO_ROOT"
APP_REL="TELF/applications/Lynx/frontend/main.py"
APP_ABS="$REPO_ROOT/$APP_REL"
PROJECTS_DIR="$TELF_DIR/projects"

if [ ! -f "$APP_ABS" ]; then
  echo "Error: Streamlit app not found at: $APP_ABS" >&2
  exit 1
fi

# Ensure projects dir
mkdir -p "$PROJECTS_DIR"

# Resolve a user-supplied directory (relative to current working dir; handles spaces)
resolve_dir() {
  _p="$1"
  case "$_p" in
    /*) ABS="$_p" ;;
    *)  ABS="$(pwd)/$_p" ;;
  esac
  if cd "$ABS" 2>/dev/null; then
    pwd -P
    cd - >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

copy_settings_into() {
  _dest="$1"
  if [ -f "$SETTINGS_SRC" ]; then
    if [ -f "$_dest/settings.json" ]; then
      mv -f "$_dest/settings.json" "$_dest/settings.json.bak"
    fi
    cp -f "$SETTINGS_SRC" "$_dest/settings.json"
    echo "  -> settings.json copied into '$(basename -- "$_dest")' (backup: settings.json.bak if existed)"
  else
    echo "  -> Warning: settings.json not found next to the script at: $SETTINGS_SRC (skipping settings copy)" >&2
  fi
}

# Copy projects (if any)
if [ "$COPY_PROJECTS" -eq 1 ] && [ -n "$PROJECTS" ]; then
  echo "$PROJECTS" | while IFS= read -r src_raw; do
    [ -n "$src_raw" ] || continue
    if ! src_abs="$(resolve_dir "$src_raw")"; then
      echo "Warning: project path not found: $src_raw (skipping)" >&2
      continue
    fi
    base_name=$(basename -- "$src_abs")
    dest="$PROJECTS_DIR/$base_name"
    echo "Copying '$src_raw' -> '$dest' ..."
    rm -rf -- "$dest"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$src_abs"/ "$dest"/
    else
      mkdir -p "$dest"
      cp -R -- "$src_abs"/. "$dest"/
    fi
    # Always drop the local settings.json into the copied project
    copy_settings_into "$dest"
  done
fi

# Optional SSH port forward
if [ -n "$SSH_TARGET" ]; then
  echo "Starting SSH tunnel: localhost:$PORT -> $SSH_TARGET:localhost:$PORT"
  if ! ssh -N -f -L "$PORT:localhost:$PORT" "$SSH_TARGET"; then
    echo "Warning: SSH port forward failed; continuing without it." >&2
  fi
fi

# Check streamlit
if ! command -v streamlit >/dev/null 2>&1; then
  echo "Error: 'streamlit' not found on PATH. Activate your env or 'pip install streamlit'." >&2
  exit 1
fi

# CD to the repo root so the *relative* Streamlit path works
cd "$REPO_ROOT"
echo "Working directory: $(pwd -P)"
echo "Ensured: $(pwd -P)/TELF/projects"
echo "Launching: streamlit run $APP_REL --server.port $PORT"
exec streamlit run "$APP_REL" --server.port "$PORT"
