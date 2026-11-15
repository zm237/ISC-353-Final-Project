#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VIDEO_ARCHIVE="${ROOT_DIR}/archive.zip"
AUDIO_ARCHIVE="${ROOT_DIR}/crema_d.tar.gz"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"
DATA_DIR="${ROOT_DIR}/data"
VIDEO_DIR="${DATA_DIR}/VideoFLV"
AUDIO_DIR="${DATA_DIR}/AudioWAV"
PYTHON="${PYTHON:-python3}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    printf 'Error: %s was not found. Install Python 3 and rerun this script.\n' "${PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${VIDEO_ARCHIVE}" ]]; then
    printf 'Error: missing video archive: %s\n' "${VIDEO_ARCHIVE}" >&2
    exit 1
fi

if [[ ! -f "${AUDIO_ARCHIVE}" ]]; then
    printf 'Error: missing audio archive: %s\n' "${AUDIO_ARCHIVE}" >&2
    exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    printf 'Error: missing requirements file: %s\n' "${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

mkdir -p "${VIDEO_DIR}" "${AUDIO_DIR}"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_DIR}"' EXIT

unzip -q "${VIDEO_ARCHIVE}" -d "${TEMP_DIR}/video"
find "${TEMP_DIR}/video" -type f \( -iname '*.flv' -o -iname '*.mp4' \) \
    -exec cp -n -- {} "${VIDEO_DIR}/" \;

tar -xzf "${AUDIO_ARCHIVE}" -C "${TEMP_DIR}"
find "${TEMP_DIR}" -type f -iname '*.wav' \
    -exec cp -n -- {} "${AUDIO_DIR}/" \;

video_count="$(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -iname '*.flv' -o -iname '*.mp4' \) | wc -l)"
audio_count="$(find "${AUDIO_DIR}" -maxdepth 1 -type f -iname '*.wav' | wc -l)"

if [[ "${video_count}" -eq 0 || "${audio_count}" -eq 0 ]]; then
    printf 'Error: no media files were found after extraction (videos: %s, audio: %s).\n' \
        "${video_count}" "${audio_count}" >&2
    exit 1
fi

printf 'Environment ready in %s\n' "${VENV_DIR}"
printf 'Video files: %s (%s)\n' "${video_count}" "${VIDEO_DIR}"
printf 'Audio files: %s (%s)\n' "${audio_count}" "${AUDIO_DIR}"
printf '\nNext steps:\n'
printf '  source .venv/bin/activate\n'
printf '  python extract_features_fast.py\n'
printf '  python model.py\n'
