#!/usr/bin/env bash
set -euo pipefail

repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
command_path=/usr/local/bin/lanllm
source_path=$repository/bin/lanllm

fail() {
    printf 'LanLLM install: %s\n' "$*" >&2
    exit 1
}

if (( EUID == 0 )); then
    fail "run this script as your regular user; it requests sudo only for the command link"
fi

command -v python3 >/dev/null || fail "Python 3.11 or newer is required"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' ||
    fail "Python 3.11 or newer is required"
command -v docker >/dev/null || fail "Docker is required"
docker compose version >/dev/null 2>&1 || fail "the Docker Compose plugin is required"
command -v nvidia-smi >/dev/null || fail "an NVIDIA driver with nvidia-smi is required"
[[ -x $source_path ]] || fail "the host command is missing or not executable: $source_path"

if [[ -e $command_path || -L $command_path ]]; then
    [[ -L $command_path ]] || fail "$command_path already exists and is not a symlink"
    [[ $(readlink -f -- "$command_path") == "$source_path" ]] ||
        fail "$command_path points to another installation"
else
    if [[ -w ${command_path%/*} ]]; then
        ln -s -- "$source_path" "$command_path"
    else
        command -v sudo >/dev/null || fail "sudo is required to create $command_path"
        sudo ln -s -- "$source_path" "$command_path"
    fi
    printf 'Installed %s\n' "$command_path"
fi

if [[ ! -e $repository/lanllm.conf ]]; then
    cp -- "$repository/lanllm.conf.example" "$repository/lanllm.conf"
    printf 'Created %s\n' "$repository/lanllm.conf"
fi
if [[ ! -e $repository/.env ]]; then
    cp -- "$repository/.env.example" "$repository/.env"
    printf 'Created %s\n' "$repository/.env"
fi

printf 'LanLLM installed. Edit %s/lanllm.conf, then run: lanllm start\n' "$repository"
