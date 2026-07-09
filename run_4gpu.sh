#!/usr/bin/env bash
set -euo pipefail

# A self-contained 4-GPU launcher for full_model_zero3_repro.py.
# Both the real and Phantora commands execute the same Python training script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/full_model_zero3_repro.py"

GPUS="${GPUS:-4}"
MODEL="${MODEL:-8B}"
SEQ_LENS="${SEQ_LENS:-512,1024,2048,4096}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${MICRO:-4}}"
GRAD_ACCUM="${GRAD_ACCUM:-${GA:-16}}"
WARMUP="${WARMUP:-${REAL_WARMUP:-2}}"
ITERATIONS="${ITERATIONS:-3}"
PHANTORA_HOME="${PHANTORA_HOME:-/opt/phantora}"
PHANTORA_VRAM_MIB="${PHANTORA_VRAM_MIB:-240000}"
PHANTORA_IGNORE_CPU_TIME="${PHANTORA_IGNORE_CPU_TIME:-0}"
PHANTORA_TIMEOUT_S="${PHANTORA_TIMEOUT_S:-3600}"
INSTALL_HOST_PACKAGES="${INSTALL_HOST_PACKAGES:-0}"
# An explicit empty value disables the allocator override for an A/B run.
if [[ -v PYTORCH_CUDA_ALLOC_CONF ]]; then
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
else
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
CUDA_ALLOCATOR_ENV=()
if [ -n "${PYTORCH_CUDA_ALLOC_CONF}" ]; then
  CUDA_ALLOCATOR_ENV=("PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
fi
RUN_ID="${RUN_ID:-full-model-sdpa-shared-4gpu-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${PWD}/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"

if [ -n "${VENV:-}" ]; then
  PYTHON="${PYTHON:-${VENV}/bin/python}"
  TORCHRUN="${TORCHRUN:-${VENV}/bin/torchrun}"
else
  PYTHON="${PYTHON:-python}"
  TORCHRUN="${TORCHRUN:-torchrun}"
fi
PHANTORA_RUN="${PHANTORA_RUN:-${PHANTORA_HOME}/dist/phantora_run}"
PHANTORA_SERVER="${PHANTORA_SERVER:-${PHANTORA_HOME}/dist/phantora_server}"

die() { echo "error: $*" >&2; exit 1; }
require_file() { [ -f "$1" ] || die "missing file: $1"; }
require_exec() { [ -x "$1" ] || die "missing executable: $1"; }

[ "${GPUS}" = 4 ] || die "this handoff package is fixed to GPUS=4"
require_file "${TRAIN_SCRIPT}"
require_exec "${PYTHON:-missing}"
require_exec "${TORCHRUN:-missing}"
require_exec "${PHANTORA_RUN}"
require_exec "${PHANTORA_SERVER}"
require_file "${PHANTORA_HOME}/tests/docker/deepspeed/config_gen.py"

mkdir -p "${RAW_DIR}"
if [ "${INSTALL_HOST_PACKAGES}" = 1 ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libopenblas-dev
fi
PYTHON_BIN_DIR="$(dirname "${PYTHON}")"
TORCHRUN_BIN_DIR="$(dirname "${TORCHRUN}")"
SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
TORCH_LIB="$("${PYTHON}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
CUDA_LIBS="${CUDA_LIBS:-${TORCH_LIB}:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/home/ray/anaconda3/lib}"
SERVER_PID=""
SOCKET_DIR=""

cleanup_server() {
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  [ -z "${SOCKET_DIR}" ] || rm -rf "${SOCKET_DIR}"
}
trap cleanup_server EXIT

common_args() {
  printf '%s\0' \
    "${TRAIN_SCRIPT}" --model "${MODEL}" --attention-impl sdpa \
    --sequence-length "$1" --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --gradient-accumulation "${GRAD_ACCUM}" --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}"
}

extract_record() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

log_path, output_path = map(Path, sys.argv[1:3])
for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
    if "REPRO_RESULT " in line:
        record = json.loads(line.split("REPRO_RESULT ", 1)[1])
        output_path.open("a", encoding="utf-8").write(json.dumps(record, sort_keys=True) + "\n")
        break
else:
    raise SystemExit(f"missing REPRO_RESULT in {log_path}")
PY
}

run_real() {
  local seq="$1" log="${RAW_DIR}/real_seq${1}.log" rc
  mapfile -d '' -t args < <(common_args "${seq}")
  echo "real sdpa seq=${seq}" | tee -a "${RAW_DIR}/commands.log"
  set +e
  env -u LD_PRELOAD -u PHANTORA -u PHANTORA_NGPU -u PYTORCH_CUDA_ALLOC_CONF \
    PATH="${TORCHRUN_BIN_DIR}:${PATH}" \
    PYTHONPATH="${SITE_PACKAGES}:${PHANTORA_HOME}/tests:${PYTHONPATH:-}" \
    "${CUDA_ALLOCATOR_ENV[@]}" \
    LD_LIBRARY_PATH="${CUDA_LIBS}" \
    "${TORCHRUN}" --nproc_per_node="${GPUS}" --master_port="$((29500 + seq % 1000))" \
    "${args[@]}" --backend real >"${log}" 2>&1
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    tail -200 "${log}" >&2 || true
    return "${rc}"
  fi
  extract_record "${log}" "${OUT_DIR}/real.jsonl"
}

patch_netconfig_host() {
  "${PYTHON}" - "$1" "$(hostname)" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(re.sub(r"host_mapping = .*", f'host_mapping = ["{sys.argv[2]}"]', path.read_text()), encoding="utf-8")
PY
}

run_phantora() {
  local seq="$1" work_dir="${RAW_DIR}/sim_seq${1}_work" log="${RAW_DIR}/sim_seq${1}.log"
  local port socket_prefix host
  mapfile -d '' -t args < <(common_args "${seq}")
  SOCKET_DIR="/tmp/phantora-repro.${RUN_ID}.${seq}"
  socket_prefix="${SOCKET_DIR}/p"
  rm -rf "${work_dir}" "${SOCKET_DIR}"
  mkdir -p "${work_dir}/home" "${SOCKET_DIR}"
  cp "${PHANTORA_HOME}/tests/docker/deepspeed/config_gen.py" "${work_dir}/config_gen.py"
  (
    cd "${work_dir}"
    "${PYTHON}" config_gen.py --nhost 1 --ngpu "${GPUS}" --vram_mib "${PHANTORA_VRAM_MIB}"
  ) >"${log}" 2>&1
  patch_netconfig_host "${work_dir}/netconfig.toml"
  host="$(hostname)"
  printf '%s slots=%d\n' "${host}" "${GPUS}" >"${work_dir}/hostfile"

  echo "phantora sdpa seq=${seq}" | tee -a "${RAW_DIR}/commands.log"
  env -u PYTORCH_CUDA_ALLOC_CONF \
  CUDA_VISIBLE_DEVICES=0 \
  PHANTORA_SOCKET_PREFIX="${socket_prefix}" PHANTORA_LOG=info PHANTORA_USE_CUPTI=1 \
  PYTHONPATH="${SITE_PACKAGES}:${PHANTORA_HOME}/tests:${PYTHONPATH:-}" \
  VIRTUAL_ENV="${VENV:-}" PATH="${PYTHON_BIN_DIR}:${PATH}" \
  "${CUDA_ALLOCATOR_ENV[@]}" LD_LIBRARY_PATH="${CUDA_LIBS}" \
    "${PHANTORA_SERVER}" --netconfig "${work_dir}/netconfig.toml" >>"${log}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 90); do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${log}" >&2; die "phantora_server exited early"; }
    find "${SOCKET_DIR}" -maxdepth 1 \( -type s -o -type f \) 2>/dev/null | grep -q . && break
    sleep 1
  done

  port=$((29600 + seq % 1000))
  set +e
  env -u PYTORCH_CUDA_ALLOC_CONF \
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="${PHANTORA_HOME}/dist:${CUDA_LIBS}" LD_PRELOAD="${PHANTORA_HOME}/dist/libcuda.so.1" \
  PHANTORA_SOCKET_PREFIX="${socket_prefix}" PHANTORA=1 PHANTORA_NGPU="${GPUS}" \
  PHANTORA_VRAM_MIB="${PHANTORA_VRAM_MIB}" PHANTORA_IGNORE_CPU_TIME="${PHANTORA_IGNORE_CPU_TIME}" PHANTORA_USE_CUPTI=1 \
  PYTHONPATH="${SITE_PACKAGES}:${PHANTORA_HOME}/tests:${PYTHONPATH:-}" \
  VIRTUAL_ENV="${VENV:-}" PATH="${PYTHON_BIN_DIR}:${PATH}" HOME="${work_dir}/home" \
  "${CUDA_ALLOCATOR_ENV[@]}" \
    timeout "${PHANTORA_TIMEOUT_S}" "${PHANTORA_RUN}" deepspeed --no_ssh -H "${work_dir}/hostfile" \
      --num_nodes 1 --num_gpus "${GPUS}" --node_rank 0 --master_addr 127.0.0.1 --master_port "${port}" \
      "${args[@]}" --backend phantora >>"${log}" 2>&1
  local rc=$?
  set -e
  cleanup_server
  [ "${rc}" -eq 0 ] || { tail -200 "${log}" >&2; return "${rc}"; }
  extract_record "${log}" "${OUT_DIR}/sim.jsonl"
}

write_summary() {
  "${PYTHON}" - "${OUT_DIR}/real.jsonl" "${OUT_DIR}/sim.jsonl" "${OUT_DIR}/summary.md" <<'PY'
import json
import sys
from pathlib import Path

real = {r["sequence_length"]: r for r in map(json.loads, Path(sys.argv[1]).read_text().splitlines())}
sim = {r["sequence_length"]: r for r in map(json.loads, Path(sys.argv[2]).read_text().splitlines())}
lines = [
    "# Shared-Loop SDPA 4-GPU Result",
    "",
    "Both columns ran `full_model_zero3_repro.py`; only `--backend` differs.",
    "",
    "| seq | real tok/s | Phantora sim tok/s | sim/real | real step s | sim step s |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
]
for seq in sorted(real):
    r, s = real[seq], sim[seq]
    lines.append(f"| {seq} | {r['tokens_per_second']:,.0f} | {s['tokens_per_second']:,.0f} | {s['tokens_per_second']/r['tokens_per_second']:.2f} | {r['step_time_s']:.4g} | {s['step_time_s']:.4g} |")
Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

: >"${OUT_DIR}/real.jsonl"
: >"${OUT_DIR}/sim.jsonl"
for seq in ${SEQ_LENS//,/ }; do run_real "${seq}"; done
for seq in ${SEQ_LENS//,/ }; do run_phantora "${seq}"; done
write_summary
echo "results: ${OUT_DIR}/summary.md"
