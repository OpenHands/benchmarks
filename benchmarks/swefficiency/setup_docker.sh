#!/usr/bin/env bash
# pin-dockerd-cpu-mem-auto.sh
# Pin dockerd to the vCPUs corresponding to the FIRST 4 PHYSICAL CORES (all their sibling threads)
# and cap its memory using systemd drop-ins.
# Optional: also apply to containerd with --also-containerd
#
# Usage:
#   sudo ./pin-dockerd-cpu-mem-auto.sh [--also-containerd] [MEM_MAX] [MEM_HIGH]
# Defaults:
#   MEM_MAX = 1G
#   MEM_HIGH = 800M

set -euo pipefail

also_containerd="no"
MEM_MAX="1G"
MEM_HIGH="800M"

# Parse args
for arg in "$@"; do
  case "$arg" in
    --also-containerd) also_containerd="yes" ;;
    *)
      if [[ "$MEM_MAX" == "1G" ]]; then
        MEM_MAX="$arg"
      elif [[ "$MEM_HIGH" == "800M" ]]; then
        MEM_HIGH="$arg"
      else
        echo "Too many positional args. Usage: $0 [--also-containerd] [MEM_MAX] [MEM_HIGH]"
        exit 1
      fi
      ;;
  esac
done

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1"; exit 1; }; }
require nproc
require taskset

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)."
  exit 1
fi

TOTAL=$(nproc --all)
(( TOTAL > 0 )) || { echo "Could not determine CPU count."; exit 1; }

join_by() { local IFS="$1"; shift; echo "$*"; }

# --- Topology detection: pick first 4 physical cores (by first-seen logical CPU),
# then collect ALL logical CPUs (threads) that belong to those cores.
detect_cpu_list_for_first4_cores() {
  local selected_pairs=()     # array of "CORE,SOCKET"
  local -A picked=()          # set of selected pairs
  local rows=()               # lines "cpu,core,socket"
  local cpus_for_selected=()  # resulting logical CPU list

  if command -v lscpu >/dev/null 2>&1; then
    # Format: CPU,CORE,SOCKET (CSV). Ignore comments.
    while IFS= read -r line; do
      [[ "$line" =~ ^# ]] && continue
      [[ -z "$line" ]] && continue
      rows+=("$line")
      IFS=',' read -r cpu core socket <<<"$line"
      local key="${core},${socket}"
      if [[ ${#selected_pairs[@]} -lt 4 && -z "${picked[$key]:-}" ]]; then
        selected_pairs+=("$key")
        picked["$key"]=1
      fi
    done < <(lscpu -p=CPU,CORE,SOCKET)
  else
    # Fallback via sysfs
    # Build rows with cpu,core,socket (socket via physical_package_id, fallback 0)
    for cpu_path in /sys/devices/system/cpu/cpu[0-9]*; do
      [[ -e "$cpu_path/topology/core_id" ]] || continue
      cpu=${cpu_path##*/cpu}
      core=$(<"$cpu_path/topology/core_id")
      if [[ -r "$cpu_path/topology/physical_package_id" ]]; then
        socket=$(<"$cpu_path/topology/physical_package_id")
      else
        socket=0
      fi
      rows+=("${cpu},${core},${socket}")
      local key="${core},${socket}"
      if [[ ${#selected_pairs[@]} -lt 4 && -z "${picked[$key]:-}" ]]; then
        selected_pairs+=("$key")
        picked["$key"]=1
      fi
    done
    # Sort rows by CPU asc (ensure stable order)
    IFS=$'\n' rows=($(printf '%s\n' "${rows[@]}" | sort -t',' -k1,1n))
  fi

  if [[ ${#selected_pairs[@]} -eq 0 ]]; then
    echo "Failed to detect CPU topology." >&2
    return 1
  fi

  # Now collect all logical CPUs that belong to the selected cores
  for row in "${rows[@]}"; do
    IFS=',' read -r cpu core socket <<<"$row"
    key="${core},${socket}"
    if [[ -n "${picked[$key]:-}" ]]; then
      cpus_for_selected+=("$cpu")
    fi
  done

  # Deduplicate + sort numeric
  IFS=$'\n' read -r -d '' -a cpus_for_selected < <(printf '%s\n' "${cpus_for_selected[@]}" | sort -n | uniq && printf '\0')
  join_by , "${cpus_for_selected[@]}"
}

CPU_LIST="$(detect_cpu_list_for_first4_cores)"
(( $(wc -c <<<"$CPU_LIST") > 1 )) || { echo "Could not compute CPU list."; exit 1; }

echo "Detected ${TOTAL} CPUs."
echo "Selected physical cores: first 4 by topology; logical CPUs: ${CPU_LIST}"
echo "MemoryHigh=${MEM_HIGH}, MemoryMax=${MEM_MAX}"

# Pin currently running dockerd immediately
if pgrep -x dockerd >/dev/null 2>&1; then
  mapfile -t DKPIDS < <(pgrep -x dockerd)
  for p in "${DKPIDS[@]}"; do
    echo "Applying taskset ${CPU_LIST} to dockerd PID ${p}..."
    taskset -pc "${CPU_LIST}" "${p}" >/dev/null
  done
else
  echo "dockerd not running right now; will still configure persistence."
fi

# Helper: find unit owning dockerd
detect_unit_for_pid() {
  local pid="$1"
  local line
  line=$(systemctl status "$pid" 2>/dev/null | sed -n '1p' || true)
  if [[ "$line" =~ ([^[:space:]]+\.service) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

find_dockerd_unit() {
  local unit=""
  if command -v systemctl >/dev/null 2>&1; then
    if pid=$(pgrep -x dockerd | head -n1); then
      if unit=$(detect_unit_for_pid "$pid"); then
        echo "$unit"; return 0
      fi
    fi
    local candidates=(
      docker.service
      docker.io.service
      moby-engine.service
      snap.docker.dockerd.service
      dockerd.service
    )
    for u in "${candidates[@]}"; do
      if systemctl list-unit-files | awk '{print $1}' | grep -qx "$u"; then
        echo "$u"; return 0
      fi
    done
  fi
  return 1
}

apply_dropin() {
  local unit="$1"
  local dir="/etc/systemd/system/${unit}.d"
  mkdir -p "$dir"
  cat > "${dir}/cpu-mem-limits.conf" <<EOF
# Auto-generated by pin-dockerd-cpu-mem-auto.sh
[Service]
# CPU: confine daemon to logical CPUs that belong to the first 4 physical cores
CPUAffinity=
CPUAffinity=${CPU_LIST}

# Memory: cgroup v2 semantics (works on most modern distros)
MemoryAccounting=yes
MemoryHigh=${MEM_HIGH}
MemoryMax=${MEM_MAX}

# Optional: uncomment to forbid swap if memory.swap controller is enabled
# MemorySwapMax=0
EOF
  echo "Wrote drop-in: ${dir}/cpu-mem-limits.conf"
}

restart_and_verify() {
  local unit="$1"
  systemctl daemon-reload
  if systemctl try-restart "$unit"; then
    echo "Restarted ${unit}."
  else
    echo "Warning: could not restart ${unit}. Limits will apply after next restart."
  fi

  echo "--- Verification for ${unit} ---"
  systemctl show -p MemoryCurrent -p MemoryHigh -p MemoryMax "$unit"
  if pidof dockerd >/dev/null; then
    local pid
    pid=$(pidof dockerd | awk '{print $1}')
    echo "dockerd PID: ${pid}"
    grep -i '^Cpus_allowed_list:' "/proc/${pid}/status" || true
  fi
}

if command -v systemctl >/dev/null 2>&1; then
  if DOCKER_UNIT=$(find_dockerd_unit); then
    echo "Detected dockerd unit: ${DOCKER_UNIT}"
    apply_dropin "${DOCKER_UNIT}"
    restart_and_verify "${DOCKER_UNIT}"
  else
    echo "Could not detect a systemd unit for dockerd."
    echo "Creating a oneshot to re-apply CPU affinity at boot (memory caps require a proper unit)."

    cat > /etc/systemd/system/pin-dockerd-affinity.service <<'EOF'
[Unit]
Description=Pin dockerd CPU affinity at boot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'set -e; CPU_LIST="__CPU_LIST__"; if pgrep -x dockerd >/dev/null; then for p in $(pgrep -x dockerd); do /usr/bin/taskset -pc "${CPU_LIST}" "$p"; done; fi'

[Install]
WantedBy=multi-user.target
EOF
    sed -i "s/__CPU_LIST__/${CPU_LIST}/" /etc/systemd/system/pin-dockerd-affinity.service
    systemctl daemon-reload
    systemctl enable --now pin-dockerd-affinity.service
    echo "Enabled CPU-affinity oneshot. Note: without a dockerd unit, memory limits cannot be persisted via systemd."
  fi

  if [[ "$also_containerd" == "yes" ]] && systemctl list-unit-files | awk '{print $1}' | grep -qx 'containerd.service'; then
    echo "Applying the same limits to containerd.service"
    apply_dropin "containerd.service"
    restart_and_verify "containerd.service"
  fi
else
  echo "systemd not present; applied immediate CPU pin with taskset."
  echo "Persistent memory limits require cgroups managed by systemd or an equivalent init. Please migrate to a systemd-managed Docker setup if you need persistent MemoryHigh/Max for dockerd."
fi

echo "Done. This constrains the daemon (dockerd). Containers still need their own limits (e.g., --cpuset-cpus, -m/--memory)."
