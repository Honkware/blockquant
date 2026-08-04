#!/bin/bash
# BlockQuant pod entrypoint.
#
# RunPod normally injects PUBLIC_KEY into the env and starts its own sshd
# when the pod is created with start_ssh=True. This entrypoint just does
# the same setup defensively so the image works whether RunPod's
# template-side SSH machinery ran or not.
set -euo pipefail

# Some hosts pair a CUDA forward-compat libcuda (/usr/local/cuda/compat) with
# an older host driver; the compat stub then wins the linker search and every
# CUDA init fails with error 804, so torch reports no GPU on a machine that
# plainly has one. Prefer the real driver when the compat one is unusable.
if [ -d /usr/local/cuda/compat ] && ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[entrypoint] CUDA recovered by preferring the host driver over /usr/local/cuda/compat"
  else
    echo "[entrypoint] WARNING: CUDA still unavailable; jobs on this pod will fail"
  fi
fi

# Set up SSH from PUBLIC_KEY if RunPod's machinery hasn't already.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

# Start sshd if it isn't running yet.
if ! pgrep -x sshd > /dev/null; then
  service ssh start || /usr/sbin/sshd -D &
fi

echo "[blockquant] pod ready · python=$(python --version 2>&1) · torch=$(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[blockquant] quant.py is at /opt/blockquant/quant.py"
echo "[blockquant] waiting for SSH-driven quant — sleeping forever"

# Keep the container alive for the local provider to SSH in.
exec sleep infinity
