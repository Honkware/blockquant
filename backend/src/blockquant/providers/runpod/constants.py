"""RunPod provider paths."""

# Remote paths — kept under /root so they survive bootstrap but are pod-local.
REMOTE_SCRIPT = "/root/quant.py"
REMOTE_LOG = "/root/bq.log"
REMOTE_RESULT = "/root/bq-result.json"
REMOTE_RUNNER = "/root/bq-run.sh"
BOOTSTRAP_MARKER = "/root/.bq-bootstrapped"
