# Running and rescuing RunPod jobs

Operator notes. The README covers what a job does; this is what to look at when one goes wrong.

## Controllers

Each pod has one controller, `backend/scripts/run_runpod_job.py`, started detached by the bot so a bot restart does not kill it. On restart the bot reattaches to any controller still running.

```
backend/logs/ctrl-<model>-<bpw>-<ts>.log     controller output, what the embed is parsed from
backend/logs/controllers/<id>.json           pid, log path, job id; read by /diag and on restart
```

The last line of a controller log is `[cost] <secs>s on <card> (<cloud>) at $<rate>/h = $<total>`, on success and on failure. Billing is timed from pod creation.

## On the pod

| Path | |
| --- | --- |
| `/root/quant.py` | the pod script, uploaded fresh each job over the baked copy |
| `/root/bq.log` | its log; the controller relays only lines with a known `[tag]` |
| `/root/bq-result.json` | the result the controller fetches at the end |
| `/quant/blockquant/model` | the unquantized download |
| `/quant/output-<bpw>bpw` | a finished quant, deleted once it uploads |

Pods are only reachable over SSH with the key in `~/.ssh/id_rsa`.

## When a job fails

- The pod is terminated on every exit path unless the controller ran with `--keep-pod`. Start a job with that by hand when you expect trouble.
- A quant that converted but did not upload is still in `/quant/output-<bpw>bpw` on a kept pod. `backend/scripts/rescue_upload.py --pod <id> --repo <org/name> --remote-dir /quant/output-<bpw>bpw --terminate` pushes it and tears the pod down.
- If the pod is already gone, the controller checks HF for the job's repo before calling it a failure, and reports the last progress line it saw.
- Self-calibration artifacts are cached to `<org>/<model>-exl3-selfcal` after each stage, so rerunning the same `/quant` starts at the stage that failed.

## Leaked pods

The bot reaps `bq-<pid>-*` pods whose controller is gone, every two minutes while it is up. `backend/scripts/pod_watchdog.py` is the backstop, from cron:

```
*/10 * * * * /root/blockquant/backend/venv/bin/python /root/blockquant/backend/scripts/pod_watchdog.py >> /root/watchdog.log 2>&1
```

It only reports unless run with `--enforce`, and keeps any pod it cannot prove abandoned.

## Choosing cards by hand

`--gpu auto` is the default and what the bot uses. `--gpu "<RunPod id>" --gpu-fallback "<id>,<id>"` pins cards instead, `--cloud COMMUNITY|SECURE` pins a pool, and `--max-price` takes a number for the pod's $/hr ceiling. Card prices and conversion speeds are in `backend/gpu_table.json`; add a measured speed there when a new card has run a job.
