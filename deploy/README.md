# Deploying to EC2

One-time setup, then GitHub Actions handles every push to `main`.

## 1. Launch the instance

- Ubuntu 24.04, `t3.micro` (or `t2.micro`), free-tier eligible.
- Security group: allow port 22 from your IP only. Leave 8001 closed —
  `suri`'s Next.js server calls this service directly over its private/public
  IP via `ML_API_URL`, not through a browser, so it doesn't need to be public.
  Put it behind Nginx + TLS instead if you want a real domain.
- Allocate an Elastic IP and associate it, so `ML_API_URL` never changes
  across stop/start.

## 2. Bootstrap the instance (once)

SSH in as `ubuntu`, then run `deploy/setup-ec2.sh` (see its header comment
for the one-liner). It clones the repo to `/opt/suri-ml-backend`, creates a
venv, installs a systemd unit (`deploy/suri-ml.service`), and grants the
`ubuntu` user passwordless sudo scoped to *only*
`systemctl restart suri-ml` — nothing broader.

After it runs, fill in `/opt/suri-ml-backend/.env` with `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` (copy from `suri/.env.local` — shared Supabase
project), then `sudo systemctl restart suri-ml`.

## 3. GitHub Actions secrets

Set these in the repo's Settings → Secrets and variables → Actions:

- `EC2_HOST` — the Elastic IP.
- `EC2_SSH_KEY` — private key for a **dedicated deploy keypair**, not your
  personal one. Its public half must be in the `ubuntu` user's
  `~/.ssh/authorized_keys` on the instance.

Every push to `main` now runs `pytest`, then SSHes in, `git reset --hard`s
to `origin/main`, reinstalls deps, restarts the service, and curls
`/health` to confirm it came back up.

## Notes

- `main.py` loads the pinned model artifact at import time, so a restart
  briefly drops the service (a few seconds) — acceptable for this
  low-traffic internal endpoint.
- Retraining and pinning a new artifact version still works the same as it
  did for Render (see `.gitignore`'s comment above `ml/artifacts/*`); commit
  the new pinned directory and push to `main` to deploy it.
