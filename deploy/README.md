# Deploying to EC2

One-time setup, then GitHub Actions handles every push to `main`. This
mirrors exactly the steps used for the current live instance (N. Virginia,
`t3.micro`, IP `54.206.66.57` at time of writing) — follow in order if
rebuilding from scratch.

## 1. Launch the instance

- AMI: **Ubuntu Server 24.04 LTS (HVM)**, `x86_64`, marked "Free tier
  eligible" on the card.
- Instance type: `t3.micro` (2 vCPU, 1 GiB memory) — or `t2.micro` where
  `t3` isn't offered.
- Key pair: create a new one (e.g. `suri-ml-access`), download the
  `.pem`. This is **your personal** access key, separate from the two
  deploy keys set up below.
- Security group — create new, two inbound rules:
  - SSH (22): source **Anywhere-IPv4 (`0.0.0.0/0`)**. Starting this
    scoped to "My IP" seems safer, but it silently breaks GitHub Actions'
    deploy step later — Actions runners connect from AWS's own rotating
    IP pool, not yours. Widening to `0.0.0.0/0` is fine in practice
    since SSH still requires a valid private key; there's no password
    auth to brute-force.
  - Custom TCP (8001): source **Anywhere-IPv4 (`0.0.0.0/0`)**. `suri`'s
    Next.js app runs as Vercel serverless functions (not in the same
    AWS network), so `ML_API_URL` has to reach this instance over the
    public internet — it can't stay closed. This is exactly why the
    `ML_API_KEY` check in step 5 exists: an open port with an
    unauthenticated endpoint behind it would let anyone spam `/predict`.
- Storage: leave the 8 GiB gp3 default.
- Launch, then **Elastic IPs** → allocate one → associate it with the
  instance, so `ML_API_URL` never changes across stop/start.

## 2. Set up the repo-read deploy key, then bootstrap

This repo is **private**, so a plain `curl`/HTTPS clone from the instance
404s (GitHub returns 404, not 403, for unauthenticated access to private
repos — hit this live). SSH into the instance as `ubuntu`, then:

```bash
ssh-keygen -t ed25519 -C "suri-ml-ec2-deploy" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Add the printed key at
`github.com/JESSEPY/suri-ml-backend/settings/keys` → **Add deploy key**,
leaving "Allow write access" unchecked (read-only). Then confirm it works
and clone:

```bash
ssh -T git@github.com   # type yes at the host-key prompt
git clone git@github.com:JESSEPY/suri-ml-backend.git /opt/suri-ml-backend
```

Now run `deploy/setup-ec2.sh` from inside the clone (it detects the
directory already exists and skips straight to setup):

```bash
cd /opt/suri-ml-backend
bash deploy/setup-ec2.sh
```

This creates a venv, installs dependencies, installs the systemd unit
(`deploy/suri-ml.service`), and grants the `ubuntu` user passwordless
sudo scoped to *only* `systemctl restart suri-ml` — nothing broader.

After it runs, fill in `/opt/suri-ml-backend/.env` with `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` (copy from `suri/.env.local` — shared
Supabase project):

```bash
echo 'SUPABASE_URL=...' >> /opt/suri-ml-backend/.env
echo 'SUPABASE_SERVICE_ROLE_KEY=...' >> /opt/suri-ml-backend/.env
sudo systemctl restart suri-ml
curl -s localhost:8001/health   # expect {"status":"ok",...,"supabase_reachable":true}
```

## 3. Set up the second deploy key, for GitHub Actions → EC2

This is a **separate** keypair from step 2 — that one lets the instance
pull *from* GitHub; this one lets GitHub Actions SSH *into* the instance
to run the deploy script. Still on the instance:

```bash
ssh-keygen -t ed25519 -f ~/gh-actions-deploy -N ""
cat ~/gh-actions-deploy.pub >> ~/.ssh/authorized_keys
cat ~/gh-actions-deploy      # copy the whole private key block, BEGIN/END lines included
```

In `github.com/JESSEPY/suri-ml-backend` → **Settings → Secrets and
variables → Actions → New repository secret**, add:

- `EC2_HOST` = the Elastic IP (e.g. `54.206.66.57`) — **not** the
  instance's Name tag, the actual address used to `ssh ubuntu@<this>`.
- `EC2_SSH_KEY` = the private key printed above.

Every push to `main` now runs `pytest`, then SSHes in, `git reset --hard`s
to `origin/main`, reinstalls deps, restarts the service, and polls
`/health` for up to 30s to confirm it came back up (see `Notes` below on
why it polls instead of a fixed sleep).

## 4. Require an API key on `/predict`

Port 8001 is open to the internet (step 1), and `/predict` has no auth of
its own beyond this check — so this step is what actually keeps it from
being a free-for-all once the port is open.

Generate a key and set it on the instance:

```bash
openssl rand -hex 32
echo 'ML_API_KEY=<paste the generated value>' >> /opt/suri-ml-backend/.env
cd /opt/suri-ml-backend && git pull   # pulls the auth-check code itself, if not already present
sudo systemctl restart suri-ml
```

Verify it's actually enforced (run from *outside* the instance, e.g. your
laptop, once port 8001 is open — not `localhost`):

```bash
# No key -- expect 401
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://<ELASTIC_IP>:8001/predict -d '{}'
# Correct key -- expect 200
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://<ELASTIC_IP>:8001/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <the same value>" \
  -d '{"crop":"Corn","data_source":"iot","ph":6.4}'
```

**The same key also needs to go into `suri`'s Vercel project** (Settings
→ Environment Variables) as `ML_API_KEY`, alongside `ML_API_URL` set to
`http://<ELASTIC_IP>:8001` — then trigger a redeploy, since env var
changes don't apply to an already-built deployment. Without this half,
`suri`'s `app/api/ml-fertilizer/route.ts` sends no key and every
`/predict` call gets rejected with 401, which surfaces to users as
"Fertilizer prediction is unavailable right now."

If `ML_API_KEY` is left unset on the instance, `/predict` silently stays
open (see the comment in `main.py` above `require_api_key`) — this is
intentional for local dev/tests, but means step 4 is easy to forget when
rebuilding the instance from scratch. Don't skip it once port 8001 is open.

## Notes

- `main.py` loads the pinned model artifact at import time, so a restart
  briefly drops the service (a few seconds, occasionally longer — cold
  imports of pandas/xgboost/scikit-learn on a `t3.micro` have taken over
  3s in practice, which is why the Actions workflow polls `/health`
  rather than using a fixed `sleep`).
- Retraining and pinning a new artifact version still works the same as it
  did for Render (see `.gitignore`'s comment above `ml/artifacts/*`); commit
  the new pinned directory and push to `main` to deploy it.
- Cost, running 24/7: ~$9.60/mo (`t3.micro`) + ~$3.65/mo (Elastic IP,
  billed whether the instance is running or stopped, as long as it's
  associated) + ~$0.65/mo (8 GiB gp3) ≈ **$13.90/month**. Stopping the
  instance when not in use drops the compute line but not the EIP line.
