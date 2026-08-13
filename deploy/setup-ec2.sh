#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 EC2 instance (t3.micro/t2.micro
# free tier). Run this once by hand on the instance (as the `ubuntu` user);
# after that, deploys happen via .github/workflows/deploy.yml (git pull +
# systemctl restart), not by re-running this script.
#
# This repo is private, so plain `curl`ing the raw file (or an HTTPS clone)
# 404s -- GitHub returns 404, not 403, for unauthenticated access to private
# repos. Before running this script, generate a deploy key on the instance
# and add it (read-only) at github.com/JESSEPY/suri-ml-backend/settings/keys:
#   ssh-keygen -t ed25519 -C "suri-ml-ec2-deploy" -f ~/.ssh/id_ed25519 -N ""
#   cat ~/.ssh/id_ed25519.pub
#   ssh -T git@github.com   # accept the host key, confirms auth works
#
# Then clone with SSH and run this script from inside the clone:
#   git clone git@github.com:JESSEPY/suri-ml-backend.git /opt/suri-ml-backend
#   cd /opt/suri-ml-backend && bash deploy/setup-ec2.sh

set -euo pipefail

REPO_URL="git@github.com:JESSEPY/suri-ml-backend.git"
APP_DIR="/opt/suri-ml-backend"

sudo apt-get update
sudo apt-get install -y python3.12-venv git

if [ -d "$APP_DIR" ]; then
  echo "$APP_DIR already exists, skipping clone. Pull manually if you need to update it."
else
  # Clone as the invoking user, not root -- the SSH deploy key lives in
  # this user's ~/.ssh, and root has no access to it.
  sudo mkdir -p "$APP_DIR"
  sudo chown "$USER:$USER" "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "No .env found. Create $APP_DIR/.env with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
  echo "(copy the values from suri/.env.local -- the ML service shares suri's Supabase project)."
  cp .env.example .env
fi

sudo cp deploy/suri-ml.service /etc/systemd/system/suri-ml.service
sudo systemctl daemon-reload
sudo systemctl enable --now suri-ml

# Let the GitHub Actions deploy user restart the service without a password,
# but nothing else -- scope this narrowly rather than granting broad sudo.
DEPLOY_SUDOERS="/etc/sudoers.d/suri-ml-deploy"
if [ ! -f "$DEPLOY_SUDOERS" ]; then
  echo "$USER ALL=(ALL) NOPASSWD: /bin/systemctl restart suri-ml" | sudo tee "$DEPLOY_SUDOERS" >/dev/null
  sudo chmod 440 "$DEPLOY_SUDOERS"
fi

echo "Done. Edit $APP_DIR/.env if you haven't, then: sudo systemctl restart suri-ml"
echo "Check status: sudo systemctl status suri-ml"
echo "Check health: curl -s localhost:8001/health"
echo ""
echo "Not done yet -- see deploy/README.md:"
echo "  3. Generate a second deploy key so GitHub Actions can SSH in (EC2_HOST/EC2_SSH_KEY secrets)"
echo "  4. Set ML_API_KEY in .env before opening port 8001, or /predict is unauthenticated on the open internet"
