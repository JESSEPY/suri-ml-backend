#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 EC2 instance (t3.micro/t2.micro
# free tier). Run this once by hand on the instance (as the `ubuntu` user);
# after that, deploys happen via .github/workflows/deploy.yml (git pull +
# systemctl restart), not by re-running this script.
#
# Usage: ssh onto the instance, then:
#   curl -fsSL https://raw.githubusercontent.com/JESSEPY/suri-ml-backend/main/deploy/setup-ec2.sh | bash
# or copy the repo over first and run it from deploy/setup-ec2.sh directly.

set -euo pipefail

REPO_URL="https://github.com/JESSEPY/suri-ml-backend.git"
APP_DIR="/opt/suri-ml-backend"

sudo apt-get update
sudo apt-get install -y python3.12-venv git

if [ -d "$APP_DIR" ]; then
  echo "$APP_DIR already exists, skipping clone. Pull manually if you need to update it."
else
  sudo git clone "$REPO_URL" "$APP_DIR"
  sudo chown -R "$USER:$USER" "$APP_DIR"
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
