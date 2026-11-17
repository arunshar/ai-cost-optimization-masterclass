#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export SSL_CERT_FILE="/etc/ssl/cert.pem"
export CURL_CA_BUNDLE="/etc/ssl/cert.pem"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
REPO_NAME="${1:-ai-cost-optimization-masterclass}"

cd "$REPO_DIR"

if [ -f "$PARENT_DIR/.gh_token" ]; then
  export GH_TOKEN="$(tr -d '\r\n ' < "$PARENT_DIR/.gh_token")"
fi

if [ -z "${GH_TOKEN:-}" ] && ! gh auth status >/dev/null 2>&1; then
  echo "Logging into GitHub CLI..."
  gh auth login
fi

# Prefer HTTPS git over SSH. This machine's gh default protocol is SSH, but
# publickey auth often fails here.
gh auth setup-git >/dev/null 2>&1 || true

GH_USER="$(gh api user -q .login)"
echo "Authenticated as GitHub user: $GH_USER"

HTTPS_REMOTE="https://github.com/${GH_USER}/${REPO_NAME}.git"

ensure_https_origin() {
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$HTTPS_REMOTE"
  else
    git remote add origin "$HTTPS_REMOTE"
  fi
}

git branch -M main

if ! gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
  echo "Creating public GitHub repository: $GH_USER/$REPO_NAME ..."
  # Create without --push/--source so we control HTTPS remote ourselves.
  gh repo create "$REPO_NAME" --public --description "AI Cost Optimization Masterclass"
  ensure_https_origin
  git push -u origin main
else
  echo "Repository $GH_USER/$REPO_NAME already exists; pushing main branch..."
  ensure_https_origin
  git push -u origin main
fi

echo "Enabling GitHub Pages (GitHub Actions workflow)..."
PAGES_BODY='{"build_type":"workflow"}'
if ! printf '%s\n' "$PAGES_BODY" | gh api -X POST "repos/$GH_USER/$REPO_NAME/pages" --input - >/dev/null 2>&1; then
  printf '%s\n' "$PAGES_BODY" | gh api -X PUT "repos/$GH_USER/$REPO_NAME/pages" --input - >/dev/null 2>&1 || true
fi

echo "=================================================================="
echo "Repository URL: https://github.com/$GH_USER/$REPO_NAME"
echo "Public Website: https://$GH_USER.github.io/$REPO_NAME/"
echo "Custom domain : https://arunshar.com/$REPO_NAME/"
echo "=================================================================="
