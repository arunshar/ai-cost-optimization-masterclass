#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="${1:-ai-cost-optimization-masterclass}"

cd "$REPO_DIR"

if ! gh auth status >/dev/null 2>&1; then
  echo "Logging into GitHub CLI..."
  gh auth login
fi

GH_USER="$(gh api user -q .login)"
echo "Authenticated as GitHub user: $GH_USER"

if ! gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
  echo "Creating public GitHub repository: $GH_USER/$REPO_NAME ..."
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
else
  echo "Repository $GH_USER/$REPO_NAME already exists; pushing main branch..."
  git branch -M main
  git push -u origin main --force
fi

echo "Enabling GitHub Pages (GitHub Actions workflow)..."
gh api -X POST "repos/$GH_USER/$REPO_NAME/pages" -f build_type="workflow" >/dev/null 2>&1 || \
gh api -X PUT "repos/$GH_USER/$REPO_NAME/pages" -f build_type="workflow" >/dev/null 2>&1 || true

echo "=================================================================="
echo "Repository URL: https://github.com/$GH_USER/$REPO_NAME"
echo "Public Website: https://$GH_USER.github.io/$REPO_NAME/"
echo "=================================================================="
