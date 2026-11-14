#!/usr/bin/env bash
# Compute the next release version from the release:* labels of a pull request.
#
# Usage:   next_version.sh "<comma-separated PR labels>"
# Output:  "<bump> <last version> <next version>"   e.g. "minor v0.1.0 v0.2.0"
#          nothing if the PR has no release:* label (no release).
#
# Used by .github/workflows/ci.yaml (preview) and mlops.yaml (real release),
# so both always compute the same version. Needs GH_TOKEN and GH_REPO.

set -euo pipefail

LABELS=",${1:-},"

# Highest label wins if several are set
case "$LABELS" in
  *,release:major,*) BUMP=major ;;
  *,release:minor,*) BUMP=minor ;;
  *,release:patch,*) BUMP=patch ;;
  *)                 exit 0 ;;
esac

# Last published release (v0.0.0 if there is none yet)
LAST=$(gh release view --json tagName --jq .tagName 2>/dev/null || echo "v0.0.0")
IFS=. read -r MAJOR MINOR PATCH <<< "${LAST#v}"

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
esac

echo "${BUMP} ${LAST} v${MAJOR}.${MINOR}.${PATCH}"
