#!/usr/bin/env bash
# Install the repository's git hooks: point git at dev/hooks (works in every worktree, where .git is a file).
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
git -C "$root" config core.hooksPath dev/hooks
echo "git hooks: core.hooksPath = dev/hooks"
