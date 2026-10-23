#!/usr/bin/env bash
# Install the repository's git hooks.
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ln -sf ../../dev/hooks/pre-commit "$root/.git/hooks/pre-commit"
echo "installed pre-commit hook"
