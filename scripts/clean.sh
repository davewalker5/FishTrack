#!/usr/bin/env bash
# Remove local generated artifacts. Run with --dry-run to preview the same targets.
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$project_root"
dry_run=false
case "${1:-}" in
    --dry-run) dry_run=true ;;
    --help|-h)
        echo "Usage: ./scripts/clean.sh [--dry-run]"
        echo "Removes local environments, models, runtime, caches, builds and evaluation results."
        echo "Also removes untracked *-diagnostics[-NNN] directories inside this checkout."
        echo "Preserves tracked files, example movies, external outputs and shared system caches."
        exit 0 ;;
    "") ;;
    *) echo "Usage: ./scripts/clean.sh [--dry-run]" >&2; exit 1 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Usage: ./scripts/clean.sh [--dry-run]" >&2
    exit 1
fi
command -v git >/dev/null 2>&1 || { echo "Git is required to protect tracked files." >&2; exit 1; }
[[ -f pyproject.toml && -d src/fishtrack ]] || { echo "Not a FishTrack checkout." >&2; exit 1; }
git ls-files >/dev/null

remove_generated() {
    local target=$1
    [[ -e "$target" || -L "$target" ]] || return 0
    # Refuse the whole target if any files in it belong to the repository.
    if [[ -n $(git ls-files -- "$target") ]]; then
        echo "Preserving $target (contains tracked files)"
        return 0
    fi
    if "$dry_run"; then
        echo "Would remove: $target"
    else
        echo "Removing: $target"
        rm -rf -- "$target"
    fi
}

for target in venv .venv .runtime models .cache build dist .pytest_cache .ruff_cache \
    .mypy_cache .hypothesis htmlcov .coverage data/evaluation; do
    remove_generated "$target"
done

# Do not follow symlinks or descend into Git metadata, environments or model caches.
while IFS= read -r -d '' target; do
    remove_generated "${target#./}"
done < <(find . \( -name .git -o -name venv -o -name .venv -o -name .runtime \
    -o -name models -o -name .cache \) -prune -o \
    -type d \( -name '*-diagnostics' -o -name '*-diagnostics-[0-9][0-9][0-9]' \
    -o -name '*.egg-info' \) -print0 -prune)

if "$dry_run"; then
    echo "Preview complete; no files removed."
else
    echo "Cleanup complete. Run ./scripts/make-venv.sh to prepare FishTrack again."
fi
