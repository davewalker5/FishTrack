#!/usr/bin/env bash

# Clean video diagnostics without removing finished videos or the tracking environment
set -euo pipefail

# Initialise
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
dry_run=false
folder=$PROJECT_ROOT
folder_set=false

# Handle command line arguments
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        --help|-h)
            echo "Usage: $(basename "$0") [--dry-run] [video-folder]"
            exit 0 ;;
        -*) echo "Unknown option: $argument" >&2; exit 1 ;;
        *)
            if "$folder_set"; then
                echo "Specify only one video folder." >&2
                exit 1
            fi
            folder=$argument
            folder_set=true ;;
    esac
done

# Change to the processing folder and collect before deleting so traversal cannot race with directory removal
folder=$(cd "$folder" && pwd -P)
targets=()
while IFS= read -r -d '' target; do
    if [[ -f "$target" ]]; then

        # Ignore unrelated CSV files that merely share a statistics filename.
        IFS= read -r header < "$target" || continue
        if [[ "$header" != source_frame,seconds,* ]]; then
            continue
        fi
        target=$(dirname "$target")
    fi
    targets+=("$target")
done < <(find "$folder" \( -name .git -o -name venv -o -name .venv -o -name .runtime \
    -o -name models -o -name .cache -o -name __pycache__ \) -prune -o \
    -type d \( -name '*-diagnostics' -o -name '*-diagnostics-[0-9][0-9][0-9]' \
    -o -name '*-debug' \) -print0 -prune -o \
    -type f \( -name statistics.csv -o -name sam2-statistics.csv \) -print0)

# Iterate over the target files
for target in "${targets[@]-}"; do
    [[ -d "$target" ]] || continue

    # A statistics file at the search root does not authorize deleting the whole folder
    if [[ "$target" == "$folder" ]]; then
        echo "Preserving search root: $target"
        continue
    fi

    # Show what would be removed (dry run) or remove the folder
    if "$dry_run"; then
        echo "Would remove: $target"
    else
        echo "Removing: $target"
        rm -rf -- "$target"
    fi
done

if "$dry_run"; then
    echo "Preview complete; no files removed."
else
    echo "Video diagnostics cleanup complete; models and environments preserved."
fi
