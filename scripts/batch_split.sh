#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_script="${script_directory}/split_dataset.py"

if [[ ! -f "${python_script}" ]]; then
    echo "Error: Could not find '${python_script}'." >&2
    exit 1
fi

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <directory_1> [directory_2 ...]
       $(basename "$0") [OPTIONS] -f <file_with_directories>

Execute split_dataset.py across a list of directories.

Options:
  -f, --file FILE         Read directory paths from a file. Lines can follow 'directory,output_directory'
                          or just 'directory' (one per line, # comments ignored).
  -o, --output-dir DIR    Base output directory. Used when no explicit output directory is provided for a row.
  -c, --continue          Continue processing remaining directories if an error occurs.
  -h, --help              Show this help message.
  --                      Stop flag parsing; all subsequent arguments are passed to split_dataset.py.

All other flags (e.g. --dry-run, -m symlink, -e tif, --rule gallery=1-4) are forwarded
directly to split_dataset.py.
EOF
}

file_input=""
base_output_directory=""
continue_on_error=false
source_directories=()
destination_directories=()
forwarded_arguments=()

parse_directory_entry() {
    local raw_line="$1"
    local source_path=""
    local destination_path=""

    if [[ "${raw_line}" == *","* ]]; then
        source_path="${raw_line%%,*}"
        destination_path="${raw_line#*,}"
    else
        source_path="${raw_line}"
        destination_path=""
    fi

    source_path="$(echo "${source_path}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    destination_path="$(echo "${destination_path}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    if [[ -n "${source_path}" ]]; then
        source_directories+=("${source_path}")
        destination_directories+=("${destination_path}")
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -f|--file)
            if [[ $# -lt 2 ]]; then
                echo "Error: Option '$1' requires a file path." >&2
                exit 1
            fi
            file_input="$2"
            shift 2
            ;;
        -o|--output-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: Option '$1' requires a directory path." >&2
                exit 1
            fi
            base_output_directory="$2"
            shift 2
            ;;
        -c|--continue)
            continue_on_error=true
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                forwarded_arguments+=("$1")
                shift
            done
            break
            ;;
        -m|--mode|-p|--pattern|--rule|--unmatched)
            if [[ $# -lt 2 ]]; then
                echo "Error: Option '$1' requires an argument." >&2
                exit 1
            fi
            forwarded_arguments+=("$1" "$2")
            shift 2
            ;;
        --mode=*|--pattern=*|--rule=*|--unmatched=*|--extensions=*)
            forwarded_arguments+=("$1")
            shift
            ;;
        -e|--extensions)
            if [[ $# -lt 2 ]]; then
                echo "Error: Option '$1' requires an argument." >&2
                exit 1
            fi
            forwarded_arguments+=("$1" "$2")
            shift 2
            while [[ $# -gt 0 && ! "$1" =~ ^- && ! -d "$1" ]]; do
                forwarded_arguments+=("$1")
                shift
            done
            ;;
        -r|--recursive|--preserve-structure|--force|--dry-run)
            forwarded_arguments+=("$1")
            shift
            ;;
        -*)
            forwarded_arguments+=("$1")
            shift
            ;;
        *)
            parse_directory_entry "$1"
            shift
            ;;
    esac
done

if [[ -n "${file_input}" ]]; then
    if [[ ! -f "${file_input}" ]]; then
        echo "Error: Directory list file '${file_input}' not found." >&2
        exit 1
    fi
    while IFS= read -r line || [[ -n "${line}" ]]; do
        trimmed_line="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        if [[ -n "${trimmed_line}" && ! "${trimmed_line}" =~ ^# ]]; then
            parse_directory_entry "${trimmed_line}"
        fi
    done < "${file_input}"
fi

# Fallback to reading from stdin pipeline if no directory arguments were given
if [[ ${#source_directories[@]} -eq 0 && ! -t 0 ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        trimmed_line="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        if [[ -n "${trimmed_line}" && ! "${trimmed_line}" =~ ^# ]]; then
            parse_directory_entry "${trimmed_line}"
        fi
    done
fi

if [[ ${#source_directories[@]} -eq 0 ]]; then
    echo "Error: No target directories specified." >&2
    echo "" >&2
    print_usage >&2
    exit 1
fi

total_directories=${#source_directories[@]}
successful_count=0
failed_directories=()

echo "Starting batch split across ${total_directories} directories..."
if [[ ${#forwarded_arguments[@]} -gt 0 ]]; then
    echo "Forwarded options: ${forwarded_arguments[*]}"
fi
echo "------------------------------------------------------------"

for ((index = 0; index < total_directories; index++)); do
    target_directory="${source_directories[index]}"
    explicit_output_directory="${destination_directories[index]}"
    display_index=$((index + 1))

    if [[ -n "${explicit_output_directory}" ]]; then
        echo "[${display_index}/${total_directories}] Processing: ${target_directory} -> ${explicit_output_directory}"
    else
        echo "[${display_index}/${total_directories}] Processing: ${target_directory}"
    fi

    execution_arguments=("${python_script}" "${target_directory}")

    if [[ -n "${explicit_output_directory}" ]]; then
        execution_arguments+=("-o" "${explicit_output_directory}")
    elif [[ -n "${base_output_directory}" ]]; then
        folder_basename="$(basename "${target_directory}")"
        execution_arguments+=("-o" "${base_output_directory}/${folder_basename}_split")
    fi

    if [[ ${#forwarded_arguments[@]} -gt 0 ]]; then
        execution_arguments+=("${forwarded_arguments[@]}")
    fi

    if python3 "${execution_arguments[@]}"; then
        successful_count=$((successful_count + 1))
    else
        echo "Failed processing: ${target_directory}" >&2
        failed_directories+=("${target_directory}")
        if [[ "${continue_on_error}" = false ]]; then
            echo "Aborting remaining batch. Use --continue to ignore errors." >&2
            exit 1
        fi
    fi

    echo "------------------------------------------------------------"
done

echo "Batch execution finished."
echo "Summary: ${successful_count}/${total_directories} directories succeeded."

if [[ ${#failed_directories[@]} -gt 0 ]]; then
    echo "Failed directories (${#failed_directories[@]}):" >&2
    for failed_path in "${failed_directories[@]}"; do
        echo "  - ${failed_path}" >&2
    done
    exit 1
fi
