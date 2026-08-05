#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /absolute/path/to/acados" >&2
    exit 2
fi

acados_source="$(realpath "$1")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "$script_dir/../..")"
python_bin="${PYTHON_BIN:-python3.11}"

export ACADOS_SOURCE_DIR="$acados_source"
export PYTHONPATH="$acados_source/interfaces/acados_template:$repo_root/.deps/casadi-3.7.2:$repo_root/.deps/acados-py311${PYTHONPATH:+:$PYTHONPATH}"

cases=(
    "crisp cartpole_soft_walls"
    "crisp push_box"
    "crisp transport"
    "crisp push_t"
    "impact push_box"
    "impact push_t"
    "impact cart_transport"
)

for item in "${cases[@]}"; do
    read -r suite problem <<< "$item"
    name="contact_${suite}_${problem}"
    generated="$repo_root/.deps/acados_generated/$name"

    "$python_bin" "$script_dir/acados_contact_benchmarks.py" \
        prepare-linux --suite "$suite" --problem "$problem"

    mapfile -t sources < <(
        find "$generated" -type f -name '*.c' ! -name 'main_*' | sort
    )
    gcc -O2 -fPIC -std=c99 -shared "${sources[@]}" \
        -I"$acados_source/include" \
        -I"$acados_source/include/acados" \
        -I"$acados_source/include/blasfeo/include" \
        -I"$acados_source/include/hpipm/include" \
        -L"$acados_source/lib" \
        -Wl,-rpath,"$acados_source/lib" \
        -lacados -lhpipm -lblasfeo -lm \
        -o "$generated/libacados_ocp_solver_${name}.so"
done
