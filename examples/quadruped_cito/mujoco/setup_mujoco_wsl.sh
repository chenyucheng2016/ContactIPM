#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
dependency_root="${repository_root}/.deps"
mujoco_target="${dependency_root}/mujoco-py311"
menagerie_target="${dependency_root}/mujoco_menagerie"
menagerie_commit="da76818e269b82289eba39808e2fb91d679d6994"
go1_target="${menagerie_target}/unitree_go1"
go1_raw_root="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/${menagerie_commit}/unitree_go1"

mkdir -p "${dependency_root}"
if [[ ! -f "${mujoco_target}/mujoco/libmujoco.so.3.9.0" ]]; then
    python3 -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --only-binary=:all: \
        --timeout 600 \
        --retries 10 \
        --target "${mujoco_target}" \
        mujoco==3.9.0
fi

mkdir -p "${go1_target}"
if [[ ! -f "${go1_target}/go1.xml" ]]; then
    curl -fsSL --retry 10 --max-time 600 \
        -o "${go1_target}/go1.xml" "${go1_raw_root}/go1.xml"
fi
if [[ ! -f "${go1_target}/scene.xml" ]]; then
    curl -fsSL --retry 10 --max-time 600 \
        -o "${go1_target}/scene.xml" "${go1_raw_root}/scene.xml"
fi

# Menagerie marks these geoms as non-colliding visual decoration. Removing
# them preserves the published inertias and collision model while avoiding a
# multi-megabyte mesh download for headless physics acceptance tests.
sed -i -e '/<mesh class="go1"/d' \
       -e '/<geom class="visual"/d' \
       "${go1_target}/go1.xml"

test -f "${mujoco_target}/mujoco/include/mujoco/mujoco.h"
test -f "${go1_target}/scene.xml"
printf 'MuJoCo 3.9.0 and Menagerie %s are ready.\n' "${menagerie_commit}"
