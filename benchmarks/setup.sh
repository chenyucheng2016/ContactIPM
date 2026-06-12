#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  acados environment setup for ContactIPM benchmarks
#  Run this ONCE after building acados, then activate for each session.
# ═══════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find acados root — try common locations
for candidate in \
    "$SCRIPT_DIR/../acados" \
    "$SCRIPT_DIR/../../acados" \
    "/mnt/c/Users/cheny/Documents/GitHub/acados"; do
    if [ -f "$candidate/lib/libacados.so" ]; then
        ACADOS_ROOT="$(cd "$candidate" && pwd)"
        break
    fi
done

if [ -z "$ACADOS_ROOT" ]; then
    echo "ERROR: acados not found. Build it first:"
    echo "  cd /mnt/c/Users/cheny/Documents/GitHub"
    echo "  git clone https://github.com/acados/acados.git && cd acados"
    echo "  git submodule update --recursive --init"
    echo "  mkdir build && cd build"
    echo "  cmake -DACADOS_INSTALL_DIR=.. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .."
    echo "  make install -j\$(nproc)"
    exit 1
fi

echo "acados root: $ACADOS_ROOT"
export ACADOS_SOURCE_DIR="$ACADOS_ROOT"
export LD_LIBRARY_PATH="$ACADOS_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ACADOS_ROOT/bin:${PATH:-}"

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "Installing Python packages..."
pip install -q -e "$ACADOS_ROOT/interfaces/acados_template" 2>/dev/null || \
    pip install -q "$ACADOS_ROOT/interfaces/acados_template"
pip install -q casadi numpy matplotlib

echo ""
python3 -c "from acados_template import AcadosOcp; print('  acados_template: OK')"
python3 -c "import casadi; print(f'  casadi: {casadi.__version__}')"
echo ""
echo "Setup complete! To activate for future sessions:"
echo "  source $VENV_DIR/bin/activate"
echo "  export ACADOS_SOURCE_DIR=$ACADOS_ROOT"
echo "  export LD_LIBRARY_PATH=$ACADOS_ROOT/lib:\$LD_LIBRARY_PATH"
