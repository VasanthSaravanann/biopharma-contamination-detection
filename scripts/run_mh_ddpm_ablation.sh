#!/usr/bin/env bash
set -euo pipefail

SEED=${1:-42}
N=${2:-100}
OUT=${3:-output/augmented/mh_ddpm}

mkdir -p "$OUT"
echo "Running MH-DDPM ablation: seed=$SEED n=$N out=$OUT"
python - <<PY
from pathlib import Path
import numpy as np
Path('$OUT').mkdir(parents=True, exist_ok=True)
np.random.seed($SEED)
with open(Path('$OUT')/'README.txt','w') as f:
    f.write('MH-DDPM ablation placeholder. Generated %d samples with seed %d\n' % ($N, $SEED))
print('Wrote placeholder augmented files to', Path('$OUT'))
PY
