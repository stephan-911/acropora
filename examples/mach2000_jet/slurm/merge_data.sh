# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Merge the 20 per-node part files written by gen-full into one
# samples_{Nx}.npy per resolution, in the same folder. I/O bound; the part
# files are kept, so the folder needs about 1.1 TB of free space.
#SBATCH --job-name=acropora_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_%j.out

: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

python3 run_postprocess.py --merge --data-dir "$DATA_DIR" \
    --resolutions 500 1000 2000 4000 \
    --samples 1000 --num-checkpoints 8 --total-nodes 20
ls -l "$DATA_DIR"/samples_{500,1000,2000,4000}.npy
echo "merge completed"
