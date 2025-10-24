#!/bin/bash
#
# One-shot setup and full pipeline runner for this repo.
# - Creates venv and installs requirements
# - Downloads CIFAR-10-C
# - Extracts clean & corrupted features for all models
# - Trains linear probe and runs k-NN on clean
# - Evaluates robustness on CIFAR-10-C (linear + k-NN)
# - Generates per-model and cross-model plots
#

set -e

# Resolve repo root as the directory containing this script
REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

echo "[run_all] Repo root: $REPO_ROOT"

# 1) Python env & dependencies
VENV_DIR="$REPO_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[run_all] Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python --version

echo "[run_all] Installing requirements"
pip install --upgrade pip wheel
pip install -r "$REPO_ROOT/requirements.txt"

# 2) Paths and models
export PYTHONPATH="$REPO_ROOT/src"
DATA_DIR="$REPO_ROOT/data"
FEATURES_DIR="$REPO_ROOT/features"
EXPER_DIR="$REPO_ROOT/experiments"
RESULTS_DIR="$REPO_ROOT/results"
RESULTS_CORR_DIR="$RESULTS_DIR/corrupted"
PLOTS_DIR="$RESULTS_DIR/plots"
PLOTS_CMP_DIR="$RESULTS_DIR/plots_compare"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$DATA_DIR" "$FEATURES_DIR" "$EXPER_DIR" "$RESULTS_DIR" "$RESULTS_CORR_DIR" "$PLOTS_DIR" "$PLOTS_CMP_DIR" "$LOG_DIR"

MODELS=(
    "vit_base_patch14_dinov2.lvd142m"
    "vit_base_patch16_clip_224.openai"
    "vit_base_patch16_224.mae"
)

# 3) Download CIFAR-10-C (clean CIFAR-10 will be auto-downloaded by torchvision)
echo "[run_all] Ensuring CIFAR-10-C is present"
bash "$REPO_ROOT/download_cifar10c.sh"

# 4) CLEAN: extract features, train linear, run k-NN
for model in "${MODELS[@]}"; do
    echo "===================================================="
    echo "[run_all] CLEAN: Extracting features for $model"
    echo "===================================================="
    python "$REPO_ROOT/src/extract_features.py" \
        --model "$model" \
        --dataset cifar10 \
        --data_root "$DATA_DIR" \
        --out_dir "$FEATURES_DIR" \
        --batch_size 256 \
        --num_workers 4

    echo "[run_all] CLEAN: Training linear probe for $model"
    python "$REPO_ROOT/src/train_linear.py" \
        --model "$model" \
        --in_dir "$FEATURES_DIR" \
        --out_dir "$EXPER_DIR" \
        --scale \
        --refit_on_full \
        --Cs 0.1 1.0 10.0

    echo "[run_all] CLEAN: k-NN baseline for $model"
    python "$REPO_ROOT/src/eval_knn.py" \
        --model "$model" \
        --in_dir "$FEATURES_DIR" \
        --out_dir "$EXPER_DIR" \
        --normalize \
        --use_val_db \
        --k 1 5 20
done

# 5) CORRUPTED: extract features
for model in "${MODELS[@]}"; do
    echo "===================================================="
    echo "[run_all] C10-C: Extracting corrupted features for $model"
    echo "===================================================="
    python "$REPO_ROOT/src/extract_features.py" \
        --model "$model" \
        --dataset cifar10c \
        --data_root "$DATA_DIR" \
        --out_dir "$FEATURES_DIR" \
        --batch_size 256 \
        --num_workers 4
done

# 6) Robustness eval: linear + k-NN, write consolidated results
for model in "${MODELS[@]}"; do
    echo "[run_all] C10-C: Evaluating robustness (linear) for $model"
    python "$REPO_ROOT/src/eval_corruptions.py" \
        --model "$model" \
        --in_dir "$FEATURES_DIR" \
        --out_dir "$RESULTS_CORR_DIR" \
        --probe linear \
        --clean_metrics "$EXPER_DIR/$model/results_linear.json" \
        --clf_path "$EXPER_DIR/$model/linear_clf.joblib"

    echo "[run_all] C10-C: Evaluating robustness (k-NN) for $model"
    python "$REPO_ROOT/src/eval_corruptions.py" \
        --model "$model" \
        --in_dir "$FEATURES_DIR" \
        --out_dir "$RESULTS_CORR_DIR" \
        --probe knn \
        --clean_metrics "$EXPER_DIR/$model/results_knn.json" \
        --k 20 \
        --metric cosine \
        --normalize \
        --use_val_db
done

# 7) Per-model plots
for model in "${MODELS[@]}"; do
    echo "[run_all] Plotting per-model figures for $model"
    python "$REPO_ROOT/src/plotting.py" \
        --results \
        "$RESULTS_CORR_DIR/${model}_linear_robustness.json" \
        "$RESULTS_CORR_DIR/${model}_knn_robustness.json" \
        --out_dir "$PLOTS_DIR"
done

# 8) Cross-model comparison plots
echo "[run_all] Plotting cross-model comparison figures (linear)"
LIN_JSONS=()
for m in "${MODELS[@]}"; do LIN_JSONS+=("$RESULTS_CORR_DIR/${m}_linear_robustness.json"); done
python "$REPO_ROOT/src/plot_compare.py" --results "${LIN_JSONS[@]}" --out_dir "$PLOTS_CMP_DIR/linear"

echo "[run_all] Plotting cross-model comparison figures (k-NN)"
KNN_JSONS=()
for m in "${MODELS[@]}"; do KNN_JSONS+=("$RESULTS_CORR_DIR/${m}_knn_robustness.json"); done
python "$REPO_ROOT/src/plot_compare.py" --results "${KNN_JSONS[@]}" --out_dir "$PLOTS_CMP_DIR/knn"

echo "[run_all] All done. Outputs:"
echo " - Features: $FEATURES_DIR/<model>"
echo " - Clean experiment summaries: $EXPER_DIR/<model>/{results_linear.json,results_knn.json}"
echo " - Robustness results: $RESULTS_CORR_DIR/<model>_{linear,knn}_robustness.json"
echo " - Plots: $PLOTS_DIR and $PLOTS_CMP_DIR/{linear,knn}"


