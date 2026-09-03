#!/bin/bash
# Evaluate the three test cases: the learned model (ours) as well as the
# two baselines (no pruning and random pruning) on a specified checkpoint.
#
# Usage: ./run_experiments.sh [ARGS]
# 
# [ARGS] catches everything and passes it to the trainer binary after stripping
# out the flags that are set to specific values by this script. Therefore, you
# can simply copy/paste the same arguments passed at training time.
#
# Experiment overrides (these can be set at the CLI or by editing this script):
#   SEEDS => number of eval episodes per arm
#   MOVE_BUDGET => robot moves per episode
#
# Outputs:
#   results/experiment_<timestamp>/train.log
#   results/experiment_<timestamp>/eval_{no_prune,random,learned}.log
#
# After the run: python3 scripts/analyze_experiments.py results/experiment_<timestamp>

set -euo pipefail

# These can be set at the CLI or by editing them here directly
SEEDS="${SEEDS:-400}"
MOVE_BUDGET="${MOVE_BUDGET:-1000}"

EXPT_DIR="results/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPT_DIR"

BIN=./target/release/trainer

# Build once so cargo output doesn't clutter each invocation.
cargo build --release --bin trainer

echo
echo -e "\e[92mexperiment: $EXPT_DIR\e[0m"
echo -e "\e[90mseeds=$SEEDS\e[0m"
echo -e "\e[90mmove_budget=$MOVE_BUDGET\e[0m"

# Extract CHECKPOINT[,LAST_ITER] from --load, and strip the flags this
# script sets per-arm below so we don't accidentally override them.
STRIP_BOOL="--test --no-prune --random"
STRIP_VAL="--num-moves --episodes --num-eval"

CHECKPOINT=""
LAST_ITER=""
FILTERED=()
skip_next=0
prev=""
for a in "$@"; do
    if [ "$skip_next" = "1" ]; then
        skip_next=0
        prev="$a"
        continue
    fi

    # Capture --load's value (`--load foo,bar` or `--load=foo,bar`).
    val=""
    [ "$prev" = "--load" ] && val="$a"
    case "$a" in --load=*) val="${a#--load=}" ;; esac
    if [ -n "$val" ]; then
        CHECKPOINT="${val%%,*}"
        rest="${val#*,}"
        [ "$rest" != "$val" ] && LAST_ITER="$rest"
    fi

    # Boolean-flag strip.
    drop=0
    for f in $STRIP_BOOL; do
        [ "$a" = "$f" ] && drop=1
    done
    # Value-flag strip, two-word form (`--foo value`): mark next arg for skip.
    for f in $STRIP_VAL; do
        if [ "$a" = "$f" ]; then drop=1; skip_next=1; fi
    done
    # Value-flag strip, equals form (`--foo=value`).
    for f in $STRIP_VAL; do
        case "$a" in "$f"=*) drop=1 ;; esac
    done

    if [ "$drop" = "0" ]; then
        FILTERED+=("$a")
    fi
    prev="$a"
done

ARGS="${FILTERED[*]}"

if [ -z "$CHECKPOINT" ]; then
    echo -e "\e[91merror: --load run_id[,iter] must be provided in ARGS\e[0m" >&2
    exit 1
fi

# Assume highest-numbered iteration file for --load if one
# isn't provided (matches behavior of Rust code)
if [ -z "$LAST_ITER" ]; then
    LAST_ITER=$(ls "models/$CHECKPOINT/"ppo*.pt \
        | sed -n 's|.*/ppo\([0-9]\+\)\.pt|\1|p' \
        | sort -n | tail -1)
fi

echo -e "\e[92meval against $CHECKPOINT iter $LAST_ITER\e[0m"
echo "$CHECKPOINT,$LAST_ITER" > "$EXPT_DIR/checkpoint.txt"

# Evaluate each arm
for ARM in no_prune random learned; do
    echo
    echo -e "\e[92mevaluating arm=$ARM\e[0m"
    ARM_LOG="$EXPT_DIR/eval_${ARM}.log"

    # Set appropriate flags for each of the three arms
    case "$ARM" in
        no_prune) EXTRA="--no-prune" ;;
        random)   EXTRA="--random"   ;;
        learned)  EXTRA=""           ;;
    esac

    $BIN \
        --test $EXTRA \
        --num-moves "$MOVE_BUDGET" \
        --episodes "$SEEDS" \
        --num-eval 1 \
        $ARGS \
        2>&1 | tee "$ARM_LOG"
done

echo
# echo -e "\e[92mdone. analyze with: python3 scripts/analyze_experiments.py $EXPT_DIR\e[0m"

source .nvenv/bin/activate
python3 scripts/analyze_experiments.py $EXPT_DIR
