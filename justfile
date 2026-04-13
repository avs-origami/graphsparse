set positional-arguments

# List available recipes
@help:
  just --list --color always --unsorted --list-prefix " -> " | sed 's,->,\x1B[91m&\x1B[0m,' | sed 's,Available recipes:,\x1B[92m&\x1B[0m,'

# Test `pack` with scale (1x | 2x | 4x) and tree anim (true | false)
@test pack scale="1x" anim="" seed="":
    bash -c "mkdir -pv {img,log,scrot}"
    cargo build -p {{pack}} --release
    cargo build -p sim --release
    bash -c "./target/release/sim -C target/release/{{pack}} -s {{scale}} {{anim}} {{seed}}"

# Train the RL model
@train *ARGS:
    bash -c "mkdir -pv {img,log,scrot,models/tmp,learning_curves}"
    cargo build -p trainer --release
    cargo build -p sim --release
    bash -c "./target/release/trainer {{ARGS}}"
    paplay -p bell.ogg

# Train the RL model with profiling
@train_profile *ARGS:
    bash -c "mkdir -pv {img,log,scrot,models/tmp,learning_curves}"
    cargo build -p trainer --release
    cargo build -p sim --release
    bash -c "flamegraph --root -- ./target/release/trainer"
    paplay -p bell.ogg

@flamegraph cmd:
    bash -c "flamegraph --root -- {{cmd}}"

# Test `pack` without the visual
@test_headless pack:
    bash -c "mkdir -pv {img,log,scrot}"
    cargo build -p {{pack}} --release
    cargo build -p sim --release
    bash -c "./target/release/sim_no_win -C $(pwd)/target/release/{{pack}} -a false"

# Same as `just test` but without building first
@run pack scale="1x" anim="false":
    bash -c "mkdir -pv {img,log,scrot}"
    bash -c "./target/release/sim -C $(pwd)/target/release/{{pack}} -a false -s {{scale}}"

# Same as `just test_headless` but without building first
@run_headless pack:
    bash -c "mkdir -pv {img,log,scrot}"
    bash -c "./target/release/sim_no_win -C $(pwd)/target/release/{{pack}} -a false"

# Build, do `num` test runs and then push the data to remote
@test_runs num: build
    bash -c "mkdir -pv {img,log,scrot}"
    ./test.sh {{num}}
    git pull
    git add img log
    git commit -m "Automated push, test run @ $(date)"
    git push origin main

# Test `pack` without the simulator, uses {pack}/test.txt as input
@debug pack:
    cargo build -p {{pack}} --release
    cat {{pack}}/test.txt | ./target/debug/{{pack}}

# Test the simulator with simbug.txt as input
@sim scale="" anim="false":
    cargo build -p sim --release
    cat simbug.txt | ./target/release/sim {{scale}} "false" "true" {{anim}}

# Build all workspace members
@build:
    cargo build --release
    bash -c "mkdir -pv {img,log,scrot}"

# Compile TeX files
@texify file="":
    ./texify.sh {{file}}

# Clear the `img` directory
@clear_img:
    rm -rf img/*

# Clear the `log` directory
@clear_log:
    rm -rf log/*

# Clear the `scrot` directory
@clear_scrot:
    rm -rf scrot/*

# Clear the `img` and `log` directories
@clear_tests:
    rm -rf img/*
    rm -rf log/*

# Clear the `img2` and `log2` directories
@clear_tests_2:
    rm -rf img2/*
    rm -rf log2/*

# Clear the `models` and `learning_curves` directories
@clear_train:
    rm -rf models/*
    rm -rf learning_curves/*

# Demo with low clutter
@demo clutter="":
    bash -c "mkdir -pv {img,log,scrot}"
    cargo build -p rrt_inc --release
    cargo build -p sim --release
    ./run.sh rrt_inc 4x "false" "true" true {{clutter}}

# Generate `num` trees
@treegen num:
    cargo build -p tree_gen --release
    ./target/release/tree_gen {{num}}

# Clear the `trees` directory
@clear_trees:
    rm -rf trees/*.tree

# Train SparRL
# train *ARGS:
#     python3 spar_rl/main.py {{ARGS}} --tree --train
#     notify-send -t 100000 "Done training!"
#     mpg123 ~/Downloads/ding.mp3

# # Evaluate SparRL
# eval *ARGS:
#     python3 spar_rl/main.py {{ARGS}} --spar_tree trees/pruned.tree --tree --load --eval
#     notify-send -t 100000 "Done evaluating!"
#     mpg123 ~/Downloads/ding.mp3

# Generate spars_tch/compile_commands.json
@gen_compile_commands:
    bear -- cargo build -p spars_tch --release
    mv compile_commands.json spars_tch/

# Train the Graph2Vec encoder
@graphvec:
    bash -c "source docvec/.venv/bin/activate && cd docvec && python main.py --data_file_name 'rrt_data.csv' --eval_data_file_name 'rrt_eval.csv' --num_epochs 1000 --batch_size 32 --num_noise_words 8 --vec_dim 256 --lr 0.0005"

# Build the cpptest for spars_tch
@spars_cpp:
    bash -c "cd spars_tch/build && make"
    ./spars_tch/build/spars_tch