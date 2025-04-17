import subprocess

def train_model(args):
    subprocess.run([
        "python", 
        "main.py"
    ] + args)

def prune_model(args):
    subprocess.run([
        "python", 
        "prune.py"
    ] + args)


def run_experiments(pruning_method):
    weights_path = f"./model_weights/{pruning_method}"
    additional_args = [] if pruning_method == "l1_norm" else ["--simple-classifier"]

    train_model([
        "base_run", 
        f"--save={weights_path}"
    ] + additional_args)

    prune_model([
        pruning_method, 
        "initial", 
        f"--model={weights_path}/base_run.pth.tar",
        f"--initial_model={weights_path}/base_run_initial_model.pth.tar",
        f"--save={weights_path}"
    ])

    prune_model([
        pruning_method, 
        "random", 
        f"--model={weights_path}/base_run.pth.tar",
        f"--save={weights_path}"
    ])

    prune_model([
        pruning_method, 
        "unpruned", 
        f"--model={weights_path}/base_run.pth.tar",
        f"--save={weights_path}"
    ])

    train_model([
        "fine_tune_base_run",
        "--epochs=20",
        f"--save={weights_path}",
        f"--fine_tuning={weights_path}/base_run.pth.tar"
    ] + additional_args)

    train_model([
        f"fine_tune_{pruning_method}_initial",
        f"--save={weights_path}",
        f"--fine_tuning={weights_path}/base_run.pth_initial_model.pth.tar"
    ] + additional_args)

    train_model([
        f"fine_tune_{pruning_method}_random",
        f"--save={weights_path}",
        f"--fine_tuning={weights_path}/base_run.pth_pruned_{pruning_method}_random.pth.tar"
    ] + additional_args) 

    train_model([
        f"fine_tune_{pruning_method}_unpruned",
        f"--save={weights_path}",
        f"--fine_tuning={weights_path}/base_run.pth_pruned_{pruning_method}_unpruned.pth.tar"
    ] + additional_args) 

if __name__ == "__main__":
    # run_experiments("l1_norm")
    run_experiments("network_slimming")