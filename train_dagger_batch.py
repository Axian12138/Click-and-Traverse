import os
import shlex
import subprocess
import sys
from datetime import datetime


python_executable = sys.executable

# Shared teacher config for all tasks below.
# Use run names under data/logs/$WANDB_PROJECT. The latest checkpoint is selected
# automatically, and each teacher's checkpoints/config.json provides its scene.
teacher_restore_names = [
    "04171623_G1Cat_0417testV0_xT0p0xempty",
    # "05111302_G1Cat_V0_xG1p0xT0p0xhurdle0",
]

tasks = [
    # cuda_devices, task, exp_name, restore_name, ground, lateral, overhead, obs_path, term_collision_threshold
    # cuda_devices can be 0, "0,1,2,3", or [0, 1, 2, 3].
    (0, "G1Cat", "dagger_debug", "none", 0.0, 0.0, 0.0, "data/assets/TypiObs/empty", 0.0),
    # ([0, 1, 2, 3], "G1Cat", "dagger_4gpu", "none", 0.0, 0.0, 0.0, "data/assets/TypiObs/empty", 0.0),
]

processes = []


def _cuda_visible_devices(cuda_devices):
    if isinstance(cuda_devices, (list, tuple)):
        return ",".join(str(device) for device in cuda_devices)
    return str(cuda_devices)


def _safe_log_token(value):
    return _cuda_visible_devices(value).replace(",", "-")


if __name__ == "__main__":
    if not teacher_restore_names:
        raise ValueError("Set teacher_restore_names before running DAgger batch training.")

    output_dir = "./output_logs"
    os.makedirs(output_dir, exist_ok=True)
    process_cmd_map = {}

    for cuda_devices, task, exp_name, restore_name, ground, lateral, overhead, obs_path, term_collision_threshold in tasks:
        cuda_visible_devices = _cuda_visible_devices(cuda_devices)
        cmd = [
            python_executable,
            "-m",
            "train_ppo_dagger",
            "--task",
            task,
            "--restore_name",
            restore_name,
            "--exp_name",
            exp_name,
            "--ground",
            str(ground),
            "--lateral",
            str(lateral),
            "--overhead",
            str(overhead),
            "--term_collision_threshold",
            str(term_collision_threshold),
            "--obs_path",
            obs_path,
            "--teacher_restore_names",
            *teacher_restore_names,
        ]
        cmd_display = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} " + " ".join(
            shlex.quote(part) for part in cmd
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cuda_log_token = _safe_log_token(cuda_devices)
        stdout_file = os.path.join(output_dir, f"{timestamp}_{cuda_log_token}_dagger_stdout.log")
        stderr_file = os.path.join(output_dir, f"{timestamp}_{cuda_log_token}_dagger_stderr.log")

        with open(stdout_file, "w") as out_file, open(stderr_file, "w") as err_file:
            print(f"Executing: {cmd_display}")
            out_file.write(f"{cmd_display}\n")
            err_file.write(f"{cmd_display}\n")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            process = subprocess.Popen(cmd, env=env, stdout=out_file, stderr=err_file)
            processes.append(process)
            process_cmd_map[process] = cmd_display

    while processes:
        for process in processes[:]:
            retcode = process.poll()
            if retcode is not None:
                if retcode != 0:
                    cmd = process_cmd_map[process]
                    print(f"\033[91mReturn code {retcode}.\nCommand: {cmd}\033[0m")
                processes.remove(process)

    print("All DAgger tasks completed.")
