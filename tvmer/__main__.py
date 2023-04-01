from __future__ import annotations

import tvm
from tvm.contrib.graph_executor import GraphModule
import numpy as np
import typer
import rich
from rich.progress import ProgressBar
from pathlib import Path
from tvm import auto_scheduler
from tvm import autotvm

from tvmer.utils import load_onnx, gen_library, load_module, tune

app = typer.Typer(rich_markup_mode="rich")


@app.command()
def run(lib_path: Path, dev: str):
    """
    run a compiled module [red](未完成)[/red]
    """
    dev = tvm.device(dev)
    module = load_module(lib_path, dev)
    # print(module.inp())
    # print(module.get_input_info())
    # module.set_input("479", np.zeros([1, 64, 8400]))
    # module.set_input("p1", np.zeros([1, 2, 8400]))
    ftimer = module.module.time_evaluator("run", dev, number=2, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print(
        f"mean: {np.mean(prof_res):.2f}ms, std: {np.std(prof_res):.2f}ms"
    )


@app.command()
def compile(target: str, model_path: Path, export_path: Path = ".tvmer/lib/default.so", dtype="float32"):
    """
    compile a model
    """
    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)
    gen_library(mod, params, target, export_path)


@app.command()
def tune(
        target: str,
        model_path: Path,
        export_path: Path = ".tvmer/lib/tuned_default.so",
        dtype="float32",
        num_measure_trials: int = 200,
):
    """
    auto-tune a model (free template)
    """
    print("Extract tasks...")
    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trials,
        measure_callbacks=[auto_scheduler.RecordToFile("log_file")],
    )
    tuner.tune(tune_option)

    gen_library(mod, params, target, export_path)


@app.command()
def tune_next(
        target: str,
        model_path: Path,
        export_path: Path = ".tvmer/lib/tuned_default.so",
        dtype="float32",
        num_measure_trials: int = 200,
):
    """
    auto-tune a model [red](未完成)[/red]
    """
    print("Extract tasks...")
    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=tvm.target.Target(target), params=params
    )

    log_filename = "tuning.log"
    tmp_log_file = "tuning.log.tmp"
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="rank", feature_type="curve")

        # if use_transfer_learning:
        #     if os.path.isfile(tmp_log_file):
        #         tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(num_measure_trials, len(tsk.config_space))

        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=None,
            measure_option=autotvm.measure_option(
                autotvm.LocalBuilder(),
                autotvm.LocalRunner()
            ),
            callbacks=[
                ProgressBar(tsk_trial),
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    with autotvm.apply_history_best(log_filename):
        gen_library(mod, params, target, export_path)


@app.command(deprecated=True)
def main(target, dev, dtype="int8", lib_path: Path = "lib/arm_cpu_default.so"):
    print("Extract tasks...")
    mod, params = load_onnx(path="./model/yolov8s_detect.onnx", batch_size=1, dtype=dtype)
    tune(mod, params, target)
    lib = gen_library(mod, params, target, lib_path)

    module = GraphModule(lib['default'](dev))
    module.set_input("479", np.zeros([1, 64, 8400]))
    ftimer = module.module.time_evaluator("run", dev, number=2, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print(
        "%-20s %-19s (%s)" % ("rk3588", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )

    target = target if isinstance(target, str) else target.device_name
    with open(f".tvmer/llvm_ir/{target}_{dtype}.source", mode="w") as f:
        f.write(lib.get_lib().get_source())


if __name__ == '__main__':
    app()
