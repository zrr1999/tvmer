from __future__ import annotations

import tvm
from tvm.contrib.graph_executor import GraphModule
import numpy as np
import typer
import rich
from rich.progress import Progress, ProgressBar
from pathlib import Path
from tvm import auto_scheduler
from tvm import autotvm

from tvmer.utils import load_onnx, gen_library, load_module, TunerStr
import tvmer.utils as utils

app = typer.Typer(rich_markup_mode="rich")


@app.command()
def run(lib_path: Path, dev: str, repeat: int = 10):
    """
    run a compiled module
    """
    dev = tvm.device(dev)
    module = load_module(lib_path, dev)
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000
    rich.print(
        f"mean: {np.mean(prof_res):.2f} ms, std: {np.std(prof_res):.2f} ms"
    )
    # print(module.inp())
    # print(module.get_input_info())
    # module.set_input("479", np.zeros([1, 64, 8400]))
    # module.set_input("p1", np.zeros([1, 2, 8400]))


@app.command()
def compile(
        model_path: Path,
        target: str = "llvm",
        target_host: str = None,
        export_path: Path = ".tvmer/lib/default.so",
        dtype="float32"
):
    """
    compile a model
    """
    target = tvm.target.Target(target, host=target_host)

    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)
    gen_library(mod, params, target, export_path)


@app.command()
def tune(
        model_path: Path,
        target: str = "llvm",
        target_host: str = None,
        export_path: Path = ".tvmer/lib/tuned_default.so",
        dtype="float32",
        num_measure_trials: int = 200,
        tuner: TunerStr = None,
):
    """
    auto-tune a model
    """
    target = tvm.target.Target(target, host=target_host)

    print("Extract tasks...")
    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)
    if tuner is None:
        utils.tune(mod, params, target, num_measure_trials)
        gen_library(mod, params, target, export_path)
    else:
        with utils.tune_with_template(mod, params, target, num_measure_trials, tuner):
            gen_library(mod, params, target, export_path)


@app.command(deprecated=True)
def tune_next(
        model_path: Path,
        target: str = "llvm",
        target_host: str = None,
        export_path: Path = ".tvmer/lib/tuned_default.so",
        dtype="float32",
        num_measure_trials: int = 200,
):
    """
    auto-tune a model
    """
    rich.print("This command is deprecated.")
    print("Extract tasks...")
    target = tvm.target.Target(target, host=target_host)
    mod, params = load_onnx(path=model_path, batch_size=1, dtype=dtype)
    with utils.tune_with_template(mod, params, target, num_measure_trials):
        gen_library(mod, params, target, export_path)


# @app.command(deprecated=True)
# def main(target, dev, dtype="int8", lib_path: Path = "lib/arm_cpu_default.so"):
#     print("Extract tasks...")
#     mod, params = load_onnx(path="./model/yolov8s_detect.onnx", batch_size=1, dtype=dtype)
#     tune(mod, params, target)
#     lib = gen_library(mod, params, target, lib_path)
#
#     module = GraphModule(lib['default'](dev))
#     module.set_input("479", np.zeros([1, 64, 8400]))
#     ftimer = module.module.time_evaluator("run", dev, number=2, repeat=10)
#     prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
#     print(
#         "%-20s %-19s (%s)" % ("rk3588", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
#     )
#
#     target = target if isinstance(target, str) else target.device_name
#     with open(f".tvmer/llvm_ir/{target}_{dtype}.source", mode="w") as f:
#         f.write(lib.get_lib().get_source())


if __name__ == '__main__':
    app()
