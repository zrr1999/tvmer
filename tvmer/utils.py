from __future__ import annotations

import os
import rich
import tvm
from rich.progress import Progress
from tvm import relay, IRModule, autotvm
from tvm.contrib.graph_executor import GraphModule
import onnx
import onnx_graphsurgeon as gs
from tvm import auto_scheduler
from pathlib import Path
from typing import TYPE_CHECKING, Union
from enum import Enum
from tvm import autotvm

if TYPE_CHECKING:
    Path = Union[Path, str]


class TunerStr(str, Enum):
    xgb = "xgb"
    ga = "ga"
    random = "random"
    gridsearch = "gridsearch"


def str2tuner(tuner_str: TunerStr, task):
    if tuner_str == TunerStr.xgb:
        tuner = autotvm.tuner.XGBTuner(task, loss_type="rank")
    elif tuner_str == "ga":
        tuner = autotvm.tuner.GATuner(task, pop_size=50)
    elif tuner_str == "random":
        tuner = autotvm.tuner.RandomTuner(task)
    elif tuner_str == "gridsearch":
        tuner = autotvm.tuner.GridSearchTuner(task)
    else:
        raise ValueError(f"Invalid tuner: {tuner_str}")
    return tuner


def load_onnx(path: Path, batch_size: int = 1, dtype: str = "int8") -> tuple[IRModule, dict[str, tvm.nd.NDArray]]:
    rich.print(f"load model from [blue]{path}[/blue]")
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    graph = gs.import_onnx(onnx_model)
    shape_dict = {inp.name: list(map(lambda _: _ if _ > 0 else batch_size, inp.shape)) for inp in graph.inputs}
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, dtype=dtype)
    if not os.path.exists(".tvmer/ir"):
        os.makedirs(".tvmer/ir")
    with open(".tvmer/ir/origin", mode="w") as f:
        f.write(str(mod))
    return mod, params


def gen_library(
        mod: IRModule,
        params: dict[str, tvm.nd.NDArray],
        target: tvm.target.Target = "llvm",
        path: Path = ".tvmer/lib/compiled.so"
):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    rich.print(f"export lib to [blue]{path}[/blue]")
    lib.export_library(path)
    return lib


def load_module(lib_path: Path, dev) -> GraphModule:
    lib = tvm.runtime.load_module(lib_path)
    return GraphModule(lib['default'](dev))


def infer_time(lib_path: Path, input_data, dev=tvm.cpu(), repeat=10):
    import time
    module = load_module(lib_path, dev)

    t = time.time()
    for _ in range(repeat):
        module.run(**input_data)
    return (time.time() - t) / repeat


def tune(
        mod: IRModule,
        params: dict[str, tvm.nd.NDArray],
        target: tvm.target.Target = "llvm",
        num_measure_trials: int = 200
):
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(".tvmer/log/log_file")],
    )
    tuner.tune(tune_option)


def tune_with_template(
        mod: IRModule,
        params: dict[str, tvm.nd.NDArray],
        target: tvm.target.Target = "llvm",
        num_measure_trials: int = 200,
        tuner: TunerStr = TunerStr.xgb,
):
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        params=params
    )

    log_filename = "tuning.log"
    tmp_log_file = "tuning.log.tmp"

    with Progress() as progress:
        task_tasks = progress.add_task("Task", total=len(tasks))
        for i, tsk in enumerate(reversed(tasks)):
            # create tuner
            tuner_obj = str2tuner(tuner, tsk)

            tsk_trial = min(num_measure_trials, len(tsk.config_space))

            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            task_tuner = progress.add_task("Tuning...", total=tsk_trial)

            # if use_transfer_learning:
            #     if os.path.isfile(tmp_log_file):
            #         tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
            # process tuning
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=None,
                measure_option=autotvm.measure_option(
                    autotvm.LocalBuilder(),
                    autotvm.LocalRunner()
                ),
                callbacks=[
                    lambda cls, inputs, results: progress.update(task_tuner, advance=len(inputs)),
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )
            progress.update(task_tasks, advance=1)

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    return autotvm.apply_history_best(log_filename)
