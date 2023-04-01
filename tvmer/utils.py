from __future__ import annotations

import os

import rich
import tvm
from tvm import relay, IRModule
from tvm.contrib.graph_executor import GraphModule
import onnx
import onnx_graphsurgeon as gs
from tvm import auto_scheduler
from pathlib import Path


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


def gen_library(mod, params, target="llvm", path: Path = ".tvmer/lib/compiled.so"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    rich.print(f"export lib to [blue]{path}[/blue]")
    lib.export_library(path)
    return lib


def load_module(lib_path, dev):
    lib = tvm.runtime.load_module(lib_path)
    return GraphModule(tvm.runtime.load_module(lib_path)['default'](dev))


def infer_time(lib_path: Path, input_data, dev=tvm.cpu(), repeat=10):
    import time
    module = load_module(lib_path, dev)

    t = time.time()
    for _ in range(repeat):
        module.run(**input_data)
    return (time.time() - t) / repeat


def tune(mod, params, target):
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        measure_callbacks=[auto_scheduler.RecordToFile("log_file")],
    )
    tuner.tune(tune_option)
