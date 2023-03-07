from __future__ import annotations

import os
import tvm
from tvm import relay, IRModule
import onnx
import onnx_graphsurgeon as gs


def load_onnx(path: str, batch_size: int = 1, dtype: str = "int8") -> tuple[IRModule, dict[str, tvm.nd.NDArray]]:
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


def gen_library(mod, params, target="llvm", path: str = ".tvmer/lib/compiled.so"):
    with tvm.transform.PassContext(opt_level=3):
        graph_json, lib, params = relay.build(mod, target=target, params=params)
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    lib.export_library(path)
    return lib
