from __future__ import annotations
import tvm
from tvmer.utils import load_onnx, gen_library, infer_time
from tvm.contrib.graph_executor import GraphModule


def main(target=tvm.target.arm_cpu(), dtype="int8", lib_path: str = "lib/arm_cpu_default.so"):
    mod, params = load_onnx(path="./model/yolov8s_detect.onnx", batch_size=1, dtype=dtype)
    lib = gen_library(mod, params, target, lib_path)
    target = target if isinstance(target, str) else target.device_name
    with open(f".tvmer/llvm_ir/{target}_{dtype}.source", mode="w") as f:
        f.write(lib.get_lib().get_source())

    # print(lib.imported_modules[0].get_source())
    # with tvm.transform.PassContext(opt_level=3):
    #     graph_json, lib, params = relay.build(mod, target=target, params=params)
    # with open(f"./generated/lookup_{i}.cu", mode="w") as f:
    #     f.write(lib.imported_modules[0].get_source())
    # for i in range(1, 11):
    #     # print(params)
    #     # print(mod)
    #     # target = "cuda"
    #     # with tvm.transform.PassContext(opt_level=3):
    #     #     graph_json, lib, params = relay.build(mod, target=target, params=params)
    #     # with open("./generated/lookup_trt.cu", mode="w") as f:
    #     #     f.write(lib.imported_modules[0].get_source())
    #     # lib.export_library(f'lib/cuda/lookup_{i}.so')


if __name__ == '__main__':
    import numpy as np

    # rk3588_target = tvm.target.Target(
    #     "llvm -keys=arm_cpu,cpu -device=arm_cpu "
    #     "-model=rk3588 -mattr=+neon"
    # )

    rk3588_target = tvm.target.Target(
        "llvm -keys=arm_cpu,cpu -device=arm_cpu "
        "-model=rk3588 -mtriple=aarch64-linux-gnu -mattr=+neon"
    )

    # main("llvm", "int8", "lib/llvm_i8.so")
    main(rk3588_target, "int8", "lib/arm_rk3588_i8.so")
    # main(tvm.target.arm_cpu(), "int8", "lib/arm_default_i8.so")
    print(infer_time("lib/arm_rk3588_i8.so", {"479": np.zeros([1, 64, 8400])}, repeat=1000))
    # print(infer_time("lib/llvm_i8.so", {"479": np.zeros([1, 64, 8400])}, repeat=1000))
    # print(infer_time("lib/arm_default_i8.so", {"479": np.zeros([1, 64, 8400])}, repeat=1000))
    # main("cuda", "float32", "lib/cuda_f32.so")
    # main("cuda", "int8", "lib/cuda_i8.so")
