from __future__ import annotations
import tvm
from tvmer.utils import load_onnx, gen_library, infer_time, tune
from tvm.contrib.graph_executor import GraphModule
from tvm import auto_scheduler



def main(target, dev, dtype="int8", lib_path: str = "lib/arm_cpu_default.so"):
    print("Extract tasks...")
    mod, params = load_onnx(path="./model/addtranspose.onnx", batch_size=1, dtype=dtype)
    tune(mod, params, target)
    lib = gen_library(mod, params, target, lib_path)

    module = GraphModule(lib['default'](dev))
    module.set_input("474", np.zeros([1, 69, 8400]))
    ftimer = module.module.time_evaluator("run", dev, number=2, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print(
        "%-20s %-19s (%s)" % ("rk3588", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )

    target = target if isinstance(target, str) else target.device_name
    with open(f".tvmer/llvm_ir/{target}_{dtype}.source", mode="w") as f:
        f.write(lib.get_lib().get_source())

    # dev = tvm.device(str(target), 0)
    # module = runtime.GraphModule(lib["default"](dev))

    # ftimer = module.module.time_evaluator("run", dev, number=1, repeat=30)

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

    # rk3588_target = tvm.target.Target(
    #     "llvm -keys=arm_cpu,cpu -device=arm_cpu "
    #     "-model=rk3588 -mtriple=aarch64-linux-gnu -mattr=+neon"
    # )

    main("llvm", tvm.device("cpu"), "int8", "lib/llvm_i8.so")
    # main(rk3588_target, tvm.device("cpu"), "int8", "lib/arm_rk3588_i8.so")

    # target = tvm.target.Target(tvm.target.mali(model='rk3588'), host=tvm.target.arm_cpu(model='rk3588'))
    # dev = tvm.device("opencl", 0)
    # main(target, dev, "int8", "lib/mali_i8.so")
    # main(tvm.target.arm_cpu(), "int8", "lib/arm_default_i8.so")
