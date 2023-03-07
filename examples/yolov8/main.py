from __future__ import annotations
import tvm
from tvmer.utils import load_onnx, gen_library


def main(target=tvm.target.arm_cpu("rk3399"), dtype="int8", lib_path: str = "lib/arm_cpu_rk3399.so"):
    mod, params = load_onnx(path="./model/yolov8s_detect.onnx", batch_size=1, dtype=dtype)
    lib = gen_library(mod, params, target, lib_path)
    # print(lib.get_source())
    # print(len(lib.imported_modules))
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
    # import tvm.contrib.hexagon
    # main(tvm.target.hexagon(), "lib/hexagon.so")
    main("cuda", "float32", "lib/cuda_f32.so")
    main("cuda", "int8", "lib/cuda_i8.so")
    main("llvm", "float32", "lib/llvm_f32.so")
    main("llvm", "int8", "lib/llvm_i8.so")
    # main(tvm.target.arm_cpu("rk3399"))
