from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="h100-hybrid-compressed-attention",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="hybrid_attention_cuda",
            sources=["csrc/bindings.cpp", "csrc/csa_attention.cu", "csrc/dsv4_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
