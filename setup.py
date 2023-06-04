from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

exec(open("pointnet2/version.py").read())

setup(
    name="pointnet2",
    version=__version__,
    description="PyTorch Pointnet++",
    long_description="Updated PyTorch Pointnet++ implementation",
    author="Michael Danielczuk",
    author_email="michael.danielczuk@gmail.com",
    license="MIT Software License",
    url="https://github.com/mjd3/pointnet2",
    keywords="robotics computer vision",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=["pointnet2"],
    install_requires=["torch", "tqdm"],
    ext_modules=[
        CUDAExtension(
            "pointnet2_cuda",
            [
                "pointnet2/src/pointnet2_api.cpp",
                "pointnet2/src/ball_query.cpp",
                "pointnet2/src/ball_query_gpu.cu",
                "pointnet2/src/group_points.cpp",
                "pointnet2/src/group_points_gpu.cu",
                "pointnet2/src/interpolate.cpp",
                "pointnet2/src/interpolate_gpu.cu",
                "pointnet2/src/sampling.cpp",
                "pointnet2/src/sampling_gpu.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
