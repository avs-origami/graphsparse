from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='tensorviz',
    ext_modules=[cpp_extension.CppExtension('tensorviz', ['tensorviz.cpp'], extra_compile_args=['-std=gnu++2c'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

Extension(
    name='tensorviz',
    sources=['tensorviz.cpp'],
    include_dirs=cpp_extension.include_paths(),
    language='c++',
)