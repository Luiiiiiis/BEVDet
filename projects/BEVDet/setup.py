import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, \
    CUDAExtension


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-gencode=arch=compute_120,code=sm_120',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == '__main__':
    setup(
        name='bevdet',
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_v2_ext',
                module='bevdet.ops.bev_pool_v2',
                sources=['src/bev_pool.cpp'],
                sources_cuda=['src/bev_pool_cuda.cu'],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
