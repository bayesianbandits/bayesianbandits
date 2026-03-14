"""Build all Takahashi Cython variants for benchmarking."""
import os
import shutil

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension

VARIANTS = [
    # (module_name, source_pyx, extra_compile_args)
    ("takahashi_baseline",       "takahashi_baseline.pyx",  []),
    ("takahashi_memview",        "takahashi_memview.pyx",   []),
    ("takahashi_baseline_ofast", "takahashi_baseline.pyx",  ["-O3", "-ffast-math", "-march=native"]),
    ("takahashi_memview_ofast",  "takahashi_memview.pyx",   ["-O3", "-ffast-math", "-march=native"]),
    ("takahashi_poisoned",       "takahashi_poisoned.pyx",  []),
    ("takahashi_no_cdiv",        "takahashi_no_cdiv.pyx",   []),
]


def build_all():
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(bench_dir)

    for name, source, extra_args in VARIANTS:
        print(f"\n--- Building {name} from {source} (flags: {extra_args or 'default'}) ---")

        # Copy source if the module name doesn't match the source basename
        actual_source = name + ".pyx"
        if actual_source != source:
            shutil.copy2(source, actual_source)

        ext = Extension(
            name,
            [actual_source],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_args,
        )

        ext_modules = cythonize([ext], force=True, annotate=True)

        dist = Distribution({"ext_modules": ext_modules})
        dist.parse_config_files()

        cmd = dist.get_command_obj("build_ext")
        cmd.inplace = True
        cmd.ensure_finalized()
        cmd.run()

    print("\n=== All variants built successfully ===")


if __name__ == "__main__":
    build_all()
