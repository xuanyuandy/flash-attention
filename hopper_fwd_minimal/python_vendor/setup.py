from setuptools import find_namespace_packages, setup


setup(
    name="hopper-fwd-minimal-python-vendor",
    version="0.0.0",
    package_dir={"": "nvidia_cutlass_dsl/python_packages"},
    packages=find_namespace_packages("nvidia_cutlass_dsl/python_packages"),
)
