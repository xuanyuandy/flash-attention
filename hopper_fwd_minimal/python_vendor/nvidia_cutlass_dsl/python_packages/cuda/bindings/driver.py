"""Small cuda.bindings.driver stub for local macOS imports.

The actual cuda-python bindings are Linux/CUDA runtime dependencies and are
installed in the Modal H100 environment. These placeholders keep imports and
editor indexing working locally.
"""


class CUresult:
    CUDA_SUCCESS = 0


class CUdevice_attribute:
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


def cuInit(flags=0):
    return (CUresult.CUDA_SUCCESS,)


def cuGetErrorString(result):
    return (CUresult.CUDA_SUCCESS, b"cuda stub")


def cuDeviceGet(device_id):
    return (CUresult.CUDA_SUCCESS, device_id)


def cuDeviceGetAttribute(attribute, device):
    return (CUresult.CUDA_SUCCESS, 0)

