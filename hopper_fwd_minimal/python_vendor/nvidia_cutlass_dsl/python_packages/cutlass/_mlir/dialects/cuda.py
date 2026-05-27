# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional, Sequence, Union, List
from .._mlir_libs._cutlass_ir._mlirDialectsCuda import *
from ._cuda_ops_gen import *
from ._cuda_ops_gen import _Dialect
from ._cuda_enum_gen import *
from ._ods_common import _cext as _ods_cext
from ..ir import *


@_ods_cext.register_operation(_Dialect, replace=True)
class KernelOp(KernelOp):
    """Specialization of the KernelOp for the CUDA dialect."""

    def __init__(
        self,
        name,
        type,
        *,
        arg_attrs=None,
        res_attrs=None,
        cu_func_attrs=None,
        body_builder=None,
        loc=None,
        ip=None
    ):
        sym_name = StringAttr.get(str(name))

        if isinstance(type, tuple):
            type = FunctionType.get(inputs=type[0], results=type[1])
        type = TypeAttr.get(type)

        super().__init__(
            sym_name,
            type,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
            cu_func_attrs=cu_func_attrs,
            loc=loc,
            ip=ip,
        )

        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    @property
    def is_external(self):
        return len(self.regions[0].blocks) == 0

    @property
    def body(self):
        return self.regions[0]

    @property
    def type(self):
        return FunctionType(TypeAttr(self.operation.attributes["function_type"]).value)

    @property
    def name(self):
        return self.operation.attributes["sym_name"]

    @property
    def entry_block(self):
        if self.is_external:
            raise IndexError("External kernel has no entry block")
        return self.regions[0].blocks[0]

    def add_entry_block(self, arg_locs: Optional[Sequence[Location]] = None):
        """Add an entry block to the function."""
        if not self.is_external:
            raise IndexError("The kernel already has an entry block")
        self.body.blocks.append(*self.type.inputs, arg_locs=arg_locs)
        return self.body.blocks[0]

    @property
    def arg_attrs(self):
        if "arg_attrs" not in self.operation.attributes:
            return ArrayAttr.get([DictAttr.get({}) for _ in self.type.inputs])
        return ArrayAttr(self.operation.attributes["arg_attrs"])

    @arg_attrs.setter
    def arg_attrs(self, attribute: Union[ArrayAttr, List]):
        if isinstance(attribute, ArrayAttr):
            self.operation.attributes["arg_attrs"] = attribute
        else:
            self.operation.attributes["arg_attrs"] = ArrayAttr.get(
                attribute, context=self.context
            )

    @arg_attrs.deleter
    def arg_attrs(self):
        del self.operation.attributes["arg_attrs"]

    @property
    def res_attrs(self):
        if "res_attrs" not in self.operation.attributes:
            return None
        return self.operation.attributes["res_attrs"]

    @res_attrs.setter
    def res_attrs(self, value):
        if value is not None:
            self.operation.attributes["res_attrs"] = value
        elif "res_attrs" in self.operation.attributes:
            del self.operation.attributes["res_attrs"]

    @res_attrs.deleter
    def res_attrs(self):
        del self.operation.attributes["res_attrs"]

    @property
    def cu_func_attrs(self):
        if "cu_func_attrs" not in self.operation.attributes:
            return None
        return self.operation.attributes["cu_func_attrs"]

    @cu_func_attrs.setter
    def cu_func_attrs(self, value):
        if value is not None:
            self.operation.attributes["cu_func_attrs"] = value
        elif "cu_func_attrs" in self.operation.attributes:
            del self.operation.attributes["cu_func_attrs"]

    @cu_func_attrs.deleter
    def cu_func_attrs(self):
        del self.operation.attributes["cu_func_attrs"]


@_ods_cext.register_operation(_Dialect, replace=True)
class FuncOp(FuncOp):
    """Specialization of the FuncOp for the CUDA dialect."""

    def __init__(
        self,
        name,
        type,
        *,
        arg_attrs=None,
        res_attrs=None,
        body_builder=None,
        loc=None,
        ip=None
    ):
        sym_name = StringAttr.get(str(name))

        if isinstance(type, tuple):
            type = FunctionType.get(inputs=type[0], results=type[1])
        type = TypeAttr.get(type)

        super().__init__(
            sym_name, type, arg_attrs=arg_attrs, res_attrs=res_attrs, loc=loc, ip=ip
        )

        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    @property
    def is_external(self):
        return len(self.regions[0].blocks) == 0

    @property
    def body(self):
        return self.regions[0]

    @property
    def type(self):
        return FunctionType(TypeAttr(self.operation.attributes["function_type"]).value)

    @property
    def name(self):
        return self.operation.attributes["sym_name"]

    @property
    def entry_block(self):
        if self.is_external:
            raise IndexError("External function has no entry block")
        return self.regions[0].blocks[0]

    def add_entry_block(self, arg_locs: Optional[Sequence[Location]] = None):
        """Add an entry block to the function."""
        if not self.is_external:
            raise IndexError("The function already has an entry block")
        self.body.blocks.append(*self.type.inputs, arg_locs=arg_locs)
        return self.body.blocks[0]

    @property
    def arg_attrs(self):
        if "arg_attrs" not in self.operation.attributes:
            return None
        return self.operation.attributes["arg_attrs"]

    @arg_attrs.setter
    def arg_attrs(self, value):
        if value is not None:
            self.operation.attributes["arg_attrs"] = value
        elif "arg_attrs" in self.operation.attributes:
            del self.operation.attributes["arg_attrs"]

    @arg_attrs.deleter
    def arg_attrs(self):
        del self.operation.attributes["arg_attrs"]

    @property
    def res_attrs(self):
        if "res_attrs" not in self.operation.attributes:
            return None
        return self.operation.attributes["res_attrs"]

    @res_attrs.setter
    def res_attrs(self, value):
        if value is not None:
            self.operation.attributes["res_attrs"] = value
        elif "res_attrs" in self.operation.attributes:
            del self.operation.attributes["res_attrs"]

    @res_attrs.deleter
    def res_attrs(self):
        del self.operation.attributes["res_attrs"]
