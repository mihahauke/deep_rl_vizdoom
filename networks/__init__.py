# -*- coding: utf-8 -*-
import inspect as _inspect
import sys as _sys

from .a3c import *
from .dqn import *
from .async_dqn import *
from util.coloring import red


def get_available_networks():
    nets = []
    for member in _inspect.getmembers(_sys.modules[__name__]):
        if _inspect.isclass(member[1]):
            member_name = member[0]
            if member_name.endswith("Net") and not member_name.startswith("_"):
                nets.append(member)
    return nets


def create_network(network_type, **args):
    if network_type is not None:
        for nname, nclass in get_available_networks():
            if network_type == nname or network_type == nclass.shortname:
                return nclass(**args)

    raise ValueError(red("Unsupported net: {}".format(network_type)))
