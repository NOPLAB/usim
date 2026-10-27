# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Optional Genesis simulator episode backend."""

from importlib import import_module

__all__ = ['Backend', 'GenesisSimulator', 'GenesisEnvironment', 'PickPlace']


def __getattr__(name: str):
    modules = {
        'Backend': '.backend',
        'GenesisSimulator': '.adapter',
        'GenesisEnvironment': '.environments',
        'PickPlace': '.environments',
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
