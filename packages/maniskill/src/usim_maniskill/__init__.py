# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ManiSkill simulator implementation for usim abstraction."""

from usim_maniskill.adapter import ManiSkillSimulator

__all__ = ['ManiSkillSimulator', 'CraneX7']


def __getattr__(name: str):
    if name == 'CraneX7':
        from usim_maniskill.agent import CraneX7

        return CraneX7
    raise AttributeError(name)
