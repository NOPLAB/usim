# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ManiSkill simulator implementation for usim abstraction."""

__all__ = ['Backend', 'ManiSkillSimulator', 'CraneX7']


def __getattr__(name: str):
    if name == 'Backend':
        from usim_maniskill.backend import Backend

        return Backend
    if name == 'ManiSkillSimulator':
        from usim_maniskill.adapter import ManiSkillSimulator

        return ManiSkillSimulator
    if name == 'CraneX7':
        from usim_maniskill.agent import CraneX7

        return CraneX7
    raise AttributeError(name)
