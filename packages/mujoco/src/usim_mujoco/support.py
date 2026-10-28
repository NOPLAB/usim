"""Shared request rejection with the portable capability contract."""

from usim.capability import SupportReport, SupportStatus


def unsupported(reason: str) -> SupportReport:
    return SupportReport(SupportStatus.UNSUPPORTED, (reason,))
