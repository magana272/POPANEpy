from dataclasses import dataclass


@dataclass
class StudyConfig:
    """Configuration for a POPANE study"""
    number: int
    name: str
    measurements: tuple[str, ...]
    dtypes: dict[str, str] | None
    description: str | None = None

    @property
    def columns(self) -> set[str]:
        return set(self.measurements)
