from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar


@dataclass
class InventoryItem:
    VALID_SECTIONS: ClassVar[set[int]] = {0, 1, 2, 3}
    VALID_SIZES: ClassVar[set[int]] = {240, 260, 280}

    section: int
    size: int
    id: int | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.section not in self.VALID_SECTIONS:
            raise ValueError(
                f"section은 0~3이어야 합니다: {self.section}"
            )

        if self.size not in self.VALID_SIZES:
            raise ValueError(
                f"size는 240, 260, 280 중 하나여야 합니다: {self.size}"
            )

        if self.id is not None and self.id < 1:
            raise ValueError("id는 1 이상의 정수여야 합니다.")

        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(
                tzinfo=timezone.utc
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "InventoryItem":
        data = document.copy()
        data.pop("_id", None)

        return cls(
            id=data.get("id"),
            section=data["section"],
            size=data["size"],
            created_at=data["created_at"],
        )