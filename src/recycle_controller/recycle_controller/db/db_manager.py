from collections.abc import Iterable
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from db_schema import InventoryItem


class MongoDBManager:
    DATABASE_NAME = "inventory_db"
    ITEM_COLLECTION_NAME = "items"
    COUNTER_COLLECTION_NAME = "counters"

    def __init__(
        self,
        uri: str = (
            "mongodb://admin:mongodb_password"
            "@localhost:27018/?authSource=admin"
        ),
        reset_on_start: bool = False,
    ) -> None:
        self.uri = uri

        self.client: MongoClient = MongoClient(
            self.uri,
            serverSelectionTimeoutMS=3000,
            tz_aware=True,
        )

        self._check_connection()

        if reset_on_start:
            self.client.drop_database(self.DATABASE_NAME)

        self.db: Database = self.client[self.DATABASE_NAME]
        self.items: Collection = self.db[
            self.ITEM_COLLECTION_NAME
        ]
        self.counters: Collection = self.db[
            self.COUNTER_COLLECTION_NAME
        ]

        self._create_indexes()

    def _check_connection(self) -> None:
        try:
            self.client.admin.command("ping")
        except PyMongoError as error:
            self.client.close()
            raise ConnectionError(
                f"MongoDB 연결 실패: {self.uri}"
            ) from error

    def _create_indexes(self) -> None:
        self.items.create_index(
            [("id", ASCENDING)],
            unique=True,
            name="unique_item_id",
        )

        self.items.create_index(
            [
                ("section", ASCENDING),
                ("size", ASCENDING),
            ],
            name="section_size_index",
        )

        self.items.create_index(
            [("created_at", DESCENDING)],
            name="created_at_index",
        )

    def _get_next_id(self) -> int:
        counter = self.counters.find_one_and_update(
            {"_id": "item_id"},
            {"$inc": {"sequence": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if counter is None:
            raise RuntimeError("ID 생성에 실패했습니다.")

        return int(counter["sequence"])

    def create_item(
        self,
        section: int,
        size: int,
    ) -> InventoryItem:
        item = InventoryItem(
            id=self._get_next_id(),
            section=section,
            size=size,
        )

        self.items.insert_one(item.to_dict())
        return item

    def get_item(self, item_id: int) -> InventoryItem | None:
        document = self.items.find_one({"id": item_id})

        if document is None:
            return None

        return InventoryItem.from_dict(document)

    def get_items(
        self,
        section: int | None = None,
        size: int | None = None,
    ) -> list[InventoryItem]:
        query: dict[str, Any] = {}

        if section is not None:
            if section not in InventoryItem.VALID_SECTIONS:
                raise ValueError("section은 0~3이어야 합니다.")
            query["section"] = section

        if size is not None:
            if size not in InventoryItem.VALID_SIZES:
                raise ValueError(
                    "size는 240, 260, 280 중 하나여야 합니다."
                )
            query["size"] = size

        documents: Iterable[dict[str, Any]] = (
            self.items.find(query).sort("created_at", ASCENDING)
        )

        return [
            InventoryItem.from_dict(document)
            for document in documents
        ]

    def get_latest_item(
        self,
        section: int | None = None,
        size: int | None = None,
    ) -> InventoryItem | None:
        query: dict[str, Any] = {}

        if section is not None:
            query["section"] = section

        if size is not None:
            query["size"] = size

        document = self.items.find_one(
            query,
            sort=[("created_at", DESCENDING)],
        )

        if document is None:
            return None

        return InventoryItem.from_dict(document)

    def update_item(
        self,
        item_id: int,
        *,
        section: int | None = None,
        size: int | None = None,
    ) -> bool:
        updates: dict[str, int] = {}

        if section is not None:
            if section not in InventoryItem.VALID_SECTIONS:
                raise ValueError("section은 0~3이어야 합니다.")
            updates["section"] = section

        if size is not None:
            if size not in InventoryItem.VALID_SIZES:
                raise ValueError(
                    "size는 240, 260, 280 중 하나여야 합니다."
                )
            updates["size"] = size

        if not updates:
            return False

        result = self.items.update_one(
            {"id": item_id},
            {"$set": updates},
        )

        return result.matched_count == 1

    def delete_item(self, item_id: int) -> bool:
        result = self.items.delete_one({"id": item_id})
        return result.deleted_count == 1

    def clear_items(self, reset_id: bool = True) -> None:
        self.items.delete_many({})

        if reset_id:
            self.counters.delete_one({"_id": "item_id"})

    def reset_database(self) -> None:
        self.client.drop_database(self.DATABASE_NAME)

        self.db = self.client[self.DATABASE_NAME]
        self.items = self.db[self.ITEM_COLLECTION_NAME]
        self.counters = self.db[self.COUNTER_COLLECTION_NAME]

        self._create_indexes()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "MongoDBManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()