"""文档增量事件：六类只追加记录的稳定合并、水位向量与续页游标。

事件来源（``type_rank`` 即合并次序，并列 created_at 时小者在前）：

    1 version        版本（versions，按文档）
    2 annotation     批注（annotations，按文档）
    3 location       迁移位置（locations，经 annotations 归属文档）
    4 confirmation   人工确认（confirmations，经 locations → annotations 归属）
    5 dependency     审阅依赖（dependencies，按文档）
    6 completion     完成记录（completions，按文档）

全部记录只追加，行 ID 单调递增。每个文档的水位向量固定为
``(versions, annotations, locations, confirmations, dependencies, completions)``
六张表在**同一 SQLite 读事务快照**内取到的该文档最大行 ID（空表记 0）；
事件只读取 ``(起点水位, 终点水位]`` 的半开区间，按
``(created_at, type_rank, id)`` 稳定合并。

游标（不透明字符串）封装游标版本、文档 ID、起止水位与最后排序键：

- 窗口中途的翻页游标始终沿用窗口起点的终点水位，读事务期间并发新增的行
  超过终点水位，不会进入本轮（不重复、不漏、不乱序）；
- 窗口读完（末页）返回 resume_cursor：起、止水位都钉在窗口终点、
  无最后排序键。用它续拉时，在新的读事务快照里固定新终点水位，
  因而**空轮也会推进水位**；
- 游标版本不符、结构损坏、范围非法（起 > 止、排序键不在窗口内）或
  跨文档使用一律抛 :class:`CursorError`，由路由映射为 422。

事件载荷只保留对应记录落库时的稳定业务字段（另附不可变的稳定引用键，
如版本号 / 批注 ID），**不计算查询时刻的派生状态**：例如 location 事件
保留原始 status / start / end 与候选，不叠加后续人工确认的生效值
（确认本身是独立的 confirmation 事件）。
"""
from __future__ import annotations

import base64
import heapq
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

# 游标格式版本；结构调整必须升版本，旧游标直接 422
CURSOR_VERSION = 1

# 合并次序（created_at 相同时按此排序），同时是六张表的固定类型编号
VERSION = "version"
ANNOTATION = "annotation"
LOCATION = "location"
CONFIRMATION = "confirmation"
DEPENDENCY = "dependency"
COMPLETION = "completion"

TYPES: Tuple[str, ...] = (
    VERSION,
    ANNOTATION,
    LOCATION,
    CONFIRMATION,
    DEPENDENCY,
    COMPLETION,
)
TYPE_RANK = {name: rank for rank, name in enumerate(TYPES, start=1)}

# 水位向量的固定槽位顺序（与 TYPES / TYPE_RANK 一一对应）
WATERMARK_TABLES: Tuple[str, ...] = (
    "versions",
    "annotations",
    "locations",
    "confirmations",
    "dependencies",
    "completions",
)

_CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

# 排序键：(created_at, type_rank, 行 ID)
SortKey = Tuple[str, int, int]


class CursorError(ValueError):
    """游标无效：版本 / 结构 / 范围错误或跨文档使用（路由映射为 422）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Cursor:
    """游标内容：文档 ID、起点 / 终点水位向量、最后排序键（末页游标为 None）。"""

    document_id: int
    start: Tuple[int, ...]
    end: Tuple[int, ...]
    last_key: Optional[SortKey] = None

    @property
    def at_round_boundary(self) -> bool:
        """是否为窗口边界游标（首轮隐式游标或末页 resume_cursor）。"""
        return self.last_key is None


def encode_cursor(cursor: Cursor) -> str:
    """把游标序列化为不透明字符串（base64url(JSON)）。"""
    payload = {
        "v": CURSOR_VERSION,
        "doc": cursor.document_id,
        "start": list(cursor.start),
        "end": list(cursor.end),
        "last": (
            None
            if cursor.last_key is None
            else [cursor.last_key[0], cursor.last_key[1], cursor.last_key[2]]
        ),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: object, document_id: int) -> Cursor:
    """解析并校验游标；``document_id`` 为路径中的文档 ID，用于跨文档检查。

    任何版本 / 结构 / 范围 / 跨文档问题都抛 :class:`CursorError`。
    """
    if not isinstance(token, str) or not token:
        raise CursorError("invalid_cursor", "游标为空或不是字符串")

    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        raise CursorError(
            "invalid_cursor", "游标不是有效的 base64url(JSON) 不透明令牌"
        ) from None

    if not isinstance(payload, dict):
        raise CursorError("invalid_cursor", "游标结构错误：顶层必须是对象")
    if payload.get("v") != CURSOR_VERSION:
        raise CursorError(
            "cursor_version_mismatch",
            f"游标版本不受支持：应为 {CURSOR_VERSION}，实际为 {payload.get('v')!r}",
        )

    doc = payload.get("doc")
    if not isinstance(doc, int) or isinstance(doc, bool) or doc < 1:
        raise CursorError("invalid_cursor", "游标结构错误：doc 必须为正整数")
    if doc != document_id:
        raise CursorError(
            "cross_document",
            f"游标属于文档 {doc}，不能在文档 {document_id} 上使用",
        )

    start = _valid_vector(payload.get("start"), "start")
    end = _valid_vector(payload.get("end"), "end")
    if any(lo > hi for lo, hi in zip(start, end)):
        raise CursorError(
            "cursor_range_invalid", "游标范围非法：起点水位不能高于终点水位"
        )

    last = payload.get("last")
    last_key: Optional[SortKey] = None
    if last is not None:
        if not isinstance(last, list) or len(last) != 3:
            raise CursorError(
                "invalid_cursor", "游标结构错误：last 必须为 [created_at, rank, id]"
            )
        created_at, rank, row_id = last
        if not isinstance(created_at, str) or not _CREATED_AT_RE.match(created_at):
            raise CursorError(
                "invalid_cursor", "游标结构错误：last.created_at 不是合法时间戳"
            )
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in TYPE_RANK.values()
        ):
            raise CursorError(
                "invalid_cursor", "游标结构错误：last.rank 不是已知事件类型编号"
            )
        if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id < 1:
            raise CursorError(
                "invalid_cursor", "游标结构错误：last.id 必须为正整数"
            )
        # 范围自洽：最后排序键必须落在其类型槽位的 (起点, 终点] 窗口内
        slot = rank - 1
        if not (start[slot] < row_id <= end[slot]):
            raise CursorError(
                "cursor_range_invalid",
                "游标范围非法：最后排序键的行 ID 不在其类型的 (起点, 终点] 水位内",
            )
        last_key = (created_at, rank, row_id)

    return Cursor(document_id=doc, start=start, end=end, last_key=last_key)


def _valid_vector(value: object, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != len(WATERMARK_TABLES):
        raise CursorError(
            "invalid_cursor",
            f"游标结构错误：{name} 必须是 {len(WATERMARK_TABLES)} 个水位值的数组",
        )
    vec = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise CursorError(
                "invalid_cursor", f"游标结构错误：{name} 水位必须为非负整数"
            )
        vec.append(item)
    return tuple(vec)


# ---------------------------------------------------------------------------
# 读事务内的水位与事件读取（调用方须持有 db.read_transaction 快照）
# ---------------------------------------------------------------------------


# 各类记录的归属方式不同：versions / annotations / dependencies / completions
# 直接带 document_id；locations 经 annotations 归属；
# confirmations 经 locations → annotations 归属。
_WATERMARK_SQL = {
    VERSION: "SELECT COALESCE(MAX(id), 0) AS m FROM versions WHERE document_id = ?",
    ANNOTATION: (
        "SELECT COALESCE(MAX(id), 0) AS m FROM annotations WHERE document_id = ?"
    ),
    LOCATION: (
        "SELECT COALESCE(MAX(l.id), 0) AS m FROM locations l "
        "JOIN annotations a ON a.id = l.annotation_id WHERE a.document_id = ?"
    ),
    CONFIRMATION: (
        "SELECT COALESCE(MAX(c.id), 0) AS m FROM confirmations c "
        "JOIN locations l ON l.id = c.location_id "
        "JOIN annotations a ON a.id = l.annotation_id WHERE a.document_id = ?"
    ),
    DEPENDENCY: (
        "SELECT COALESCE(MAX(id), 0) AS m FROM dependencies WHERE document_id = ?"
    ),
    COMPLETION: (
        "SELECT COALESCE(MAX(id), 0) AS m FROM completions WHERE document_id = ?"
    ),
}


def current_watermark(con: sqlite3.Connection, document_id: int) -> Tuple[int, ...]:
    """同一快照内固定六张表的文档级最大 ID（空表记 0）。"""
    return tuple(
        con.execute(_WATERMARK_SQL[name], (document_id,)).fetchone()["m"]
        for name in TYPES
    )


def watermark_dict(vec: Sequence[int]) -> dict:
    """水位向量的稳定 JSON 表示（槽位名即表名）。"""
    return {table: vec[i] for i, table in enumerate(WATERMARK_TABLES)}


# 范围扫描直接取装配载荷所需的全部列（行 ID 升序）；
# 只追加表内同一写入事务的行共享毫秒时间戳，跨表合并再统一按
# (created_at, type_rank, id) 排序。附加的版本号 / 批注 ID 均为不可变引用键。
_STREAM_SQL = {
    VERSION: """
        SELECT id, document_id, number, content, created_at
        FROM versions
        WHERE document_id = ? AND id > ? AND id <= ?
        ORDER BY id
    """,
    ANNOTATION: """
        SELECT a.id, a.document_id, a.body, a.selection, a.left_context,
               a.right_context, a.orig_start, a.orig_end,
               a.created_version_id, a.created_at,
               v.number AS created_version_number
        FROM annotations a
        JOIN versions v ON v.id = a.created_version_id
        WHERE a.document_id = ? AND a.id > ? AND a.id <= ?
        ORDER BY a.id
    """,
    LOCATION: """
        SELECT l.id, l.annotation_id, l.version_id, l.status, l.start, l."end",
               l.left_match, l.right_match, l.score, l.candidates,
               l.carried_from, l.created_at, v.number AS version_number
        FROM locations l
        JOIN annotations a ON a.id = l.annotation_id
        JOIN versions v ON v.id = l.version_id
        WHERE a.document_id = ? AND l.id > ? AND l.id <= ?
        ORDER BY l.id
    """,
    CONFIRMATION: """
        SELECT c.id, c.location_id, c.start, c."end", c.note, c.created_at,
               l.annotation_id, l.version_id, v.number AS version_number
        FROM confirmations c
        JOIN locations l ON l.id = c.location_id
        JOIN annotations a ON a.id = l.annotation_id
        JOIN versions v ON v.id = l.version_id
        WHERE a.document_id = ? AND c.id > ? AND c.id <= ?
        ORDER BY c.id
    """,
    DEPENDENCY: """
        SELECT id, document_id, prerequisite_id, dependent_id, created_at
        FROM dependencies
        WHERE document_id = ? AND id > ? AND id <= ?
        ORDER BY id
    """,
    COMPLETION: """
        SELECT id, document_id, annotation_id, note, created_at
        FROM completions
        WHERE document_id = ? AND id > ? AND id <= ?
        ORDER BY id
    """,
}


def _version_event(row: sqlite3.Row) -> dict:
    return {
        "type": VERSION,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "version_id": row["id"],
            "document_id": row["document_id"],
            "version_number": row["number"],
            "content": row["content"],
            "created_at": row["created_at"],
        },
    }


def _annotation_event(row: sqlite3.Row) -> dict:
    return {
        "type": ANNOTATION,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "annotation_id": row["id"],
            "document_id": row["document_id"],
            "body": row["body"],
            "selection": row["selection"],
            "left_context": row["left_context"],
            "right_context": row["right_context"],
            "orig_start": row["orig_start"],
            "orig_end": row["orig_end"],
            "created_version_id": row["created_version_id"],
            "created_version_number": row["created_version_number"],
            "created_at": row["created_at"],
        },
    }


def _location_event(row: sqlite3.Row) -> dict:
    # 只呈现落库时的原始迁移记录，不叠加后续确认的生效状态 / 范围；
    # version_number 是不可变引用键（版本只追加、编号不变）。
    return {
        "type": LOCATION,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "location_id": row["id"],
            "annotation_id": row["annotation_id"],
            "version_id": row["version_id"],
            "version_number": row["version_number"],
            "status": row["status"],
            "start": row["start"],
            "end": row["end"],
            "left_match": row["left_match"],
            "right_match": row["right_match"],
            "score": row["score"],
            "candidates": json.loads(row["candidates"]),
            "carried_from_location_id": row["carried_from"],
            "created_at": row["created_at"],
        },
    }


def _confirmation_event(row: sqlite3.Row) -> dict:
    return {
        "type": CONFIRMATION,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "confirmation_id": row["id"],
            "location_id": row["location_id"],
            "annotation_id": row["annotation_id"],
            "version_id": row["version_id"],
            "version_number": row["version_number"],
            "start": row["start"],
            "end": row["end"],
            "note": row["note"],
            "created_at": row["created_at"],
        },
    }


def _dependency_event(row: sqlite3.Row) -> dict:
    return {
        "type": DEPENDENCY,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "dependency_id": row["id"],
            "document_id": row["document_id"],
            "prerequisite_id": row["prerequisite_id"],
            "dependent_id": row["dependent_id"],
            "created_at": row["created_at"],
        },
    }


def _completion_event(row: sqlite3.Row) -> dict:
    return {
        "type": COMPLETION,
        "event_id": row["id"],
        "created_at": row["created_at"],
        "payload": {
            "completion_id": row["id"],
            "document_id": row["document_id"],
            "annotation_id": row["annotation_id"],
            "note": row["note"],
            "created_at": row["created_at"],
        },
    }


_EVENT_BUILDERS = {
    VERSION: _version_event,
    ANNOTATION: _annotation_event,
    LOCATION: _location_event,
    CONFIRMATION: _confirmation_event,
    DEPENDENCY: _dependency_event,
    COMPLETION: _completion_event,
}


def _merged_rows(
    con: sqlite3.Connection,
    document_id: int,
    start: Sequence[int],
    end: Sequence[int],
) -> Iterator[Tuple[SortKey, sqlite3.Row]]:
    """六类记录各自范围扫描后，按 (created_at, type_rank, id) 堆合并。"""
    streams = []
    for i, name in enumerate(TYPES):
        rank = TYPE_RANK[name]
        rows = con.execute(
            _STREAM_SQL[name], (document_id, start[i], end[i])
        ).fetchall()
        # 立即用各类型自己的 rank 构造排序键，避免生成器延迟求值时
        # 循环变量已指向最后一个类型。
        streams.append(
            iter([((row["created_at"], rank, row["id"]), row) for row in rows])
        )
    return heapq.merge(*streams, key=lambda item: item[0])


def read_events(
    con: sqlite3.Connection,
    document_id: int,
    start: Sequence[int],
    end: Sequence[int],
    limit: int,
    after: Optional[SortKey] = None,
) -> Tuple[list, bool]:
    """在调用方持有的读事务快照内读取 (start, end] 区间的合并事件。

    返回 ``(events, has_more)``：``events`` 为 ``after`` 之后至多 ``limit``
    条事件；``has_more`` 表示窗口内其后是否还有事件（单次合并多取一个
    排序键判定，调用方无需再扫一遍）。全程在同一快照内，不漏、不重、不乱序。
    """
    merged = _merged_rows(con, document_id, start, end)
    if after is not None:
        merged = (item for item in merged if item[0] > after)

    events = []
    has_more = False
    for (_, rank, _row_id), row in merged:
        if len(events) >= limit:
            has_more = True
            break
        events.append(_EVENT_BUILDERS[TYPES[rank - 1]](row))
    return events, has_more
