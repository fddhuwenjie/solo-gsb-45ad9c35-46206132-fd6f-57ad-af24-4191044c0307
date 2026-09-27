"""文档增量事件接口测试。

覆盖：
- 首轮全量：六类记录按 (created_at, type_rank, id) 稳定合并、水位向量；
- 分页：limit 切页不重 / 不漏 / 不乱序，next_cursor 始终沿用同一终点水位；
- 续拉：resume_cursor 只取 (旧终点, 新终点]，空轮也推进水位；
- 并发新增夹在翻页期间不会进入当前窗口，且不会在下一轮漏掉；
- 游标错误（版本 / 结构 / 范围 / 跨文档）→ 422；limit 越界 → 422；
- 事件载荷为落库时的稳定字段（location 不叠加确认后的生效状态）；
- 不串文档：其他文档的低 ID 行不进入水位与事件。
"""
from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import events as events_mod


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_doc(client, content, title="事件测试文档"):
    r = client.post("/documents", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def _make_ann(client, doc, text, selection, left="", right="", body="批注",
              start=None):
    if start is None:
        start = text.index(selection)
    r = client.post(
        f"/documents/{doc}/annotations",
        json={
            "body": body,
            "selection": selection,
            "left_context": left,
            "right_context": right,
            "start": start,
            "end": start + len(selection),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["annotation_id"]


def _new_version(client, doc, content):
    r = client.post(f"/documents/{doc}/versions", json={"content": content})
    assert r.status_code == 201, r.text
    return r.json()


def _drain(client, doc, limit, cursor=None):
    """从首轮（或指定游标）一路翻到末页，返回 (全部事件, 末页响应)。"""
    all_events = []
    seen_cursors = set()
    while True:
        params = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        page = client.get(f"/documents/{doc}/events", params=params)
        assert page.status_code == 200, page.text
        data = page.json()
        all_events.extend(data["events"])
        if data["resume_cursor"] is not None:
            return all_events, data
        cursor = data["next_cursor"]
        assert cursor is not None and cursor not in seen_cursors, "游标必须推进"
        seen_cursors.add(cursor)


def _sort_keys(events):
    return [
        (e["created_at"], events_mod.TYPE_RANK[e["type"]], e["event_id"])
        for e in events
    ]


@pytest.fixture(scope="module")
def populated(client):
    """造出六类事件齐全、且含歧义 + 确认场景的文档。"""
    v1 = "说：目标。说：目标。"
    doc = _make_doc(client, v1)
    ann = _make_ann(client, doc, v1, "目标", left="说：", right="。", body="甲")
    # v2：两处相同上下文 → ambiguous
    r = _new_version(client, doc, v1)
    assert r["results"][0]["status"] == "ambiguous"
    chosen = r["results"][0]["candidates"][0]
    # 第二条批注，用于依赖边
    ann2 = _make_ann(client, doc, v1, "说", body="乙", start=0)
    # 依赖：ann → ann2（方向无环即可）
    r = client.post(
        f"/documents/{doc}/dependencies",
        json={"relations": [
            {"prerequisite_id": ann2, "dependent_id": ann},
        ]},
    )
    assert r.status_code == 201, r.text
    dep_id = r.json()["dependencies"][0]["dependency_id"]
    # 确认 v2 的歧义
    r = client.post(
        f"/documents/{doc}/annotations/{ann}/confirmations",
        json={
            "version_number": 2,
            "start": chosen["start"],
            "end": chosen["end"],
            "note": "取前一处",
        },
    )
    assert r.status_code == 201, r.text
    # 完成记录
    r = client.post(
        f"/documents/{doc}/annotations/{ann2}/completions",
        json={"note": "乙先完成"},
    )
    assert r.status_code == 201, r.text
    completion_id = r.json()["completion_id"]
    # v3：确认后恢复迁移
    r = _new_version(client, doc, "说：目标。")
    assert r["results"][0]["status"] == "resolved"
    return {
        "doc": doc,
        "ann": ann,
        "ann2": ann2,
        "dep_id": dep_id,
        "completion_id": completion_id,
    }


# ---------------------------------------------------------------------------
# 首轮：合并顺序、类型全集、水位
# ---------------------------------------------------------------------------


def test_first_page_merge_and_watermark(client, populated):
    doc = populated["doc"]
    r = client.get(f"/documents/{doc}/events", params={"limit": 200})
    assert r.status_code == 200, r.text
    data = r.json()

    # 首轮就是末页（事件数 < 200）：给 resume_cursor，不给 next_cursor
    assert data["next_cursor"] is None
    assert isinstance(data["resume_cursor"], str)
    assert data["has_more"] is False
    assert data["window_start_watermark"] == {
        "versions": 0,
        "annotations": 0,
        "locations": 0,
        "confirmations": 0,
        "dependencies": 0,
        "completions": 0,
    }
    wm = data["window_end_watermark"]
    # 水位 = 本文档各表最大行 ID（全局自增），与首轮事件中各类 event_id
    # 的最大值一致；空槽为 0（本文档此前不存在确认等情况不适用——六类齐全）
    expected_max = {table: 0 for table in wm}
    table_by_type = {
        "version": "versions", "annotation": "annotations",
        "location": "locations", "confirmation": "confirmations",
        "dependency": "dependencies", "completion": "completions",
    }
    for e in events:
        table = table_by_type[e["type"]]
        expected_max[table] = max(expected_max[table], e["event_id"])
    assert wm == expected_max
    assert wm["confirmations"] >= 1 and wm["dependencies"] >= 1
    assert wm["completions"] >= 1

    events = data["events"]
    types = [e["type"] for e in events]
    # 六类齐全
    assert set(types) == {
        "version", "annotation", "location", "confirmation",
        "dependency", "completion",
    }
    # 全局按 (created_at, type_rank, id) 非严格有序
    keys = _sort_keys(events)
    assert keys == sorted(keys)
    # 第一版版本事件在最前；文档三个版本按版本号 1、2、3 给出
    version_events = [e for e in events if e["type"] == "version"]
    assert version_events[0]["payload"]["version_number"] == 1
    assert [e["payload"]["version_number"] for e in version_events] == [1, 2, 3]


def test_type_rank_breaks_created_at_ties(client):
    """同一写事务内的行共享 created_at：跨表合并严格按
    (created_at, rank, id) 有序，并列时类型 rank 小者在前。"""
    doc = _make_doc(client, "甲乙丙")
    # 建批注与其初始迁移记录在同一写事务插入（created_at 相同）
    _make_ann(client, doc, "甲乙丙", "甲", body="x", start=0)
    r = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    events = r["events"]
    keys = _sort_keys(events)
    assert keys == sorted(keys)
    tail = events[-2:]
    assert [e["type"] for e in tail] == ["annotation", "location"]
    assert tail[0]["created_at"] == tail[1]["created_at"]


# ---------------------------------------------------------------------------
# 分页：各种 limit 下不重 / 不漏 / 不乱序
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 7, 200])
def test_pagination_is_stable(client, populated, limit):
    doc = populated["doc"]
    full = client.get(
        f"/documents/{doc}/events", params={"limit": 200}
    ).json()["events"]

    events, last_page = _drain(client, doc, limit)
    assert len(events) == len(full)
    assert _sort_keys(events) == _sort_keys(full)
    # 去重：(type, event_id) 唯一
    identities = [(e["type"], e["event_id"]) for e in events]
    assert len(identities) == len(set(identities))
    # 每页至多 limit 条
    assert last_page["resume_cursor"] is not None


def test_mid_pages_keep_pinned_end_watermark(client, populated):
    """翻页过程中终点水位钉死；中途页只有 next_cursor。"""
    doc = populated["doc"]
    first = client.get(f"/documents/{doc}/events", params={"limit": 3}).json()
    assert first["has_more"] is True
    assert first["next_cursor"] and first["resume_cursor"] is None
    pinned = first["window_end_watermark"]

    cursor = first["next_cursor"]
    pages = [first]
    while True:
        page = client.get(
            f"/documents/{doc}/events", params={"limit": 3, "cursor": cursor}
        ).json()
        pages.append(page)
        assert page["window_end_watermark"] == pinned, "终点水位在翻页中不得移动"
        assert page["window_start_watermark"] == first["window_start_watermark"]
        if page["resume_cursor"] is not None:
            # 末页：起 = 止 = 旧终点
            assert page["window_start_watermark"] == pinned
            break
        cursor = page["next_cursor"]

    # 每页（除末页）恰满 limit；末页不满
    assert all(len(p["events"]) == 3 for p in pages[:-1])
    assert len(pages[-1]["events"]) <= 3


# ---------------------------------------------------------------------------
# 续拉：resume_cursor 与空轮推进
# ---------------------------------------------------------------------------


def test_resume_returns_only_new_events_and_advances_on_empty_round(client):
    doc = _make_doc(client, "第一版")
    first = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    first_ids = [(e["type"], e["event_id"]) for e in first["events"]]
    assert first["resume_cursor"]

    # 空轮：没有任何新增，仍返回推进后的 resume_cursor（水位保持一致），
    # 且不重复旧事件
    empty = client.get(
        f"/documents/{doc}/events",
        params={"limit": 200, "cursor": first["resume_cursor"]},
    ).json()
    assert empty["events"] == []
    assert empty["resume_cursor"]
    assert empty["window_start_watermark"] == first["window_end_watermark"]
    assert empty["window_end_watermark"] == first["window_end_watermark"]

    # 空轮之后再新增：只取增量
    _make_ann(client, doc, "第一版", "第一", body="新批注")
    incr = client.get(
        f"/documents/{doc}/events",
        params={"limit": 200, "cursor": empty["resume_cursor"]},
    ).json()
    new_ids = [(e["type"], e["event_id"]) for e in incr["events"]]
    assert set(new_ids).isdisjoint(first_ids)
    types = {t for t, _ in new_ids}
    # 新批注自带一条初始迁移记录
    assert types == {"annotation", "location"}
    assert incr["window_start_watermark"] == empty["window_end_watermark"]
    # 新窗口终点相对起点推进：新增 1 条批注、1 条迁移，版本无新增
    start_wm = incr["window_start_watermark"]
    end_wm = incr["window_end_watermark"]
    assert end_wm["versions"] == start_wm["versions"]
    assert end_wm["annotations"] - start_wm["annotations"] == 1
    assert end_wm["locations"] - start_wm["locations"] == 1
    # 续拉累计并集 = 最新全量
    latest = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    latest_ids = {(e["type"], e["event_id"]) for e in latest["events"]}
    assert set(first_ids) | set(new_ids) == latest_ids


def test_concurrent_inserts_during_paging_stay_in_next_round(client):
    """翻页期间并发新增：不进入当前窗口，但下一轮能取到（不漏）。"""
    doc = _make_doc(client, "基准文本XYZ")
    _make_ann(client, doc, "基准文本XYZ", "基准", body="a")
    first = client.get(f"/documents/{doc}/events", params={"limit": 2}).json()
    pinned = first["window_end_watermark"]
    assert first["next_cursor"]

    # “并发新增”：拿到中途游标后写入更多数据
    _new_version(client, doc, "基准文本XYZ-修订")
    _make_ann(client, doc, "基准文本XYZ-修订", "修订", body="b",
              start=len("基准文本XYZ-"))

    # 用旧的中途游标继续翻完：窗口仍是旧快照范围
    mid_events, last_page = _drain(client, doc, 2, cursor=first["next_cursor"])
    assert last_page["window_end_watermark"] == pinned
    drained_ids = {
        (e["type"], e["event_id"])
        for e in first["events"] + mid_events
    }
    # 并发新增绝不可能出现在本轮
    latest = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    new_rows = {
        (e["type"], e["event_id"])
        for e in latest["events"]
        if (e["type"], e["event_id"]) not in drained_ids
    }
    assert new_rows, "并发新增应存在于最新全量中"
    assert new_rows.isdisjoint(drained_ids)

    # 下一轮（resume_cursor）恰好取出并发新增，不多不少
    nxt = client.get(
        f"/documents/{doc}/events",
        params={"limit": 200, "cursor": last_page["resume_cursor"]},
    ).json()
    next_ids = {(e["type"], e["event_id"]) for e in nxt["events"]}
    assert next_ids == new_rows
    assert nxt["window_start_watermark"] == pinned


# ---------------------------------------------------------------------------
# 载荷：稳定业务字段，不计算派生状态
# ---------------------------------------------------------------------------


def test_payloads_keep_persisted_fields_not_derived(client, populated):
    doc = populated["doc"]
    ann = populated["ann"]
    events = client.get(
        f"/documents/{doc}/events", params={"limit": 200}
    ).json()["events"]
    by_key = {(e["type"], e["event_id"]): e for e in events}

    # v2 上的歧义迁移记录：即使后来被人工确认，事件里仍保持落库时的
    # 原始 ambiguous / 无范围，而不是查询时刻的 confirmed 生效视图
    v2_locs = [
        e for e in events
        if e["type"] == "location"
        and e["payload"]["annotation_id"] == ann
        and e["payload"]["version_number"] == 2
    ]
    assert len(v2_locs) == 1
    loc_payload = v2_locs[0]["payload"]
    assert loc_payload["status"] == "ambiguous"
    assert loc_payload["start"] is None and loc_payload["end"] is None
    assert loc_payload["carried_from_location_id"] is None
    assert len(loc_payload["candidates"]) == 2
    assert set(loc_payload) == {
        "location_id", "annotation_id", "version_id", "version_number",
        "status", "start", "end", "left_match", "right_match", "score",
        "candidates", "carried_from_location_id", "created_at",
    }

    # 确认事件独立存在，载荷指向该迁移记录
    conf = [e for e in events if e["type"] == "confirmation"]
    assert len(conf) == 1
    cp = conf[0]["payload"]
    assert cp["location_id"] == loc_payload["location_id"]
    assert cp["annotation_id"] == ann
    assert cp["version_number"] == 2
    assert cp["note"] == "取前一处"
    assert cp["start"] == loc_payload["candidates"][0]["start"]

    # 顺延占位（v3 之前不存在顺延；这里检查字段形态在 ambiguous 记录上成立）
    # 版本事件含全文；完成 / 依赖载荷字段稳定
    comp = by_key[("completion", populated["completion_id"])]
    assert comp["payload"]["annotation_id"] == populated["ann2"]
    dep = by_key[("dependency", populated["dep_id"])]
    assert dep["payload"] == {
        "dependency_id": populated["dep_id"],
        "document_id": doc,
        "prerequisite_id": populated["ann2"],
        "dependent_id": populated["ann"],
        "created_at": dep["payload"]["created_at"],
    }
    ver1 = [
        e for e in events
        if e["type"] == "version" and e["payload"]["version_number"] == 1
    ][0]
    assert ver1["payload"]["content"] == "说：目标。说：目标。"


# ---------------------------------------------------------------------------
# 错误：404、limit 422、游标 422
# ---------------------------------------------------------------------------


def test_document_not_found(client):
    r = client.get("/documents/999999/events")
    assert r.status_code == 404


@pytest.mark.parametrize("bad_limit", [0, -1, 201, 1000])
def test_limit_out_of_range_422(client, populated, bad_limit):
    r = client.get(
        f"/documents/{populated['doc']}/events", params={"limit": bad_limit}
    )
    assert r.status_code == 422


def _encode_cursor(obj):
    return base64.urlsafe_b64encode(
        json.dumps(obj, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")


def test_cursor_errors_422(client, populated):
    doc = populated["doc"]
    other = _make_doc(client, "无关文档")
    first = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    wm = first["window_end_watermark"]
    end = [wm[k] for k in (
        "versions", "annotations", "locations", "confirmations",
        "dependencies", "completions",
    )]

    def with_cursor(token, d=doc):
        return client.get(
            f"/documents/{d}/events", params={"limit": 100, "cursor": token}
        )

    # 非 base64 / 乱码
    assert with_cursor("not-a-cursor!!!").status_code == 422
    # 篡改的合法 next_cursor（改最后一个字符）也应 422
    tampered = first["resume_cursor"][:-1] + (
        "A" if first["resume_cursor"][-1] != "A" else "B"
    )
    assert with_cursor(tampered).status_code == 422

    cases = [
        # 版本不符
        {"v": 99, "doc": doc, "start": [0] * 6, "end": end, "last": None},
        # 跨文档
        {"v": 1, "doc": other, "start": [0] * 6, "end": end, "last": None},
        # 起 > 止
        {"v": 1, "doc": doc, "start": [9] * 6, "end": [0] * 6, "last": None},
        # 水位数组长度错误
        {"v": 1, "doc": doc, "start": [0] * 5, "end": end, "last": None},
        # last 结构错误
        {"v": 1, "doc": doc, "start": [0] * 6, "end": end, "last": [1, 2, 3]},
        # last 指向窗口外
        {
            "v": 1, "doc": doc, "start": [0] * 6, "end": [0] * 6,
            "last": ["2026-01-01T00:00:00.000Z", 1, 1],
        },
        # doc 非整数
        {"v": 1, "doc": "1", "start": [0] * 6, "end": end, "last": None},
    ]
    for obj in cases:
        r = with_cursor(_encode_cursor(obj))
        assert r.status_code == 422, obj
        detail = r.json()["detail"]
        assert isinstance(detail, dict) and detail["error"]

    # 跨文档：把文档 1 的真实游标拿到文档 2 上用
    r = with_cursor(first["resume_cursor"], d=other)
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "cross_document"


def test_resume_cursor_reusable_is_idempotent(client, populated):
    """resume_cursor 表示水位而非消费动作：重复使用结果一致（空轮）。"""
    doc = populated["doc"]
    first = client.get(f"/documents/{doc}/events", params={"limit": 200}).json()
    r1 = client.get(
        f"/documents/{doc}/events",
        params={"limit": 200, "cursor": first["resume_cursor"]},
    ).json()
    r2 = client.get(
        f"/documents/{doc}/events",
        params={"limit": 200, "cursor": first["resume_cursor"]},
    ).json()
    assert r1["events"] == []
    assert r2["events"] == []
    assert r1["window_end_watermark"] == r2["window_end_watermark"]


def test_events_are_scoped_per_document(client):
    """另一个文档的行（可能 ID 更小）不得进入本文档窗口与水位。"""
    d1 = _make_doc(client, "文档一")
    d2 = _make_doc(client, "文档二")
    _make_ann(client, d2, "文档二", "文档", body="属于文档二")
    r = client.get(f"/documents/{d1}/events", params={"limit": 200}).json()
    # location / confirmation 事件本身不带 document_id（document 维度由
    # 归属连接隐式确定）；带 document_id 的事件必须指向 d1
    assert all(
        e["payload"].get("document_id", d1) == d1 for e in r["events"]
    )
    assert [e["type"] for e in r["events"]] == ["version"]
    wm = r["window_end_watermark"]
    # d1 只有建文档时的版本行：annotations 等表的文档级水位为 0，
    # versions 水位等于该版本事件自身的行 ID（不会取到 d2 的行）
    assert wm["annotations"] == 0
    assert wm["locations"] == 0
    assert wm["versions"] == r["events"][0]["event_id"]
