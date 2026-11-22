"""引擎层测试：KV 缓存正确性、块内分叉、写时复制、满池分叉、原子拒绝、回收、并发。"""
import threading

import numpy as np
import pytest

from app.blocks import CapacityError
from app.engine import Engine, ValidationError

VOCAB = 64


def make_engine(num_blocks=8, block_size=4, d_model=16, seed=7):
    return Engine.from_config(vocab_size=VOCAB, d_model=d_model,
                              block_size=block_size, num_blocks=num_blocks,
                              seed=seed)


def cached_logits(eng, session_id, branch_id):
    """从块缓存路径算出的 logits（即预测端实际走的路径）。"""
    with eng.lock:
        br = eng.sessions[session_id].branches[branch_id]
        _, _, q = eng.model.kvq(br.tokens[-1], br.length - 1)
        K, V = eng._gather_kv(br)
        return eng.model.logits_from_kv(q, K, V)


def assert_matches_scratch(eng, session_id, branch_id, tokens):
    """缓存路径的预测必须与按各自词元从头计算一致。"""
    got = cached_logits(eng, session_id, branch_id)
    want = eng.model.scratch_logits(tokens)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    assert int(np.argmax(got)) == int(np.argmax(want))


def test_cached_prediction_matches_scratch_across_appends():
    eng = make_engine()
    r = eng.create_session([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])  # 10 词元，跨 3 块
    sid, bid = r["session_id"], r["branch"]["id"]
    eng.append(sid, bid, [5, 8, 9])
    eng.append(sid, bid, [7, 9])
    tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    assert_matches_scratch(eng, sid, bid, tokens)
    pred = eng.predict(sid, bid)["prediction"]
    assert pred["argmax"] == int(np.argmax(eng.model.scratch_logits(tokens)))


def test_in_block_fork_continues_outside_visible_prefix_then_isolates():
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([5, 6, 7, 8, 9, 10])  # 块0满，块1占2槽
    sid = r["session_id"]
    main = r["branch"]["id"]

    f = eng.fork(sid, main, 5)  # 分叉点 5 落在块1内部（块1只有槽0对子分支可见）
    child = f["branch"]["id"]
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert c.block_ids == p.block_ids            # 前缀块共享
        assert eng.pool.refcount[p.block_ids[0]] == 2
        assert eng.pool.refcount[p.block_ids[1]] == 2
        shared_last = p.block_ids[1]                 # 被共享的未满块

    used_before = eng.stats()["used_blocks"]
    pred_child_before = eng.predict(sid, child)["prediction"]["argmax"]
    # 父分支写入块1的槽2、3 —— 落在子分支可见前缀（仅槽0）之外，直接原地续写
    eng.append(sid, main, [11, 12])
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        assert p.block_ids[1] == shared_last         # 没有写时复制
        assert eng.pool.refcount[shared_last] == 2   # 仍共享
    assert eng.stats()["used_blocks"] == used_before  # 未新增块
    # 子分支可见前缀的缓存未被覆盖，预测不变且与从头计算一致
    assert eng.predict(sid, child)["prediction"]["argmax"] == pred_child_before
    assert_matches_scratch(eng, sid, main, [5, 6, 7, 8, 9, 10, 11, 12])
    assert_matches_scratch(eng, sid, child, [5, 6, 7, 8, 9])

    eng.append(sid, child, [21, 22])                 # 子分支写入位置被子分支自己和他方…
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert c.block_ids[0] == p.block_ids[0]      # 满块仍共享
        assert c.block_ids[1] != p.block_ids[1]      # 争用的未满尾块已隔离
        assert p.block_ids[1] == shared_last         # 原块留在父分支（父分支内容受保护）
        assert eng.pool.refcount[shared_last] == 1
        assert p.tokens == [5, 6, 7, 8, 9, 10, 11, 12]

    # 两条分支的预测都与各自从头计算一致
    assert_matches_scratch(eng, sid, main, [5, 6, 7, 8, 9, 10, 11, 12])
    assert_matches_scratch(eng, sid, child, [5, 6, 7, 8, 9, 21, 22])
    # 子分支的隔离写入不影响父分支预测
    assert eng.predict(sid, main)["prediction"]["argmax"] == \
        int(np.argmax(eng.model.scratch_logits([5, 6, 7, 8, 9, 10, 11, 12])))


def test_full_pool_longer_branch_continues_in_shared_tail():
    """池满 + 块内分叉后，较长分支在共享尾块的可见前缀之外直接续写，不再误报块不足。"""
    eng = make_engine(num_blocks=3, block_size=4)    # 容量 12 词元
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 占满 3 块，尾块占 2 槽
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 9)["branch"]["id"]   # 子分支只看得见尾块槽0
    assert eng.stats()["free_blocks"] == 0

    # 父分支写尾块槽2、3：子分支可见前缀（槽0）之外，应原地续写且零新增块
    out = eng.append(sid, main, [11, 12])
    assert out["branch"]["length"] == 12
    assert eng.stats()["free_blocks"] == 0           # 没有新占块
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert p.block_ids == c.block_ids            # 仍然共享
        assert eng.pool.refcount[p.block_ids[-1]] == 2
    # 子分支的可见前缀保持不变，预测与从头计算一致
    assert_matches_scratch(eng, sid, child, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert_matches_scratch(eng, sid, main, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # 池已满：子分支再无空闲槽可写（它接下来要写槽1，正被父分支可见），整体拒绝
    before = eng.get_session(sid)
    with pytest.raises(CapacityError):
        eng.append(sid, child, [21])
    assert eng.get_session(sid) == before
    assert eng.stats()["used_blocks"] == 3


def test_contended_tail_isolation_preserves_tokens_and_cache():
    """两条分支同时可见尾块写入位置时，隔离写入并保住对方的词元与 KV。"""
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([5, 6, 7, 8, 9, 10])      # 块1占2槽
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 6)["branch"]["id"]   # 两个分支都可见槽0、1
    pred_child_before = eng.predict(sid, child)["prediction"]["argmax"]
    child_tokens_before = list(eng.get_session(sid)["branches"][1]["tokens"])

    eng.append(sid, main, [11, 12])                  # 父分支原地续写槽2、3
    eng.append(sid, child, [21, 22])                 # 子分支要写槽2，已被父分支可见 -> 隔离

    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert c.block_ids[0] == p.block_ids[0]      # 满块继续共享
        assert c.block_ids[1] != p.block_ids[1]      # 尾块已分家
        # 隔离时原尾块不能回收：父分支的词元与缓存仍在其中
        assert p.tokens == [5, 6, 7, 8, 9, 10, 11, 12]
        assert eng.pool.refcount[p.block_ids[1]] == 1
    # 父分支词元、缓存、预测都没被子分支覆盖
    assert_matches_scratch(eng, sid, main, [5, 6, 7, 8, 9, 10, 11, 12])
    assert_matches_scratch(eng, sid, child, [5, 6, 7, 8, 9, 10, 21, 22])
    assert eng.get_session(sid)["branches"][1]["tokens"] == child_tokens_before + [21, 22]


def test_append_cost_counts_whole_token_run_and_visibility():
    """跨块追加的块数按整串词元所需 + 写入时的可见范围计算。"""
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6])       # 2 块，尾块 2/4
    sid, main = r["session_id"], r["branch"]["id"]

    # 独占尾块时：追加 6 个词元 -> 长度 12 -> 3 个逻辑块，只需新增 1 块
    out = eng.append(sid, main, [7, 8, 9, 10, 11, 12])
    assert out["branch"]["length"] == 12
    assert eng.stats()["used_blocks"] == 3

    # 块边界分叉，尾块是整块：追加 1 词元跨入新块，只需 1 块
    child = eng.fork(sid, main, 12)["branch"]["id"]
    out = eng.append(sid, child, [13])
    assert out["branch"]["length"] == 13
    assert eng.stats()["used_blocks"] == 4

    # 私有未满尾块：不隔离，按整串长度新增 1 块
    used_before = eng.stats()["used_blocks"]
    out = eng.append(sid, child, [14, 15, 16, 17])
    assert out["branch"]["length"] == 17
    assert eng.stats()["used_blocks"] == used_before + 1
    assert_matches_scratch(eng, sid, main, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert_matches_scratch(eng, sid, child,
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])


def test_contended_tail_plus_new_blocks_cost():
    """共享且争用的未满尾块 + 跨块：隔离 1 块 + 新增整块，整串词元一次算清。"""
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6])       # 2 块，尾块 2/4
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 6)["branch"]["id"]   # 双方都可见尾块槽0、1
    eng.append(sid, main, [7, 8])                    # 父分支填满尾块，长度 8
    assert eng.stats()["used_blocks"] == 2

    # 子分支长度 6，尾块被父分支可见：追加 4 词元 -> 长度 10 -> 3 逻辑块
    # 需隔离 1 块 + 新增 1 整块 = 2 块
    out = eng.append(sid, child, [21, 22, 23, 24])
    assert out["branch"]["length"] == 10
    assert eng.stats()["used_blocks"] == 4
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert p.block_ids[0] == c.block_ids[0]      # 满块共享
        assert p.block_ids[1] != c.block_ids[1]      # 争用尾块已隔离
    assert_matches_scratch(eng, sid, main, [1, 2, 3, 4, 5, 6, 7, 8])
    assert_matches_scratch(eng, sid, child, [1, 2, 3, 4, 5, 6, 21, 22, 23, 24])


def test_rejected_append_keeps_branch_and_block_usage():
    """池满且无可写空槽时，整次拒绝：分支、块表、块占用原样保留。"""
    eng = make_engine(num_blocks=3, block_size=4)    # 容量 12 词元
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 9)["branch"]["id"]
    eng.append(sid, main, [11, 12])                  # 填满尾块
    before = eng.get_session(sid)
    used_before = eng.stats()["used_blocks"]
    with pytest.raises(CapacityError):
        eng.append(sid, child, [21, 22])             # 需隔离 1 块，池已满
    assert eng.get_session(sid) == before
    assert eng.stats()["used_blocks"] == used_before


    eng = make_engine(num_blocks=3, block_size=4)   # 容量 12 词元
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])
    sid, main = r["session_id"], r["branch"]["id"]
    eng.create_session([9, 10, 11, 12])             # 占满最后一块
    assert eng.stats()["free_blocks"] == 0

    f1 = eng.fork(sid, main, 8)                     # 块边界分叉：不分配块
    f2 = eng.fork(sid, main, 3)                     # 块内分叉：同样不分配块
    assert eng.stats()["free_blocks"] == 0
    assert f2["branch"]["block_ids"] == f1["branch"]["block_ids"][:1]

    with pytest.raises(CapacityError):              # 但任何追加都放不下
        eng.append(sid, f1["branch"]["id"], [9])
    # 拒绝后状态不变
    s = eng.get_session(sid)
    lens = {b["name"]: b["length"] for b in s["branches"]}
    assert lens == {"main": 8, "main@8": 8, "main@3": 3}
    assert eng.stats()["used_blocks"] == 3


def test_append_rejected_atomically_when_too_large():
    eng = make_engine(num_blocks=4, block_size=4)   # 容量 16 词元
    r = eng.create_session([1, 2, 3, 4])
    sid, bid = r["session_id"], r["branch"]["id"]
    eng.create_session([5, 6, 7, 8, 9, 10, 11, 12])  # 再占 2 块，只剩 1 块空闲
    before = eng.get_session(sid)["branches"][0]
    used_before = eng.stats()["used_blocks"]

    with pytest.raises(CapacityError):
        eng.append(sid, bid, list(range(20, 28)))   # 8 个词元需要 2 个新块，只剩 1 个

    after = eng.get_session(sid)["branches"][0]
    assert after == before                          # 分支内容、块表完全没变
    assert eng.stats()["used_blocks"] == used_before


def test_close_branch_recycles_only_unreferenced_blocks():
    eng = make_engine(num_blocks=6, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])  # 块0、块1
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 8)["branch"]["id"]    # 共享块0、块1
    grand = eng.fork(sid, main, 4)["branch"]["id"]    # 只共享块0
    assert eng.stats()["used_blocks"] == 2

    eng.close_branch(sid, main)                       # 块0:3->2，块1:2->1，都不回收
    assert eng.stats()["used_blocks"] == 2
    eng.close_branch(sid, child)                      # 块1 归零回收，块0 仍被引用
    assert eng.stats()["used_blocks"] == 1
    eng.close_branch(sid, grand)                      # 全部回收
    assert eng.stats()["used_blocks"] == 0
    assert eng.stats()["free_blocks"] == 6


def test_fork_at_zero_and_empty_branch():
    eng = make_engine()
    r = eng.create_session([1, 2, 3])
    sid, main = r["session_id"], r["branch"]["id"]
    empty = eng.fork(sid, main, 0)["branch"]
    assert empty["block_ids"] == [] and empty["length"] == 0
    with pytest.raises(ValidationError):
        eng.predict(sid, empty["id"])
    eng.append(sid, empty["id"], [7, 7, 7, 7, 7])
    assert_matches_scratch(eng, sid, empty["id"], [7, 7, 7, 7, 7])


def test_concurrent_appends_respect_total_capacity():
    eng = make_engine(num_blocks=8, block_size=4)     # 容量 32 词元
    r = eng.create_session([1, 2, 3, 4])              # 占 1 块，空闲 7
    sid, main = r["session_id"], r["branch"]["id"]
    other = eng.fork(sid, main, 4)["branch"]["id"]
    branches = [main, other]
    results, errors = [], []

    def worker(i):
        try:
            eng.append(sid, branches[i % 2], [10 + i] * 8)  # 每次需 2 个新块
            results.append(i)
        except CapacityError:
            errors.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 空闲 7 块、每次追加需 2 块：成功次数至多 3 次，且必有被拒
    assert 1 <= len(results) <= 3
    assert len(results) + len(errors) == 10
    st = eng.stats()
    assert st["used_blocks"] <= st["num_blocks"]
    assert st["used_blocks"] == 1 + 2 * len(results)

    # 每条分支的长度 = 4 + 8 * 其成功次数，且预测与从头计算一致
    sess = eng.get_session(sid)
    for b in sess["branches"]:
        n_ok = sum(1 for i in results if branches[i % 2] == b["id"])
        assert b["length"] == 4 + 8 * n_ok
        assert_matches_scratch(eng, sid, b["id"], b["tokens"])


@pytest.mark.parametrize("round_no", range(5))
def test_concurrent_contention_on_shared_tail(round_no):
    """两条分支并发争用同一未满尾块：各自隔离写入，互不覆盖对方词元与缓存。"""
    eng = make_engine(num_blocks=10, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6])       # 块1占2槽
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 6)["branch"]["id"]   # 双方都可见尾块槽0、1
    assert eng.stats()["used_blocks"] == 2

    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker(bid, start):
        try:
            barrier.wait(timeout=5)
            eng.append(sid, bid, [start + k for k in range(4)])  # 各需隔离 1 + 新增 1
            results.append(bid)
        except CapacityError:
            errors.append(bid)

    threads = [threading.Thread(target=worker, args=(main, 10)),
               threading.Thread(target=worker, args=(child, 30))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors and len(results) == 2         # 空闲 8 块，两次都放得下
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert p.block_ids[0] == c.block_ids[0]      # 满块仍共享
        assert p.block_ids[1] != c.block_ids[1]      # 两分支各自持有私有尾块
    # 双方的词元、缓存、预测都与各自从头计算一致，完全没有互相覆盖
    assert eng.get_session(sid)["branches"][0]["tokens"] == [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    assert eng.get_session(sid)["branches"][1]["tokens"] == [1, 2, 3, 4, 5, 6, 30, 31, 32, 33]
    assert_matches_scratch(eng, sid, main, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13])
    assert_matches_scratch(eng, sid, child, [1, 2, 3, 4, 5, 6, 30, 31, 32, 33])


def test_concurrent_contention_rejects_whole_append_when_short():
    """争用尾块的并发续写按整串所需块数算：后到者需隔离+新增、空闲不足则整体拒绝。"""
    eng = make_engine(num_blocks=4, block_size=4)    # 初始占 2 块，空闲 2
    r = eng.create_session([1, 2, 3, 4, 5, 6])
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 6)["branch"]["id"]   # 双方都只可见尾块槽0、1

    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker(bid, start):
        try:
            barrier.wait(timeout=5)
            eng.append(sid, bid, [start + k for k in range(4)])  # -> 长度 10，3 个逻辑块
            results.append(bid)
        except CapacityError:
            errors.append(bid)

    threads = [threading.Thread(target=worker, args=(main, 10)),
               threading.Thread(target=worker, args=(child, 30))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 先到者写槽2、3（在对方可见前缀之外）只需新增 1 块；后到者写同一尾块
    # 已被先到者可见，需隔离 1 块 + 新增 1 块 = 2 块，但只剩 1 块 -> 整体拒绝
    assert len(results) == 1 and len(errors) == 1
    assert eng.stats()["used_blocks"] == 3           # 2 原有 + 先到者新增的 1 块
    loser = errors[0]
    # 被拒分支原样保留，与从头计算一致
    lb = next(b for b in eng.get_session(sid)["branches"] if b["id"] == loser)
    assert lb["tokens"] == [1, 2, 3, 4, 5, 6]
    assert_matches_scratch(eng, sid, loser, [1, 2, 3, 4, 5, 6])
    winner = results[0]
    wb = next(b for b in eng.get_session(sid)["branches"] if b["id"] == winner)
    assert_matches_scratch(eng, sid, winner, wb["tokens"])
