---
title: 受け方向詰み探索の GPU 検証手順を verification.md に追加する
date: 2026-08-01
status: applied
applied_in: PENDING
target:
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: §4.7 に受け方向詰み探索 (`--ab-mode defmate`) の検証手順を追加する

## Trigger

user 指示「Colab の L4 で GPU 環境確認を実施するので，検証方法を提示して
ほしい」「Colab の python セルで実行する形式に修正したうえで承認する」．
PR #424 (`8412f48`) でマージした受け方向詰み探索の GPU 検証．

CPU 側の全数走査は済んでいる (gate 該当 7.6% / 30 局中 7 局に避けられた敗着)．
GPU で確かめるのは **(a) 機構が実機で発火するか / (b) CPU 競合で探索速度を
奪わないか / (c) 棋力に効くか** の 3 点．

## 前提 — 計器

発火量を数えられなければ「効果なし」と結論してはいけない (計測規律 §10.2)．
`SearchStats` に 2 つのカウンタがある (PR #425):

- `defensive_mates` — 受け方向が「手番側が詰まされる」と証明した回数
  (root + 王手中の葉)．**0 なら機構が発火していない**
- `filtered_root_moves` — 敗着として除外した root 候補手の数．
  **0 なら着手選択に影響していない**

`maou search` の Stats 行と PyO3 `PySearchResult` に出る
(`maou analyze-game` も同じ `SearchEngine` 経由なので棋譜の再解析で拾える)．

---

# 検証手順 (本レビューの追加対象 — verification.md §4.7)

環境構築は §1 のとおり (wheel + `ldconfig` + モデル配置)．以下は
**その続きから流す Colab の python セル列**．

## セル 1 — 共通設定と pin 局面

```python
# 4.7-1. 共通設定．以降のセルはこの変数を使う．
import json, re, subprocess, statistics, time

MODEL = "/content/model_fp16.onnx"
GPU = ["--model-path", MODEL, "--tensorrt", "--cuda",
       "--threads", "1", "--batch-size", "64",
       "--trt-cache-dir", "/content/trt_cache"]

# 自己対局で実際に踏んだ局面 (game_0006)．
BLUNDER = "l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1NR3/1G1PPg3/1S2K1+b2/LN7 b BSgsnl8p 117"
MATED   = "l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1N4/1G1PPR3/1S2K1+b2/LN1s5 b BGSgnl8p 119"
QUIET   = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

def run_search(sfen, time_ms=11000, defensive=True):
    """maou search を叩いて bestmove / 候補 / Stats を返す."""
    cmd = (["maou", "search", "--sfen", sfen, "--time-ms", str(time_ms)]
           + GPU + (["--defensive-mate"] if defensive else ["--no-defensive-mate"]))
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    best = re.search(r"^Bestmove:\s*(\S+)", out, re.M)
    winrate = re.search(r"^WinRate:\s*([\d.]+)", out, re.M)
    stats = dict(re.findall(r"(\w+)=([\w.-]+)", re.search(r"^Stats:.*", out, re.M).group(0)))
    cands = re.findall(r"^(\S+) \(visits=(\d+),.*?(proven=\w+)?\)$", out, re.M)
    return {"best": best.group(1) if best else None,
            "winrate": float(winrate.group(1)) if winrate else None,
            "stats": stats, "cands": cands, "raw": out}

print("ok")
```

TensorRT 有効時は teardown 回避のため destructor を走らせずに終了するが
(§8.5)，**出力は flush 済みで exit code も 0** なので `subprocess` で問題なく
拾える．

## セル 2 — Stage A: 機構が実機で発火するか

```python
# 4.7-2. Stage A: pin 局面 3 つで発火と着手を確認する (各 11 秒 × 4 回 ≒ 1 分)．
def check(label, cond, detail):
    print(f"[{'OK ' if cond else 'NG '}] {label}: {detail}")
    return cond

ok = True

# A-1 敗着局面 (まだ受かる / 合法手 4)．
r = run_search(BLUNDER); rb = run_search(BLUNDER, defensive=False)
n_loss = sum(1 for c in r["cands"] if c[2])
ok &= check("A-1 bestmove", r["best"] == "5h6i", f"{r['best']} (対照 {rb['best']})")
ok &= check("A-1 除外手数", n_loss == 3, f"proven=loss が {n_loss} 手")
ok &= check("A-1 発火", int(r["stats"]["filtered_root_moves"]) >= 1,
            f"filtered_root_moves={r['stats']['filtered_root_moves']} "
            f"defensive_mates={r['stats']['defensive_mates']}")
for mv, v, pv in r["cands"]:
    print(f"      {mv:6s} visits={int(v):>8,} {pv or ''}")

# A-2 被詰み局面 (もう詰んでいる)．
r = run_search(MATED); rb = run_search(MATED, defensive=False)
ok &= check("A-2 勝率", r["winrate"] == 0.0, f"{r['winrate']} (対照 {rb['winrate']})")
ok &= check("A-2 発火", int(r["stats"]["defensive_mates"]) >= 1,
            f"defensive_mates={r['stats']['defensive_mates']}")

# A-3 静かな局面 (偽陽性ゼロ)．
r = run_search(QUIET)
ok &= check("A-3 偽陽性なし",
            int(r["stats"]["defensive_mates"]) == 0
            and int(r["stats"]["filtered_root_moves"]) == 0,
            f"defensive_mates={r['stats']['defensive_mates']} "
            f"filtered_root_moves={r['stats']['filtered_root_moves']}")

print("\nStage A:", "PASS" if ok else "FAIL — 下の判定表を参照")
```

| 判定 | 期待 | 外れたときに疑うこと |
|---|---|---|
| A-1 bestmove = `5h6i` | 唯一の受かる手 | 対照と同じ手なら未発火か gate 外れ |
| A-1 `proven=loss` が 3 手 | 4f4g(mate-41) / 5h5i(mate-1) / 5h6h(mate-29) | 予算不足なら `--time-ms` を上げる |
| A-2 `WinRate = 0.0` | 被詰みの正直な報告 | 対照は 0.49 前後の互角 |
| A-3 両カウンタ 0 | 偽陽性ゼロ | **0 でなければ即調査** (偽の被詰みは自滅につながる) |

**`filtered_root_moves` が 3 未満でも異常ではない** — 木の proven 伝播
(leaf-mate 経由) が先に確定させた手はフィルタが上書きしないため内訳は動く．
**見るべきは「除外が 3 手」と「bestmove = 5h6i」**．

DevContainer / mock 評価器での実測 (参考値．GPU では内訳が動く):

```
Bestmove: 5h6i
5h6i (visits=990,    winrate=0.4846)               ← 選ばれた
5h6h (visits=541509, winrate=0.4987, proven=loss)
4f4g (visits=482872, winrate=0.4988, proven=loss)  ← 対局で選んだ敗着
5h5i (visits=1744,   winrate=0.1197, proven=loss)
Stats: ... leaf_mates=6439 defensive_mates=19 filtered_root_moves=2 ...
```

**visits では敗着が 50 万回で圧倒的なのに 990 回の手が選ばれる** — MCTS の
訪問分布では敗着が勝っており，確定値の除外だけが手を変えている．
A-2 は `WinRate 0.0000` / `defensive_mates=50` / `filtered_root_moves=0`
(既に詰みなら候補選別はしない = 設計どおり)．

## セル 3 — Stage B: CPU 競合で探索速度を奪わないか

```python
# 4.7-3. Stage B: 静かな局面で nps を on/off 比較する (各 15 秒 × 2 回)．
import os
print("vCPU:", os.cpu_count(),
      "/ 常駐スレッド = MCTS 1 + 攻め dfpn 1 + 受け dfpn 1 + leaf-mate 1 = 4")

on  = run_search(QUIET, time_ms=15000, defensive=True)
off = run_search(QUIET, time_ms=15000, defensive=False)
nps_on, nps_off = int(on["stats"]["nps"]), int(off["stats"]["nps"])
drop = (1 - nps_on / nps_off) * 100
print(f"nps on={nps_on:,} off={nps_off:,} → 低下 {drop:.1f}%")
print(f"充填率 on={float(on['stats']['avg_batch'])/64:.2f} "
      f"off={float(off['stats']['avg_batch'])/64:.2f} "
      f"(外れ値を見たら充填率と collisions を必ず併記する — §1)")
print("判定:", "許容 (<5%)" if drop < 5 else "要調査 (>=5%) — 並列度は上げない")
```

**静かな局面で測る**こと．gate 該当は全局面の 7.6% (終盤 21.6%) なので，
静かな局面での低下はそのまま「常時払う税」になる．
5% 以上なら `--defensive-mate-threads` を上げずに Stage C へ進む．

## セル 4 — Stage C: 棋力 A/B

```python
# 4.7-4. Stage C: A/B 40 局．持ち時間モード必須 / --parallel 1 必須 / GPU 必須．
!maou selfplay --ab-mode defmate --clock-ms 300000 --inc-ms 10000 \
    --games 40 --parallel 1 \
    --model-path /content/model_fp16.onnx --tensorrt --cuda \
    --threads 1 --batch-size 64 --trt-cache-dir /content/trt_cache \
    --output /content/ab_defmate.jsonl --kifu-dir /content/kifu_defmate
```

regime ゲート (先に通すこと):

- **持ち時間モード必須** — 固定 playout 予算では敗着フィルタが使える壁時計が
  変わる (フィルタは MCTS 終了の停止フラグで打ち切られる)
- **`--parallel 1` 必須** — フィルタが CPU を使うので同時対局は互いを歪める
- **GPU 必須** — CPU 22 p/s では手の分布が別物になる

**期待効果の見積り**: CPU 全数走査では 30 局中 7 局 (23%) に「避けられた
強制詰みの敗着」があった．1 局あたり平均 0.23 手の改善であり，
**n=40 が検出できる ~150 Elo 級に届くかは未知**．有意にならなくても
機構が効いていないことにはならない — だから Stage D を併せて見る．

## セル 5 — Stage D: 棋譜による直接確認 (Elo より感度が高い)

```python
# 4.7-5. Stage D: A/B の棋譜を走査し「避けられた敗着」を A/B で数える．
#   1. 合法手 <= 16 を gate  2. 各合法手を指した後を攻め方向 dfpn で判定
#   3. 「指した手が敗着 かつ 安全な代替手が存在した」を数える
import glob, re
from maou._rust import maou_shogi as S

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
STAGES = [1_000, 50_000, 500_000]   # production と同じ段階予算

def scan(path):
    """1 局を走査し，避けられた敗着の明細を返す (A/B の割当は CSA の N+ から)."""
    text = open(path).read()
    m = re.search(r"^N\+(.*)$", text, re.M)
    black_is_a = (m.group(1).strip().upper() == "A") if m else True
    rec = S.parse_csa_str(text)[0]
    b = S.PyBoard(); b.set_sfen(rec.sfen or START)
    hits = []
    for i, mv in enumerate(rec.moves):
        legal = b.legal_moves()
        if len(legal) <= 16:
            played = S.move_to_usi(mv)
            unresolved, losing, safe = list(legal), [], []
            for budget in STAGES:
                nxt = []
                for m2 in unresolved:
                    b.push(m2); child = b.sfen(); b.pop()
                    # find_shortest=False 必須 — 既定 (True) は最小性を確定
                    # できないと Unknown を返し，長手数の詰みを見逃す
                    r = S.solve_tsume(child, nodes=budget,
                                      find_shortest=False, timeout_secs=60)
                    (losing if r.status == "checkmate" else
                     safe if r.status == "no_checkmate" else nxt).append(m2)
                unresolved = nxt
                if not unresolved:
                    break
            if played in [S.move_to_usi(x) for x in losing] and safe:
                side = "A" if ((i + 1) % 2 == 1) == black_is_a else "B"
                hits.append({"ply": i + 1, "side": side, "played": played,
                             "safe": [S.move_to_usi(x) for x in safe][:3]})
        b.push(mv)
    return hits

tally = {"A": 0, "B": 0}
for path in sorted(glob.glob("/content/kifu_defmate/game_*.csa")):
    for h in scan(path):
        tally[h["side"]] += 1
        print(f"  {path.split('/')[-1]} ply{h['ply']:3d} [{h['side']}] "
              f"{h['played']} → 安全手 {h['safe']}")
print(f"\n避けられた敗着: A={tally['A']} B={tally['B']}")
print("判定:", "機構は効いている" if tally["A"] < tally["B"] else "要調査")
```

判定: **A 側の件数が B 側より少なければ機構は効いている**．Elo が有意で
なくても，この差は直接観測できる (30 局の事前走査では 7 件あった)．

## 記録先

結果は §9 のとおり worklog + compass へ．A/B が有意でなかった場合も
**「発火したが Elo 差は検出限界未満」と「発火しなかった」を必ず区別して
書くこと** — 前者は追試の対象，後者は実装のバグである．
