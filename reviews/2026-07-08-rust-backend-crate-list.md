---
status: applied
applied_in: 40651cd
date: 2026-07-08
target: [docs/rust-backend.md, CLAUDE.md]
risk: low
reversibility: trivial
---

# rust-backend.md / CLAUDE.md の crate 一覧に maou_search を追加

## Trigger
worklog/2026-07-08-065408.md — 新 crate `rust/maou_search` (v0.1.0, pure Rust
MCTS 探索エンジン) を 39a4d89 で追加したため，workspace 構成を記述する
durable docs が stale になった．

## Proposed change

**docs/rust-backend.md** § "Rust Project Structure" のディレクトリツリーに
maou_search を追加する:

```
├── maou_index/             # インデックスクレート
├── maou_search/            # MCTS 1局面探索エンジンクレート (pure Rust)
│   └── src/
│       ├── lib.rs
│       ├── evaluator.rs    # Evaluator trait (NN 推論の抽象境界) + MockEvaluator
│       ├── tree.rs         # 固定容量ノードプール + lock-free 統計
│       └── search.rs       # PUCT 探索本体 (バッチ収集 + virtual loss)
└── maou_shogi/             # 将棋エンジンクレート
```

**CLAUDE.md** § "Versioning (Rust crates)" の crate 列挙に 1 行追加:

```
  - `rust/maou_search/Cargo.toml` for `maou_search` crate
```

## Motivation
worklog/2026-07-08-065408.md「Architectural decisions」— maou_search は
workspace members に既に入っており (Cargo.toml @ 39a4d89)，docs だけが
4 crate 構成のまま．CLAUDE.md のバージョニング規則は crate を名指しで列挙して
いるため，maou_search が対象外に見える穴がある (規則の意図は全 crate 適用)．

## Alternatives considered
1. **docs 更新を PyO3 露出時まで遅延** — maou_search はまだ Python に露出して
   いないので `maou._rust.*` 系の記述は変わらない．しかし versioning 規則の穴と
   ツリー図の stale は露出前から実害があるため棄却．
2. **docs/design/position-search/ の設計 doc と一括提案** — 設計 doc は
   GC・dfpn 統合・最終手基準が未確定で churn が大きく，事実ベースの
   crate 一覧更新だけ先に通す方が audit trail が細かく残るため分離．

## What this enables
- 新 crate のバージョニング規則が明文で binding になる (5.4.0 系 maou_shogi と
  独立に 0.x で回せる)
- 次セッション以降の読者が workspace 構成を docs から正しく把握できる

## What this constrains
- maou_search too は「変更 = Cargo.toml bump 必須」の MUST 対象になる (意図通り)

## Rollback plan
docs/rust-backend.md と CLAUDE.md の該当行を revert するだけ (コード非依存，
crate 本体は 39a4d89 のまま残る)．
