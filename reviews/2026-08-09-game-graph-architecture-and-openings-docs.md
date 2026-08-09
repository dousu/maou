---
status: pending          # pending | approved | applied | rejected
applied_in:
date: 2026-08-09
target: [docs/architecture.md, docs/commands/visualize.md]
risk: low
reversibility: trivial
---

# `game_graph` の建築上の居場所と，定跡データベースの内容を文書化する

## Trigger

`audits/coverage.md` § Out-of-scope backlog の 2 行
([2026-08-08 game_graph](../audits/2026-08-08-src-maou-domain-game-graph.md)
Out of scope 4 / 5)．`/audit-backlog` の T6 として選択され，HEAD
(`2e54fd4`) に対して再検証した:

- `OpeningDatabase` / `find_opening` / `openings.py` を
  `docs/` `CLAUDE.md` `AGENTS.md` `README.md` に対して検索 → **0 ヒット**．
  9 エントリ (`_DEFAULT_OPENINGS`) のどれがサポートされているのかを
  知る手段がドキュメント側に存在しない．
- `game_graph` を `docs/architecture.md` `CLAUDE.md` `AGENTS.md`
  `README.md` および `docs/adr-*.md` に対して検索 → **0 ヒット**．
  `docs/` 全体では `docs/commands/build_game_graph.md`,
  `docs/commands/visualize.md`, `docs/design/game-analysis/gui.md` の
  3 ファイルにのみ登場する．いずれもコマンドの使い方であって，
  `domain/game_graph` + `app/game_graph` という 2 レイヤにまたがる
  サブシステムがアーキテクチャ上どこに属するかは書かれていない．

前回 run (`3600b32`) で**利用者向けの挙動**(定跡行と平手限定という
制約) は `docs/commands/visualize.md:183-186` に着地している．
本提案が埋めるのはその残り，すなわち (a) サポート定跡の一覧と
(b) アーキテクチャ上の居場所である．

## Proposed change

### 1. `docs/architecture.md` — `## domain` の直後に新節を挿入

**Before** (`docs/architecture.md:40-44`):

```markdown
## domain

- ここではentityだけが存在する

## Shogi Engine (Rust) Encapsulation
```

**After**:

```markdown
## domain

- ここではentityだけが存在する

## Game Graph サブシステム

棋譜群を局面 (Zobrist hash) をノード，指し手をエッジとする DAG に
畳み込み，分岐と勝率を可視化するサブシステム．2 レイヤにまたがる:

| 層 | パス | 責務 |
|---|---|---|
| domain | `src/maou/domain/game_graph/` | Polars スキーマ (`schema.py`)，グラフの entity (`model.py`)，定跡データベース (`openings.py`) |
| app | `src/maou/app/game_graph/` | グラフ構築 (`builder.py`) とクエリ (`query.py`) のユースケース |

依存方向は他と同じく app → domain のみ．`openings.py` の
`OpeningDatabase.find_opening()` は**平手初期局面からの手順**を前提と
する純粋関数で，起点が平手かどうかの検査は行わない (手順だけからは
起点が分からない)．この前提の担保は interface 層
(`game_graph_visualization.py` の `_root_is_startpos()`) の責務である．

CLI は [build-game-graph](commands/build_game_graph.md) と
[visualize](commands/visualize.md)．

## Shogi Engine (Rust) Encapsulation
```

### 2. `docs/commands/visualize.md` — 定跡の説明にサポート一覧を追加

**Before** (`docs/commands/visualize.md:182-186`):

```markdown
- **局面統計**: Zobrist Hash，勝率(手番視点)，最善手勝率，深さ，分岐数，
  定跡(一致する定跡がある場合のみ「定跡名(カテゴリ)」形式で表示)．
  定跡データベースは平手初期局面からの指し手列で定義されているため，
  `build-game-graph --initial-sfen` で平手以外から構築したグラフでは
  定跡行は表示されない
```

**After**:

```markdown
- **局面統計**: Zobrist Hash，勝率(手番視点)，最善手勝率，深さ，分岐数，
  定跡(一致する定跡がある場合のみ「定跡名(カテゴリ)」形式で表示)．
  定跡データベースは平手初期局面からの指し手列で定義されているため，
  `build-game-graph --initial-sfen` で平手以外から構築したグラフでは
  定跡行は表示されない

#### サポートしている定跡

`src/maou/domain/game_graph/openings.py` の `_DEFAULT_OPENINGS` が
唯一の定義元である (下表はその写しなので，増減時は同時に更新する)．
ルートからの指し手列を**最長一致**で照合し，最も具体的なパターンを
採用する．

| 定跡名 | カテゴリ | 一致する手順 (USI) |
|---|---|---|
| 横歩取り模様 | 相居飛車 | `7g7f 3c3d 2g2f 8c8d 2f2e 8d8e` |
| ゴキゲン中飛車 | 振り飛車 | `7g7f 3c3d 2g2f 5c5d 2f2e 5d5e` |
| 相掛かり | 相居飛車 | `2g2f 8c8d 2f2e 8d8e` |
| 相振り飛車模様 | 振り飛車 | `7g7f 3c3d 6g6f 3d3e` |
| 矢倉 | 相居飛車 | `7g7f 8c8d 7i6h` / `7g7f 8c8d 6i7h` |
| 角換わり | 相居飛車 | `7g7f 3c3d 8h2b+` |
| 振り飛車模様 | 振り飛車 | `7g7f 3c3d 6g6f` |
| 先手中飛車 | 振り飛車 | `5g5f` |

一致しなかった局面には定跡行が出ない．表に無い戦型は
「定跡ではない」ではなく「このデータベースが知らない」だけである．
```

## Motivation

2 つの backlog 行はどちらも「文書が間違っている」ではなく
「文書が**存在しない**」という種類の finding で，2 回の
`/audit-backlog` run で T6 に留置され続けた．留置の理由は毎回同じ
「新規執筆が必要」であり，再検証を繰り返しても状況は変わらない．

実害は 2 つある:

1. **定跡一覧の非公開**．UI に定跡行が出ないとき，利用者には
   「この戦型は定跡データベースに無い」のか「バグで出ていない」のか
   区別できない．前回 run で平手限定という制約は文書化されたが，
   何が入っているかは依然としてソースを読むしかない．
2. **アーキテクチャ上の孤児**．`docs/architecture.md` は infra /
   interface / app / domain と Rust / Data I/O を説明するが，
   `game_graph` は 2 レイヤに実体を持つのに一度も現れない．
   `docs/architecture.md` を読んで全体像を得ようとする読者
   (人間もエージェントも) は，このサブシステムの存在自体を知らない
   まま作業を始める．今回の run で `find_opening` の前提条件を
   ドメイン側で検査しないと決めた根拠 (interface 側にガードがある)
   も，どこにも書かれていなかった．

## Alternatives considered

1. **`docs/design/game-graph/` として独立した設計文書を新設する．**
   将来 `docs/design/tsume-solver/` のような規模になれば適切だが，
   現状の実体は 5 ファイル程度で，設計上の未解決論点も抱えていない．
   索引だけが増えて読まれない文書ができる可能性が高い．
   `architecture.md` に 1 節を置くほうが，全体像を探す読者の動線に
   合う．規模が育った時点で切り出せばよい (その時は本節がリンクの
   置き場になる)．
2. **定跡一覧は書かず，`openings.py` を読めと案内する．**
   ソースが唯一の真であるという点では正しく，二重管理も避けられる．
   ただし利用者向けドキュメントの読者にソースを読ませるのは
   `docs/commands/` の役割と噛み合わない．また
   `docs/architecture.md:142-144` が `array_type` について
   「正準の定義は Literal で，増えた場合はそちらが真」と写しを置く
   前例を作っているので，同じ形 (写し + 唯一の定義元の明示) に
   揃えた．
3. **`CLAUDE.md` に `game_graph` の記述を足す．**
   `CLAUDE.md` はエージェント向けの MUST/SHOULD 規則が本体で，
   サブシステムの構成説明を置く場所ではない．Documentation Links
   表に行を足す案も検討したが，`architecture.md` 経由で辿れるので
   索引の重複になる．

## What this enables

- `docs/architecture.md` だけを読んで，リポジトリのサブシステムを
  取りこぼさずに把握できる (`game_graph` が最後の空白だった)．
- 「定跡行が出ない」問い合わせに対し，ドキュメントの表を見せるだけで
  データベースの範囲かどうかを切り分けられる．
- `find_opening` の前提条件と，その担保が interface 層にあるという
  設計判断が，ソースの docstring 以外の場所にも残る．

## What this constrains

- `_DEFAULT_OPENINGS` を増減したら `docs/commands/visualize.md` の表も
  更新しなければならない (写しなので drift する)．表のすぐ上に
  「唯一の定義元」を明記して drift 時にどちらが正かを曖昧にしない
  形にしたが，同期義務そのものは増える．
- `game_graph` のファイル構成を変えたら `docs/architecture.md` の表の
  パスも直す必要が出る．

## Rollback plan

`docs/architecture.md` と `docs/commands/visualize.md` の 2 ファイルを
revert するだけ．コードは一切参照していないので何も壊れない．
本提案の `status:` を `rejected` にして do-not-redo の記録として残す．
