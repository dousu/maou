---
kind: backlog
date: 2026-08-15
path:
  - src/maou/domain/data/schema.py
  - src/maou/domain/data/columnar_batch.py
  - tests/maou/domain/data/test_schema.py
  - tests/maou/domain/data/test_columnar_batch.py
scope: python
level: medium
last_sha: 846c0e9
record_sha: 0b3f1ab
---

# backlog consumption — writeable 契約の確立と，残る 4 行のうち 3 行の設計判断

`/audit-backlog` (2026-08-15, `as8mni`)．前 run
([2026-08-15 columnar-uninitialized-fields](2026-08-15-backlog-columnar-uninitialized-fields.md))
が「次 run の先頭候補」と名指しした D13(1) を出荷し，**残る 4 行のうち 3 行の
設計判断をユーザから得た** run である．

**backlog は 5 行で始まり 5 行で終わった**が，動いたのは行数ではない —
**ゲートの数が 4 から 2 に減り，そのうち 1 つは意味が縮小した**．

## Classification

分類は**ファイルに触れる前に**会話へ出した (step 3a)．

| ID | 由来 | Target | クラス | クラスを決めた test | ゲート |
|---|---|---|---|---|---|
| **B-1** | [2026-08-10 file_system](2026-08-10-src-maou-infra-file-system.md) D13 (1) | `domain/data/schema.py:782` | **P4** | 返る**値**は同一だが，read-only ビューが copy に変わり**ピークメモリが増える**．P3 の「memory may differ」に収まるとも読めるが，`moveWinRate` は N×1496 float32 (N=100k で約 598MB) で無視できる量ではないので fail-safe の向き (上) に倒した | **なし** — G2 は残る作業 (2)(3)(4) にのみ掛かり，(1) は `domain/data/` に閉じる．G4 は前 run で retire 済み |
| **D2** | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `app/learning` | P4 | 挙動不変を意図した ~585 行の refactor | **G3** |
| **D5** | 同上 Deferred 5 | `training_loop.py:1117` | P4 | 休眠中の性能改善 | **G1** |
| **D7** | 同上 Deferred 7 | `gradient_noise_scale.py` | P4 | 既定を渡していない既存の実行は挙動が変わる | **G1** |
| **O9** | [2026-08-10 file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `infra/bigquery` | P4 | 同じ呼び出しで返る行が変わる | **G1** |

**自動帯は空 (14 run 連続)** — 5 行すべてが P4 以上．

### 再検証 (step 2)

**stale 0 / changed shape 1 / confirmed 4**．

- **D2 / D5 / D7 / O9 は confirmed，行番号のずれも 0**．前 run が `src/` に
  入れた変更 (`033d49f`) が `infra/file_system` にしか触れていないため．
  `diff` で `stage_component_factory.py:704-731 ≡ :794-821` の完全一致も再確認．
- **D13 は "changed shape"** — 所見自体は confirmed だが，`file_data_source.py`
  の行番号が前 run の `033d49f` により **+15〜+20** ずれていた
  (`_columnar_batch_to_structured_array` `:422`→`:437` など)．
  `schema.py:782` は記録どおり．

### read-only の実測 (B-1 の前提の確認)

記録の「実測済み」を鵜呑みにせず，**修正前の状態で**独立に再現した
(`uv run --no-project --with polars --with numpy`, polars 1.43.2):

```
uint8 match   -> uint8   writeable=False  contig=True
astype path   -> float16 writeable=True
nested uint8  -> uint8   writeable=False
float32 match -> float32 writeable=False
```

polars 側 dtype が目標 dtype と一致する列だけが read-only になる，という
記録の診断は正確だった．

## Consumed

| 行 | 由来 | 出荷したもの | PR |
|---|---|---|---|
| **D13 の残る作業 (1) のみ** | [2026-08-10 file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `_explode_list_column` の writeable 契約 | [#505](https://github.com/dousu/maou/pull/505) |

**行は削除していない** — (1) だけの消化で，(2)(3)(4) は G2 が残る．
行は (1) を消化済みと明記して縮めてある．

## Applied

- `src/maou/domain/data/schema.py:820` (`0b3f1ab`) — fast path に
  `elif not result.flags.writeable: result = np.array(result, copy=True)`．
  **条件付き**なので `astype` が走った場合 (既にコピー) と polars が所有
  バッファを返した場合には追加コストが無い．
- `src/maou/domain/data/schema.py` (`0b3f1ab`) — `_explode_list_column` の
  docstring に「C-contiguous かつ writeable」を契約として明記．
- `src/maou/domain/data/columnar_batch.py` (`0b3f1ab`) — `ColumnarBatch` の
  docstring の「C-contiguous」に「writeable」を足し，保証の出所を明示
  (backlog 行が指定した形)．
- `tests/maou/domain/data/test_schema.py` — `TestExplodeListColumnWriteability`
  (5 本)．
- `tests/maou/domain/data/test_columnar_batch.py` —
  `TestColumnarBatchWriteabilityContract` (5 本)．

**非空虚性が最大の論点だった**ので，テスト側にもそれを書いた: polars 側
dtype を目標 dtype と明示的に一致させ，`astype` が挟まらない経路である
ことを事前 assert している．**無効化テストで 10 本中 7 本が落ちる**ことを
確認済み (残る 3 本は元から writeable な経路 — `astype` 経路 / null
fallback 経路 / スライス後 — を固定する characterization test)．

## Decisions asked

`AskUserQuestion` 1 回，**受理 1 問 + 設計判断 3 問**．4 件とも回答を得た．

### Q1 (受理) — PR #505 を受理するか

- 提示: 「マージする (推奨)」/「マージしない (現状維持)」
- **ユーザの選択: マージする**
- 提示した内容: 値は不変・データ互換性の破壊なし，変わるのはピークメモリで
  `moveWinRate` (N×1496 float32) が実質的なコスト (N=100k で約 598MB の一時増)．
  却下した代替案として「`writeable` フラグを立てるだけ」(Arrow バッファは
  polars 側と共有されているのでテンソルが書き戻すと元データを壊す) を明示．

### Q2 (設計判断) — D2 の G3 をどう扱うか

- 提示: 「G3 を retire する (推奨)」/「4 組を 1 PR ずつに分割して出荷」/
  「G3 を残す」
- **ユーザの選択: G3 を retire する**
- **settles**: Deferred 2 の 1 行．
- **本 run で実装したか: していない** — 決定を行に書いて G3 を retire した．
  **これで Deferred 2 に残るゲートは無くなり，次 run の先頭候補になった**．
- 効いた材料: 2026-08-14 に Deferred 5/6/7 へ与えた「GPU で測れなくても
  意味論的に等価な変更は出荷してよい」という既存の判断と，統合が挙動不変を
  意図した純粋な refactor であること．

### Q3 (設計判断) — D7 の安価な緩和を今取るか

- 提示: 「既定を上げる (推奨)」/「既定は `1` のまま」/
  「既定を廃して caller に必須指定させる」
- **ユーザの選択: 既定を上げる**
- **settles**: Deferred 7 の緩和策の向き (本丸の device スカラー化は別問題)．
- **本 run で実装したか: していない．理由は下記の "Corrections" を参照** —
  質問時に「1 行で contained」と述べたのが**誤り**で，実際には CLI 既定
  (`learn_model.py:217`) を変えないと本番経路に効かず，`docs/commands/` の
  更新も伴う．受理済みの PR #505 にその scope を後から足すのは受理の範囲を
  超えると判断した．

### Q4 (設計判断) — O9 のテスト土台をどうするか

- 提示: 「fake client の土台を同梱 (推奨)」/「先に既存の 2 本を作り直す」/
  「実 BigQuery 環境まで着手しない」
- **ユーザの選択: fake client の土台を同梱**
- **settles**: O9 の 1 行．
- **本 run で実装したか: していない** — 決定を行に書いた．
- **この回答は G1 の意味を縮小した**: 決定的ハッシュ条件の要点は「同じ
  `page_num` を 2 回引くと同じ行が返る」ことで，これは fake で CI 上に載せ
  られる．「実 BigQuery が無いと着手できない」は成り立たなくなり，G1 は
  「出荷前の実地確認」に縮小した形でのみ残る．

### 予算に入らなかった設計判断 (次 run の待ち行列)

1. **Deferred 5** — G1．方針は 2026-08-14 に決定済み (「mask の産出者を配線
   する変更と同時に，GPU 上で測って直す」)．**設計判断としては未着手ではなく，
   今回の 4 問の枠に入らなかっただけ**．本行は「産出者ゼロで休眠中」という
   前提の上に立つ性能改善なので，問うとすれば「休眠のまま backlog に置き
   続けるか，mask 配線の予定が無いなら行ごと落とすか」になる．

## In flight

**なし** — Q1 が同一セッション内で回答されたため，判断帯の PR は
開いたまま残っていない．

## Re-triaged

**なし** — 本 run が触れた 4 行はいずれも「決定を得て前進した」側であり，
文言だけ鋭くして残した行は無い．

## 決定は得たが実装しなかった項目

| 行 | 決定 | 実装しなかった理由 |
|---|---|---|
| **Deferred 2** | G3 を retire | ~585 行の統合は本 run の枠に入らない．**ゲートは無くなったので次 run の先頭候補** |
| **Deferred 7** | `measurement_interval` の既定を上げる | 質問時の scope 説明が誤っていた (下記 Corrections)．受理済み PR に CLI 既定 + doc の変更を足すのは受理の範囲を超える |
| **O9** | fake client の土台を同梱 | (0) テスト土台の新設から始まる 4 段階の作業で，本 run の枠に入らない |

**これは run の失敗ではない** — 3 行とも「人間が決めないと誰も書けない」
状態から「決まっていて，あとは書くだけ」に変わった．

## Corrections to the source records

**元記録への訂正はなし**．今回の再検証で誤りが見つかったのは記録ではなく
**本 run 自身が質問時にユーザへ述べた scope の説明**である．

### 本 run の Q3 の説明が誤っていた (`measurement_interval` の既定)

Q3 で「1 行で contained，この環境で出荷できる」と述べたが，実装直前に
`grep` したところ **`measurement_interval` の既定は 3 箇所**にあり，本番
経路を決めているのは**クラス既定ではなく CLI 既定**だった:

| # | 場所 | 値 |
|---|---|---|
| (i) | `gradient_noise_scale.py:85` | `measurement_interval: int = 1` |
| (ii) | `adaptive_batch.py:68` | `measurement_interval: int = 1` |
| (iii) | **`infra/console/learn_model.py:217`** | `--adaptive-batch-measurement-interval` の `default=1` |

CLI は `learn_model.py:984` で `measurement_interval=adaptive_batch_measurement_interval`
と**常に明示的に渡す**ので，**(i)(ii) だけ変えても本番経路の挙動は
1 ミリも変わらない**．決定を実現するには (iii) を変える必要があり，CLI
オプションの既定変更は CLAUDE.md の MUST により `docs/commands/learn_model.md`
の更新を伴う．

この訂正は行に書いてあるので，次 run は正しい scope から始められる．
**なおリポジトリ自身が既に推奨値を持っている** — docstring `:76-78` と CLI
help `:222` がともに「5〜10 (大規模モデル)」と書き，`training_benchmark.py:2310`
の `_recommend_measurement_interval` は 1 / 5 / 10 を返す．次 run の推奨は
**5** (推奨帯の下限)．

## Doc findings

**なし**．`_explode_list_column` の writeable 契約を記述している durable doc は
存在せず (`docs/` を `ColumnarBatch` / ゼロコピー / `_explode_list_column` で
横断検索して確認)，本 run の修正が無効化する記述は無かった．
`docs/rust-backend.md:729` は `iter_batches()` が `ColumnarBatch` を
structured array に変換して返す話で，本修正の影響を受けない．
したがって `reviews/*.md` の提案は起票していない．

**次 run が D7 に着手するときは doc 更新が発生する** — CLI オプションの
既定変更なので `docs/commands/learn_model.md` の更新が CLAUDE.md の MUST に
掛かる (訂正後の本文はコードから一意に決まるので P2 の drift correction)．

## Out of scope

**新規の out-of-scope 所見はなし**．`coverage.md` の backlog 表に追加した行は
無い．ただし**台帳自身のバグを 1 件見つけて同 run 内で修復した** (下記 B-2)．

### B-2 — Deferred 5 行の本文が描画時に丸ごと消えていた (台帳の retrieval bug)

台帳の更新後に全テーブル行のセル数を検査したところ，**Deferred 5 の行だけが
3 セルではなく 5 セル**になっていた．原因は行本文に含まれる
`legal_move_mask: torch.Tensor | None = None` の **`|` が 2 個ともエスケープ
されていない**ことで，GFM はこれをセル区切りとして解釈し，余剰セルを捨てる．

結果として，**最初の `| None` より後ろ — 2026-08-13 の dormant の証明
(`TrainingContext` の構築箇所が 1 つだけで `:514` にハードコードされている
こと，`src/` に産出者がゼロであること)，2026-08-14 のユーザ回答，2026-08-15 の
行番号更新と `RawLogitsTrainingLoop` によるクラス到達不能性 — がすべて
描画時に不可視**だった．本 run が Deferred 5 を「方針は決定済み」と判断できた
のは raw ファイルを読んだからであり，**レンダリングされた台帳しか見ない読者に
とっては，この行は根拠を欠いた 1 行に見えていた**．

これは 2026-08-14 の run が見つけた **B1 と同じクラスのバグ**である
(B1 は Deferred 3 行と O9 行が 4 セル目を持ち，設計判断の記述が消えていた)．
B1 の修復は当該 2 行のみを対象としており，**同種の欠陥が他の行に残っていないかを
検査する仕組みは入らなかった**ため，Deferred 5 が生き残った．

- **クラス: P1** — `audits/` のみに触れるので shipped file はゼロ，version bump 不要．
- **修復**: `|` を `\|` にエスケープ (2 箇所)．
- **backlog 行は起票していない** — B1 と同じく同 run 内で修復済みのため．
- **再発防止のために本 run が使った検査** (次 run 以降も台帳更新後に流すとよい):
  各テーブル行を**エスケープされていない `|`** で分割し，同一テーブル内で
  セル数が揃っているかを確認する．`grep` で `|` を数えるだけでは
  `\|` と `|` を区別できず見逃す．

## Environment notes

- **G3 は発生していない**．`uv sync --extra cpu` で torch を導入し，
  `domain/data` だけでなく消費側の `app/learning` と `infra/file_system` の
  suite も回した (**862 passed, 1 skipped**)．
- `uv run` の初回起動が Rust 拡張の maturin ビルドを引き起こすため数分かかる．
  polars の read-only 挙動だけを修正前に確認したいときは
  `uv run --no-project --with polars --with numpy` で maou パッケージの
  ビルドを回避できる (本 run で実際に使った)．
- **GPU は無い** — D5 / D7 の本丸 (device スカラー化) の数値等価性は
  この環境では測れない (G1)．
- **BigQuery は無い** — ただし Q4 の回答により，O9 の要点は fake で検証
  可能になった (G1 の縮小)．

## QA

| チェック | 結果 |
|---|---|
| `uv run ruff format src/ tests/maou/domain/data/` | 2 files reformatted |
| `uv run ruff check src/ tests/maou/domain/data/ --fix` | All checks passed! |
| `uv run mypy src/` | Success: no issues found in 135 source files |
| `uv run pytest tests/maou/domain/data/` | 206 passed |
| `uv run pytest tests/maou/app/learning/ tests/maou/infra/file_system/ tests/maou/domain/data/` | **862 passed, 1 skipped** |
| 無効化テスト | 修正を外すと 10 本中 **7 本が落ちる**ことを確認 |

## Version

`pyproject.toml` `0.92.1` → **`0.92.2`** (`fix:` → patch)．`uv.lock` 再生成済み．

## Stack

**指定ブランチ `claude/audit-backlog-as8mni` 1 本**の制約によりクラス毎の
PR 分割ができないため，**単一 PR** ([#505](https://github.com/dousu/maou/pull/505),
`base: main`)．レビュー単位は commit が担う．
台帳の更新 (D13 行の縮小 + D2/D7/O9 行への決定の書き込み + 本記録) は
**6a の separability test により同じ PR に同梱**した — 別 PR にすると
`main` へのマージが 2 回になり wheel を 2 回焼くため．
**`main` へのマージは 1 回**．
