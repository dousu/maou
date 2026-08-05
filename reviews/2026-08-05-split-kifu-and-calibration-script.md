---
title: 棋譜の対局単位分割コマンドと較正測定スクリプトを追加し，utility グループを軽量化する
date: 2026-08-05
status: applied
applied_in: 71285fd
target:
  - docs/commands/utility_split_kifu.md
  - docs/commands/utility_fetch_floodgate.md
  - docs/commands/floodgate.md
  - docs/commands/visualize.md
  - docs/commands/utility_generate_stage1_data.md
  - docs/commands/utility_generate_stage2_data.md
  - docs/dependency_management.md
  - docs/commands/analyze_game.md
  - docs/commands/floodgate.md
  - docs/design/training-quality/index.md
risk: low
reversibility: easy
---

# 提案: `maou utility split-kifu` と `scripts/measure_calibration.py` を追加する

## 背景

`--stage3-validation-data-path` (PR #440) を使うには，棋譜を**対局単位で
2 分割してからそれぞれ前処理する**必要がある．前処理は局面を Zobrist hash で
全コーパス横断に集約するため，集約後に対局の同一性は復元できないためである．

しかしその分割手段が無く，利用者が自前でスクリプトを書く必要があった．

また較正測定 (ECE / Brier / 対局単位 bootstrap) はこのプロジェクトの
North-star 指標であり，再学習のたびに実行する．セッション限りのスクリプトでは
再現できないためリポジトリに置く必要がある．

## 変更内容

### 1. `maou utility split-kifu` (新規コマンド)

棋譜ファイルを対局単位で train / val に分割する．

- 分割の単位はファイル．floodgate は 1 ファイル 1 対局なのでこれが対局単位になる
- **決定的**: 同じ入力と同じ `--seed` からは常に同じ分割 (シャッフル前に
  パスでソートするので入力の列挙順にも依存しない)
- `--mode {copy,symlink,hardlink}`: 60 万局規模ではコピーがディスクを圧迫する
  ため，追加容量を使わない選択肢を用意する
- 入力からの相対パスを出力先でも保つ (floodgate の `YYYY/MM/DD` 構造が残る)
- `--dry-run` で件数だけ確認できる

実データ (floodgate 2026-03 の 878 局) で動作確認済み:
train 790 / val 88，両側のファイル名の重複ゼロ，ディレクトリ構造保持．

配置:

- `src/maou/app/utility/kifu_split.py` — `plan_split` (純粋関数) /
  `transfer` / `apply_split`
- `src/maou/interface/kifu_split_interface.py` — 検証と JSON 出力
- `src/maou/infra/console/utility.py` — CLI

### 2. `scripts/measure_calibration.py` (新規)

held-out 棋譜で value head の較正を測る．

- 実測勝率は HCPE の `gameResult` のみから算出．棋譜中のエンジン評価値
  (`eval` 列) はクライアント依存のため使わない
- 同一対局の局面は同じ勝敗ラベルを共有するため，信頼区間は**対局単位の
  bootstrap** で求める
- ECE / Brier に加え，**診断値としての温度 `T`** を出す．`T` が 1 から
  離れているほど過信であり，`T` が 1 に近づくことが過学習抑制の成功指標になる
  (後処理として適用する方針は採らない — 設計文書 §3 Step 1)

### 3. `utility` グループの軽量化と `fetch-floodgate` の移動

`maou fetch-floodgate` を `maou utility fetch-floodgate` に移した
(user 判断: maou の根幹コマンドではなく wget 等でも代替できるため)．

ただし移動前の `utility` グループは **torch を必須**としていた
(`app.py` の `required_packages`，および `utility.py` が module 冒頭で
`maou.interface.learn` を import)．そのまま移すと**棋譜取得に学習依存が
必要**になるため，先にグループを軽くした．

- `src/maou/infra/console/lazy.py` (新規) — `LazyGroup` / `LazyCommandSpec` /
  `PackageRequirement` を `app.py` から分離．トップレベルと `utility` の
  双方から使う
- `src/maou/infra/console/utility_group.py` (新規) — 軽い `utility` グループ．
  torch を要する 4 サブコマンド (benchmark 系 / stage データ生成) は
  `LazyGroup` で遅延解決し，`fetch-floodgate` / `split-kifu` / `screenshot` は
  直接登録する
- `src/maou/infra/console/split_kifu.py` (新規) — `split-kifu` を
  `utility.py` から分離 (`utility.py` は重いため)
- `app.py` — top-level の `fetch-floodgate` を削除，`utility` の
  `required_packages` を撤去

検証: `utility_group` を import しても `torch` / `maou.interface.learn` /
`maou.app.learning.dl` がいずれも `sys.modules` に載らないことを確認した．
これは CLAUDE.md の「MUST NOT import heavy optional dependencies ... in
`src/maou/infra/console/` or any path a light command reaches」にも適合する．

### 4. ドキュメント

- `docs/commands/utility_split_kifu.md` (新規) — CLI 仕様と，
  「なぜ前段で分けるのか」の説明，および fetch → split → convert →
  pre-process → learn-model の実行例
- `docs/commands/fetch_floodgate.md` → **`utility_fetch_floodgate.md` に改名**．
  utility サブコマンドのドキュメントは `utility_*` 接頭辞で揃っているため
  (`utility_benchmark_*` / `utility_screenshot` / `utility_split_kifu`)．
  コマンド名を `maou utility fetch-floodgate` に更新し，**§次の手順**として
  `split-kifu` への導線を追加 (取得と分割を分けた理由も記載)
- `docs/commands/analyze_game.md` / `floodgate.md` / `docs/dependency_management.md`
  — コマンド名とリンク先の参照を更新
- **命名規約の統一**: `generate-stage1-data.md` / `generate-stage2-data.md` も
  `utility_generate_stage1_data.md` / `utility_generate_stage2_data.md` に改名．
  これで utility サブコマンドのドキュメントがすべて `utility_` 接頭辞
  (アンダースコア区切り) で揃った
- `docs/design/training-quality/index.md` §4 — 較正測定の手順を
  `scripts/measure_calibration.py` を使う形に更新
- `scripts/check-cli-docs.sh` — `UTILITY_DOCS` に新ドキュメントを追加し，
  `utility_group.py` も `utility.py` と同じ扱いにする．
  `fetch_floodgate.py` / `split_kifu.py` のマッピングも追加
  (`fetch_floodgate.py` は従来マッピング漏れだった)

## リスク

- **低**: 新規コマンドと新規スクリプトのみ．既存の挙動は変わらない
- `--mode hardlink` は同一ファイルシステムでのみ動作する (異なる場合は
  `os.link` が例外を投げるので黙って失敗することはない)
- 出力先に同名ファイルがある場合は上書きする
- **`maou fetch-floodgate` → `maou utility fetch-floodgate` は破壊的変更**．
  既存スクリプトの書き換えが必要
- 版数は `maou 0.77.0` → `0.78.0` (コマンド追加 + CLI 表面の破壊的変更．
  0.x では破壊的変更を minor で扱う既存慣行に従う)
