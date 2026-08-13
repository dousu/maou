---
kind: backlog
date: 2026-08-13
path:
  - src/maou/domain/data/arrow_format.py
  - src/maou/domain/data/rust_io.py
level: medium
last_sha: c9bc24a
---

# `/audit-backlog` — 中途書き込み `.feather` の落ち方 (D15)

同日 3 本目の `/audit-backlog`．`coverage.md` の backlog 2 表から
**17 行** (deferred 8 + out-of-scope 9) を拾い，HEAD `c9bc24a` に対して
再検証した．

- **stale: 0**
- **changed shape: 1** — D15 (下記)
- **confirmed: 16**

うち **1 件 (D15)** を消化した．**この run の消化対象が 1 件しかないのは，
残り 16 行が全てゲート付きだからで，取りこぼしではない** (§ Classification)．

**5 行は既に in flight** ([PR #493](https://github.com/dousu/maou/pull/493))．
この PR の head はこのセッションの指定ブランチそのものなので，1e の規定
どおり**新しい PR は開かず #493 に積んだ**．

**この run はユーザに何も聞いていない．** 5c の分割テストに該当する項目
が無かった — D15 の修正の形は 1 通りに定まる．

## Classification

### 消化した 1 件

| ID | backlog 行 | 対象 | クラス | そのクラスに決めたテスト | ゲート |
|---|---|---|---|---|---|
| P3-1 | D15 | `domain/data` (行の対象は `infra/file_system`) | **P3** | `BaseException.add_note` は例外の型も同一性も制御フローも変えない．返り値・成果物は不変で，増えるのは診断文だけ | **G4 を撤回** / **G2 を引き込みで解決** |

**D15 は "changed shape" だった．** 行は「`_is_temp_artifact` は末尾拡張子
リストで判定するので**原理的に捕捉できず**，size/footer 検査が要る」と
書いていたが，実機で作った中途書き `.feather` を読ませたところ:

| 入力 | 実際に起きること |
|---|---|
| File 形式のまま footer が欠ける (中断した書き込み) | `ComputeError: out-of-spec: InvalidFooter` |
| 0 バイト | `OSError: failed to fill whole buffer` |

**素通しした先で黙って壊れたデータを読むのではなく，読み込み時点で確実に
落ちている．** つまり記録が想定した「検査を足す」は不要で，実際の欠陥は
**どちらのメッセージもファイル名を含まない**ことだった．数百ファイルを
渡す運用では犯人を特定できない．

これは記録の**診断が誤っていた**ケースなので，6b の narrow exception に
従って元記録に訂正を追記した (§ Corrections)．

**G4 撤回の理由．** 行の未決点は 2 つ — 「運用上のリスクとして実在するか
は要判断」と「size/footer 検査が要る」．前者は実機で確認でき (実在する)，
後者は「既に落ちているので不要」と分かった．残る行動は「エラーに
ファイル名と原因を載せる」だけで，これに第二の版は無い．**前 run で
この行を鋭くしたことが，そのまま今回の撤回条件になった** — Re-triaged
セクションが機能した例．

**G2 の扱い．** 読み込みの合流点は `domain/data`
(`arrow_format.scan_row_count` の 2 経路，`rust_io.load_*` の 5 経路) で，
行の対象 `infra/file_system` の外にある．4a の 2 択のうち**引き込む**方を
採った — infra 側でラッパを重複させる方が悪く，かつ #493 はどのみち
未マージなので追加コストが無い．

### 残した 16 件のクラスとゲート

| ID | クラス | ゲート | 残す理由 |
|---|---|---|---|
| D-L2 / D-L4 | P4 | **G3** | 250-400 行の学習経路リファクタ．この環境で等価性を示せない |
| D-L3 | P6 | **G3** | 6 adapter クラス．公開名が消えるうえテストが参照している |
| D-L5 / D-L6 / D-L7 | P4 | **G1** | GPU が無い．D-L5 は dormant のまま |
| FS-D5 (本体) | P6 | **G4** | ノブ廃止は O5 と一体 |
| FS-D10/D11 (1) | P4 | **G4** | `total_pages()` は caller ゼロ．意味の決めが要る |
| FS-D13 | P4 | **G2** | 根本解決は `app/learning/dataset.py` と ABC を触る |
| FS-D14 (b) | P6 | **G2** | 二重 ABC．外すと `benchmark_polars_io` の対応が要る |
| O5 (本体) | P6 | **G4** | bool flag と dir の整合の決めが要る |
| O9 | P4 | **G1** | BigQuery 実環境が無い |
| N4 | P1 | **G4** | 2 案が未決．**5 run 連続で実害** |
| N6-2 | P4 | **G4** | 改名 vs 分岐．**前 run の記述を訂正** (下記) |
| N6-1 / N7 | P4 / P1 | **in flight** | #493 に載っている．二重作業しない |

## Consumed

| ID | 由来 | 対象 | 出荷したもの | コミット |
|---|---|---|---|---|
| P3-1 | D15 | `domain/data/arrow_format.py`, `domain/data/rust_io.py` | `feather_read_errors_name_the_file` を新設し，行数スキャン 2 経路と rust_io の読み込み 5 経路が通るようにした | `672431a` |

すべて `672431a`:

- `src/maou/domain/data/arrow_format.py:39` (`feather_read_errors_name_the_file`),
  `:158` / `:184` (行数スキャンの 2 経路)
- `src/maou/domain/data/rust_io.py:119`/`:170`/`:219`/`:273`/`:327` (5 つの読み込み経路)
- `tests/maou/domain/data/test_feather_read_error_context.py` (回帰テスト 8 件)

バージョン: `0.89.7` → `0.89.8` (`pyproject.toml`)．

## Reconciliation (6d)

`17 (再検証した行) + 0 (新規 finding) = 0 resolved + 6 in flight +
11 re-triaged + 0 new rows + 0 not-a-finding`

- **resolved 0** — 6a は「マージされたら削除」なので，#493 が開いて
  いるうちは 1 行も消せない
- **in flight 6** — 前 run からの 5 行 (O5 / FS-D5 / FS-D10+D11 /
  N6-1 / N7) に，この run の **D15** が加わった
- **re-triaged 11** — D-L2 / D-L3 / D-L4 / D-L5 / D-L6 / D-L7 /
  FS-D13 / FS-D14(b) / O9 / N4 / N6-2．うち記述を鋭くしたのは N6-2 の 1 件
- **new row 0** / **not-a-finding 0**
- backlog 行数: **17 → 17** (削除も追加もなし)

## In flight

**PR #493 のみ** (base: `main`)．この run のコミットもここに積んだので，
PR が持つ判断は前 run の 2 点 (BigQuery 契約 / 未測定の並列化) に
**D15 は加わらない** — D15 は P3 でゲートも解決済みなので，判断を要さない．

## Re-triaged

上表「残した 16 件」．今回**記述を鋭くできた**のは 1 件だけ:

- **N6-2** — 前 run で私は「N6-1 の修正で基底の既定実装を通る
  production 経路はゼロになった」と書いたが，**これは不正確だった**．
  `StreamingHcpeDataSource` (`console/pre_process.py:494` が構築する
  production の pre-process ソース) が `preprocess.DataSource` を継承
  しつつ `iter_batches_df` を override していない．ゼロなのは*呼び出し*
  であって*継承*ではない．しかも唯一の継承者は hcpe 専用なので，
  **基底の HCPE 決め打ち既定実装はそこでは正しい**．
  この訂正は行の判断に効く: 「HCPE 専用と明記する」方向の根拠が
  強まり，「array_type で分岐させる」方向は継承者が居ない分だけ弱まる．
  ただし前者は行の文言上 **改名 (P6) を含む**ので，2 案はまだ 1 案には
  潰れていない．G4 継続．

## Corrections to the source records

`audits/2026-08-10-src-maou-infra-file-system.md` の D15 (§ Cross-module
sweep) — 「size/footer 検査が要る」という提案は誤り．検査を足さなくても
読み込み時点で確実に落ちる (`InvalidFooter` / `failed to fill whole
buffer`)．実際の欠陥は落ちないことではなく，**メッセージがファイル名を
含まないこと**だった．記録に訂正を追記した．

## Doc findings

なし．D15 の修正は挙動 (診断文) だけを変え，どの `docs/` の記述も
無効にしない．`docs/commands/` に中途書き `.feather` の扱いを説明した
箇所は存在しない．

## Out of scope

なし．

## Environment notes

- 前 run で構築した venv (`uv sync --extra cpu`) と Rust 拡張
  (`maturin develop --release`) がそのまま使えた．コンテナが生きている
  間は再構築が要らない．
- pre-commit は毎コミットで全テストを回す (約 6 分)．
- 撤回できなかった G1: GPU 無し (D-L5/D-L6/D-L7)，BigQuery 実環境無し
  (O9)．
- 撤回できなかった G3: `app/learning` の大規模リファクタ (D-L2/D-L3/D-L4)．
