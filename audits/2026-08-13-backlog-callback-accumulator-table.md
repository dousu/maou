---
kind: backlog
date: 2026-08-13
path:
  - src/maou/app/learning/callbacks.py
level: medium
last_sha: 686bcae
---

# `/audit-backlog` — コールバック蓄積テンソルの宣言表化 (Deferred 4)

`coverage.md` の backlog 2 表から **14 行** (deferred 8 + out-of-scope 6)
を拾い，HEAD `686bcae` に対して全件を再検証した．開いている PR は無く，
`reviews/` の `pending` 提案も無い (`README.md` の雛形を除く)．

- **stale: 0** — 14 件すべてが今も成立する
- **changed shape: 2** — Deferred 3 と O9．どちらも「記録が書いた
  修正方針が，再検証すると不足または誤りだった」型 (§ Corrections)
- **confirmed: 12**

うち **1 件** (Deferred 4) を消化した．**自動帯に入ったのはこの 1 件だけ**
で，残る 13 件はすべてゲート付きか判断帯である — これはこの backlog が
既に 4 回の `/audit-backlog` run で削られた後の状態で，安く消せるものが
尽きて「GPU 実機が要る / 設計判断が要る / 選択されていない場所を巻き込む」
硬い核だけが残っているためである．

## Classification

判断コスト P1-P6 + ゲートで分類した．`P6 → P1` の順に評価し，最初に
引っ掛かったクラスを採る．

### 消化した 1 件

| ID | backlog 行 | 対象 | クラス | そのクラスに決めたテスト | ゲート |
|---|---|---|---|---|---|
| P3-1 | [2026-08-08 app/learning] Deferred 4 | `app/learning/callbacks.py` | **P3** | 蓄積テンソルの dtype・初期値・移送のタイミング・`reset` の分岐をすべて保つ．受理する全入力に対し，書き出す成果物 (TensorBoard スカラー，チェックポイント) も返す値 (`get_average_metrics` 等) も不変 | **なし → 自動帯** |

P3 の根拠は主張ではなく実測で置いた: `tests/maou/app/` +
`tests/maou/interface/` の **1178 passed / 2 skipped** が変更前後で同一，
加えて表と実挙動を突き合わせる characterization test を新設した (§ Applied)．

### 処理しなかった 13 件

| ID | backlog 行 | 対象 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|---|
| P3-2 | Deferred 2 | `app/learning` | P3 | Stage1/Stage2 経路の重複除去のみのはず | **G3** — ~400 行の学習経路リファクタで，この環境では等価性を実証できない |
| P4-1 | Deferred 3 | `app/learning` | P4 (**記録の見立てより上**) | アダプタ統合は `isinstance` 分岐の**検証**まで消す (§ Corrections) | **G2** — テストの書き換えを伴う |
| P4-2 | Deferred 5 | `app/learning` | P4 | 毎バッチのホスト同期の除去は挙動が変わる | **G1** — GPU 無し |
| P4-3 | Deferred 6 | `app/learning` | P4 | `stream.synchronize()` → `wait_stream()` は同期意味論の変更 | **G1** |
| P4-4 | Deferred 7 | `app/learning` | P4 | GNS はアダプティブバッチ制御器を駆動するので数値等価性の確認が要る | **G1** |
| P4-5 | [2026-08-10 infra/file_system] D10+D11 (1) | `infra/file_system` | P4 | `total_pages()` の意味 (ファイル数 / yield 数) を変える | **G4** |
| P4-6 | D13 (残り) | `infra/file_system` + `app/learning` | P4 | `__getitem__` の per-sample `np.empty` を根絶する改修 | **G2** — `app/learning/dataset.py` と ABC を巻き込む |
| P4-7 | O9 | `infra/bigquery` | P4 | 返る行集合が変わる (§ Corrections) | **G1** — BigQuery 無し |
| P6-1 | D5 (残り) | `infra/file_system` | P6 | `cache_mode` ノブの廃止 | **G4** — O5 と一体 |
| P6-2 | D14 (b) | `infra/file_system` | P6 | 継承を外すと `preprocess.DataSource` の公開契約が消える | **G2** — `infra/utility/benchmark_polars_io.py:419-451` を巻き込む |
| P6-3 | O5 | `infra/console` + `infra/object_storage` | P6 | CLI オプションの削除・再定義 | **G4** |
| P6-4 | N6-2 | `app/pre_process` | P6 | 行の文言上，改名を含む | **G4** |
| P1-1 | N4 | `tests/` | P1 | `tests/` しか触らないので version bump も不要 | **G4** — 2 案が今回も 1 案に潰れなかった |

**質問は上げていない．** 判断帯のどれも，今セッションで実装する予算が
無い状態で聞いても答えを実装できず，「同じ所見をユーザに 2 回レビュー
させる」だけになるため — one-check 原則の後半 (「間違えた方を実装すると
レビューして捨てる作業が無駄になる」) が成立しない．代わりに行の文言を
鋭くして次の run に渡した (§ Re-triaged)．

## Consumed

| backlog 行 | 由来の記録 | 対象 | 出荷したもの | PR |
|---|---|---|---|---|
| Deferred 4 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/app/learning/callbacks.py` | `_ACCUMULATORS` / `_COUNTERS` 宣言表と，そこから導出する `_init_accumulators` / `_ensure_device` / `_reset_accumulators` | [PR #494](https://github.com/dousu/maou/pull/494) |

## Applied

`refactor(learning): 蓄積テンソルを宣言表から導出する` — `ead707d`

- `callbacks.py:144` `BaseCallback` — `_ACCUMULATORS: tuple[tuple[str,
  torch.dtype | None], ...]` と `_COUNTERS: tuple[str, ...]` を追加し，
  `_init_accumulators()` / `_ensure_device()` / `_reset_accumulators()`
  の 3 メソッドをそこから導出する形で実装．
- **同形に複製されていた `_ensure_device` 6 本を削除** (旧
  `:238` `LoggingCallback` / `:362` `ValidationCallback` / `:1047`
  `TimingCallback` / `:1436` `Stage2F1Callback` / `:1561`
  `Stage1AccuracyCallback` / `:1708` `Stage3LossCallback`)．基底の 1 本に
  なった．
- **`ValidationCallback` の 13 テンソル × 3 箇所の手書きを表 1 本に集約**
  (`__init__` / `_ensure_device` / `reset`)．カウンタ 6 個も同様．
  `reset()` の本体は 49 行 → 1 行．
- 差し引き `callbacks.py` は **-210 / +154 行**．

**挙動不変の担保**: `dtype=None` の要素はこれまでどおり `torch.tensor(0.0)`
(既定の浮動小数 dtype に追随) で作り，`torch.zeros((), dtype=torch.float32)`
のように dtype を固定しない．`_reset_accumulators` は「移送済みなら
`zero_()`，未移送なら作り直し」という旧 `reset()` の分岐をそのまま保つ．

**回帰テスト**: `tests/maou/app/learning/test_callbacks.py`
`TestAccumulatorDeclaration` — 5 クラス × 4 観点 (生成 / 移送 / ゼロ化 /
移送後のテンソル同一性) + `ValidationCallback` の「表に書き忘れた
テンソルが無いこと」を `vars()` と突き合わせる 1 本．

**非空虚性を 2 通りの neuter で確認**:

1. `_ensure_device` が表の最後の 1 件を取りこぼす (= 手書き時代の欠陥そのもの)
   → `test_ensure_device_moves_every_declared_accumulator` が 5 クラスとも失敗．
2. `_reset_accumulators` が分岐を捨てて常に作り直す (= 移送後に CPU テンソルへ
   戻り，次エポックで device mismatch)
   → `test_reset_after_device_migration_keeps_tensor_identity` が 5 クラスとも失敗．

いずれも復元後は全件 pass．

## In flight

なし．この run の PR は自動帯 (P3) と監査記録 (P1) だけで構成されており，
判断帯の項目を含まない．

## Re-triaged

いずれも **行は残す**．文言を鋭くしたものだけ挙げる．

1. **Deferred 3 (P4-1)** — 記録は「6 クラスは 3 組の重複で，`Stage1ModelAdapter`
   と `Stage2ModelAdapter` は **0 文字差**」とし，統合は自明としていた．
   再検証で**統合の隠れたコスト**が出た: `tests/.../test_stage_component_factory.py:297`
   は `isinstance(components.model, Stage1ModelAdapter)`，`:398` は
   `isinstance(..., Stage2ModelAdapter)` を主張している．両名を同一クラスの
   別名にすると**どちらのアサーションも通ってしまい，2 本のテストが
   識別力を失う** (Stage 1 のテストが Stage 2 のアダプタを構築しても緑になる)．
   したがって統合は「テストの書き換えとセット」でしか出荷できず，
   P3 ではなく **P4 + G2**．
2. **O9 (P4-7)** — 記録は「`TABLESAMPLE` をページごとに引き直すので件数が
   ずれる」としていたが，再検証すると**非決定性は二重**である．
   (i) `TABLESAMPLE SYSTEM` はクエリごとに独立に評価されるので
   `__get_total_rows`(`:212-234`) が数えた行集合と `__fetch_from_bigquery`
   (`:395-411`) が返す行集合は別物，(ii) その上で `LIMIT/OFFSET` を
   **ORDER BY 無しの再サンプル結果**に掛けているので，同じ `page_num` を
   2 回引いても同じ行が返る保証が無い．つまり `sample_ratio` 指定時の
   `get_page`/`indicies` は「ずれる」ではなく**再現性が無い**．
   修正の向きも 1 案に潰れていない: (a) サンプルを一時テーブルへ 1 度
   実体化してそこからページングする，(b) `FARM_FINGERPRINT` 等の決定的
   ハッシュ条件へ置き換える (キー列の決めが要る)，(c) `sample_ratio` と
   ページングの併用を拒否する．**G1 は据え置き** (BigQuery がこの環境に無い)．
3. **D10+D11 (1) (P4-5)** — 記録の `file_data_source.py:898` は
   **`:775` へ移動**している (`total_pages()`)．中身と結論 (dormant，
   production caller ゼロ，意味の決めが要る) は不変．
4. **N4 (P1-1)** — **5 回連続で実害を確認**．この run もコンテナが冷えた
   状態から始まり，`uv run` の base 同期 (maturin による Rust 拡張の
   ビルドを含む) だけで数分，そのあと `uv sync --extra cpu` が別途必要
   だった．base 環境のままだと `import torch` が失敗し，`app/learning` の
   テストは**丸ごと skip されて緑に見える** — 今回の変更は
   `app/learning/callbacks.py` なので，入れずに QA を回していれば
   **1178 件のうち中核が無検証のまま通過していた**．2 案 (薄いテストへの
   切り出し / CPU extra の必須化) はいずれも今回も潰れず **G4 継続**．

## Corrections to the source records

[2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) の Deferred 3
に，上記 1 の内容で訂正を追記した．記録が「`isinstance` 分岐は同一の
2 クラスを選び分けるためだけに存在する」と書いた点が誤りで，実際には
**未対応の head 型を `TypeError` で弾く検証**も担っている
(`stage_component_factory.py:876-882` の `else` 腕)．統合時にこの腕を
落とすと，これまで拒否されていた head 型が黙って通るようになる．

## Doc findings

なし．`_ensure_device` や蓄積テンソルの構成を記述した durable doc は
存在しない (`docs/stage2-speed-investigation.md` が `Stage2F1Callback` に
触れているが，記述しているのは「GPU テンソル上で蓄積しエポック終了時
のみ `.item()`」という**不変のまま保った**性質)．したがって
`reviews/*.md` の提案は起こしていない．

## Out of scope

**新規 1 件** — `coverage.md` の Out-of-scope backlog に行を追加した
(**N8**)．

`TimingCallback` の `_last_batch_loss` は，`on_batch_end`
(`callbacks.py:1030`) で `self._last_batch_loss = loss_detached` と
**呼び出し側のテンソルを参照ごと持つ**．`loss_detached =
context.loss.detach()` は `context.loss` とストレージを共有するので，
次の `reset()` が走ると `_reset_accumulators` の `zero_()` が
**呼び出し側の loss テンソルをその場で 0 にする**．`_total_loss` が
`+=` で自前のストレージに足し込むのと非対称で，こちらだけ所有権が
外に漏れている．

今回はこれを直していない: 直すなら `copy_()` にするのが自然だが，
「`reset()` が `context.loss` を 0 にしなくなる」という**観測可能な
変化**を伴い，それを誰も読んでいないことを安く証明できないため
**フェイルセーフ側に倒して P4** とした (P3 の「受理する全入力に対し
返す値が不変」を満たすと言い切れない)．今回の変更で入った
`test_reset_after_device_migration_keeps_tensor_identity` が，
「`reset` はその場で `zero_()` する」契約を明示的に固定しているので，
この所有権の穴は以前より見つけやすくなっている．

## Environment notes

- **GPU 無し** — `G1` を付けた 4 件 (Deferred 5/6/7, O9 の一部) は
  この環境で正しさを確立できない．CUDA 同期の除去は実機測定なしに
  出荷しない．
- **BigQuery 無し** — O9 の検証は fake client の単体テストまでしか
  できず，`TABLESAMPLE` の実際の非決定性は再現できない．
- **コンテナが冷えた状態から開始** — `uv run` が maturin 経由で Rust
  拡張をビルドするところから始まり，さらに `uv sync --extra cpu` が
  必要だった (N4 の実害，5 回目)．**QA はすべて CPU extra を入れた後に
  実行している**．
- 実行した QA: `uv run ruff format src/ tests/` (2 files reformatted) /
  `uv run ruff check src/ tests/ --fix` (1 fixed, 0 remaining) /
  `uv run mypy src/` (**Success: 135 source files**) /
  `uv run pytest tests/maou/app/ tests/maou/interface/`
  (**1178 passed, 2 skipped**)．Rust 側は変更していないので `cargo` は
  回していない．

## Reconciliation (6d)

```
14 (触れた backlog 行) + 1 (新規所見) = 15
  = 1  (resolved:   Deferred 4 — 行削除，PR #494 でマージ)
  + 0  (in flight:  判断帯の PR は無い)
  + 13 (re-triaged: 行を残した 13 件．うち 4 件は文言を鋭くした
        — Deferred 3 / O9 / D10+D11(1) / N4．
        残る 9 件は分類とゲートを付けたうえで文言変更なし
        — Deferred 2/5/6/7, D5, D13, D14(b), O5, N6-2)
  + 1  (new row:    N8 — TimingCallback._last_batch_loss の所有権)
  + 0  (not a finding)
```

**backlog 行数: 14 → 14** (Deferred 4 を削除し，N8 を追加)．
main table の `src/maou/app/learning` 行の Open items は
`6 deferred` → `5 deferred` に更新した．
