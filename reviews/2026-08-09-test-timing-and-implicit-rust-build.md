---
status: applied
applied_in: 830bbdf
date: 2026-08-09
target:
  - docs/testing-guide.md
risk: low
reversibility: easy
---

# テスト所要時間と `uv run` の暗黙 Rust ビルドを記載する

## Trigger

`/audit-backlog` の doc 追随コミットで pre-commit の `test` フックが
20分以上返らなかった．私はこれを「`uv run` の再解決が
`pypi.nvidia.com` (tensorrt) で詰まっている」と報告し，フックを SKIP
した．**この診断は誤りだった**とユーザ指摘を受けて実測した．

切り分けの結果:

- `target/debug/lib_rust.so` のタイムスタンプが **04:38**．ハングした
  実行の開始は **04:18** — つまりあの20分は **Rust debug ビルドが
  コールドで走って完了していた**．失敗ではなかった．
- ビルドが温まった状態で `uv run python -c pass` は **49秒**
  (ビルド48秒 = インクリメンタル)．
- `uv run pytest -q` は **1725 passed, 54 skipped in 191s** で完走．
  `test` フックは壊れていない．

`docs/testing-guide.md` には所要時間の記載が**一切ない**
(`grep '分\|minute\|時間\|maturin'` で0件)．コストが見えないので，
同じ誤診断が繰り返されうる．ユーザ要望は「以後時間を意識して test
できるようにドキュメント記載しておくこと」．

## 改訂 (2026-08-09, 承認後)

初版は **profile 変更前**の数値で書かれていた．承認後の適用時点で，
同じ会話の後続作業により前提が変わっている:

- `[tool.maturin] profile` が `dev` → `py-ext` (`6cb9248`)
- `.cargo/config.toml` から `jobs = 1` を削除 (`7331986`)

初版の記述のうち以下は**現在は誤り**なので，そのまま適用すると
「誤診断の再発防止」という本提案の目的に反する:

| 初版の記述 | 現在の事実 |
|---|---|
| `profile = "dev"` なので debug ビルド | 既定は `py-ext` |
| 暗黙ビルド コールド約20分 / ウォーム約50秒 | 10分09秒 (jobs=4) / 0〜1秒 |
| `uv run pytest` 全体 191秒 | 92〜100秒 |
| release 拡張が約2倍速い (100秒 vs 191秒) | 差は検出不能 (95.5秒 vs 92〜100秒) |
| `uv run` が debug で上書きするので `.venv/bin/python` を直接呼ぶ | パリティ修正済みで上書きされない．回避策不要 |
| `profile` を release に変えるのは範囲外 | py-ext に変更済み |

また初版が「release 拡張は約2倍速い」と書いた根拠は**負荷由来の外れ値**
だった．py-ext のスイートを計4回測ると 134.75 / 約110 / 99.93 / 92.23 秒で，
最初の1回だけ他のビルドと並行していた．`overflow-checks` の実行時代償は
**測定不能**が結論．

以下が改訂後の Proposed change である．

## Proposed change

`docs/testing-guide.md` § "Running Tests by Layer" のコードブロック直後
に新規小節を追加する (既存記述の変更はない)．

**After** (追加分):

````markdown
### 実行時間と Rust 拡張のビルドコスト

Python テストは `maou._rust` 拡張に依存するので，所要時間は「拡張のビルド」
と「テスト本体」に分かれる．どちらもプロファイルと並列度で大きく変わる．

`[tool.maturin] profile = "py-ext"` が Python 拡張の既定 (定義は root
`Cargo.toml`)．明示ビルド (`maturin develop`) と `uv run` の暗黙リビルドが
同じプロファイルを使うので，最適化拡張が debug 拡張に差し替わることはない．

#### 実測値 (2026-08-09, 4 CPU / 16GB, crate registry 温)

コールドビルド:

| プロファイル | `jobs=1` | `jobs=4` | ピーク RSS |
|---|---|---|---|
| py-ext (既定) | 37分47秒 | **10分09秒** | 3.8GB / **7.2GB** |
| release | 30分41秒 | — | 3.7GB (jobs=1) |

反復ビルド (`rust/maou_shogi` を実際に内容変更):

| プロファイル | 再ビルド | 変更なし | `uv run` (変更なし) |
|---|---|---|---|
| py-ext (既定) | **6秒** | 3秒 | 0〜1秒 |
| release | 129〜133秒 | 3秒 | — |

テスト本体 (`pytest` 報告値, 1725 passed / 54 skipped):

| 拡張 | 時間 |
|---|---|
| py-ext (既定) | 92〜100秒 |
| release | 95.5秒 |
| dev (2026-08-09 以前の既定) | 191秒 |

`.so` サイズ: py-ext 56MB / release 50MB / dev 234MB．

#### 読み方

- **並列度が最大のレバー**．コールドビルドが `jobs=1` と `jobs=4` で
  **3.7倍**違い，プロファイル選択の影響を桁で上回る．
  `.cargo/config.toml` は `jobs` を設定していないので cargo 既定 (CPU 数)
  が効く．ただし `jobs=4` はピーク 7.2GB を要求するので，メモリ制約環境は
  `scripts/dev-init.sh` が user cargo config に `jobs = 1` を書いて絞る．
- **`--release` を付けると反復ビルドが 20倍以上遅くなる**．
  `lto = "thin"` は cdylib を再リンクするたびに依存グラフ全体の thin LTO を
  やり直し，**この工程はキャッシュできない**ため，再ビルドごとに約 127秒 の
  固定費が乗る．py-ext は `lto = false` なのでこれが無い．
  release 相当の数値が必要な性能計測を除き，`--release` は付けない．
- **テスト本体の測定は ±25% ばらつく**．他のビルドと並行させると 92秒の
  スイートが 135秒に見える．性能比較をするなら他の作業を止め，複数回測る．

#### pre-commit の `test` フック

`uv run pytest -v -s` を `always_run: true` で回すので
(`.pre-commit-config.yaml`)，**doc だけの変更でも**テストが走る．
拡張がコールドな新しいコンテナでは最初のコミットがビルド時間ごと待たされる
ので，先に

```bash
uv run python -c pass     # 拡張ビルドを温めるだけ
```

を一度流しておくと以後のコミットが速い．

#### 遅いのか壊れたのかの切り分け

`uv run` が返ってこないとき，この2つは見分けられる:

- **遅いだけ** — `Building maou @ file:///...` を出したあと，cargo の
  コンパイル中は**無出力で数分〜数十分黙る**．`pgrep -f rustc` で進行中か
  確認できる．
- **本当に失敗** — 1分程度で `hint:` / `help:` ブロックを出して終了する．
  例: `tensorrt-cu12-libs` の sdist ビルドは `nvidia-smi` を要求するため
  CPU only 環境では失敗しうる (2026-08-09 に一度観測，同日の再試行では
  再現しなかった)．

**無出力で長い = ビルド中，メッセージが出て止まる = 失敗**と読む．
待つ前に `pgrep -f rustc` を見るのが最短の切り分けになる．
````

## Motivation

コストが文書化されていないと，「20分返ってこない」を故障と誤読してフックを
SKIP する判断に流れる．今回それが実際に起きた．所要時間の表と
`pgrep -f rustc` という具体的な切り分け手段があれば，待つべき場面と
調査すべき場面を区別できる．

反復ビルドの 6秒 vs 129秒 は，`--release` を安易に付けると開発ループが
20倍遅くなることを示す．これはどこにも書かれておらず，しかも
`docs/rust-backend.md` は従来「性能計測は --release で」と勧めていた．

## Alternatives considered

- **`audits/` の Environment notes に書く**．却下: あれは1 run の記録で，
  「テストにどれくらいかかるか」を探す人は開かない．
- **CLAUDE.md § "重いテスト (Rust dfpn)" に併記**．却下: あの節は Rust の
  `[SLOW]` テスト固有の話で，Python 側の全体所要時間とは読者の関心が違う．
  CLAUDE.md の Documentation Links は既に `docs/testing-guide.md` を
  指しているので到達性は足りている．

## What this enables

- コミット前に「今のコンテナは温まっているか」を意識でき，長い待ちを最初の
  1回に寄せられる．
- フックが返らないときに，待つ / 調べるの判断が数秒で付く．
- `--release` の反復コストが可視化され，開発ループで誤って付けなくなる．

## What this constrains

- 実測値なので，マシンやコンテナ世代が変わればずれる．測定日と測定環境
  (4 CPU / 16GB / registry 温) を明記して「実測値・環境依存」と分かる形に
  した．
- `[tool.maturin] profile` や `.cargo/config.toml` の `jobs` を変えた場合は
  この表の前提が崩れる．変更時はここも直す必要がある．

## Rollback plan

`docs/testing-guide.md` に小節を1つ追加するだけなので，該当ブロックを削除
すれば元に戻る．既存記述には触れていない．
