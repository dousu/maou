---
status: pending
applied_in:
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

## Proposed change

`docs/testing-guide.md` § "Running Tests by Layer" のコードブロック直後
に新規小節を追加する (既存記述の変更はない)．

**After** (追加分):

````markdown
### 実行時間と `uv run` の暗黙 Rust ビルド

`uv run` は Rust 拡張が古いと**毎回暗黙に再ビルドする**．
`[tool.maturin] profile = "dev"` (`pyproject.toml`) なので debug ビルドで，
新しいコンテナでは初回だけ大きな待ちが出る．

| 操作 | コールド | ウォーム |
|---|---|---|
| `uv run <任意>` の暗黙ビルド (debug) | **約20分** | 約50秒 |
| `maturin develop --release` (明示) | **約31分** | — |
| `uv run pytest` 全体 (debug 拡張) | ビルド + 191秒 | 191秒 |
| `.venv/bin/python -m pytest` 全体 (release 拡張) | — | 100秒 |
| 3ディレクトリ subset (363件) | — | 38秒 |

(2026-08-09 に remote container で実測．1725 passed / 54 skipped 時点．
CPU 8コア相当・GPU なし．マシンによって上下する．)

**pre-commit の `test` フックは `uv run pytest -v -s` で
`always_run: true`** (`.pre-commit-config.yaml`) なので，**doc だけの
変更でも**このコストを払う．新しいコンテナでは最初のコミットの前に

```bash
uv run python -c pass     # 拡張ビルドを温めるだけ
```

を一度流しておくと，以後のコミットは数分で済む．

**release 拡張の方が実行時間は約2倍速い** (100秒 vs 191秒)．Rust debug
ビルドは実行時性能が落ちるため．ビルドには31分かかるが，テストを何度も
回すなら元が取れる．ただし **`uv run` を実行すると debug 拡張で
上書きされる**ので，release の速度を保ちたい場合は `uv run` を使わず
`.venv/bin/python -m pytest` を直接呼ぶ．

#### 遅いのか壊れたのかの切り分け

`uv run` が返ってこないとき，この2つは見分けられる:

- **遅いだけ** — `Building maou @ file:///...` を出したあと，cargo の
  コンパイル中は**無出力で数十分黙る**．`pgrep -f rustc` で進行中か
  確認できる．
- **本当に失敗** — 1分程度で `hint:` / `help:` ブロックを出して終了する．
  例: `tensorrt-cu12-libs` の sdist ビルドは `nvidia-smi` を要求するため
  CPU only 環境では失敗しうる (2026-08-09 に一度観測，同日の再試行では
  再現しなかった)．

**無出力で長い = ビルド中，メッセージが出て止まる = 失敗**と読む．
待つ前に `pgrep -f rustc` を見るのが最短の切り分けになる．
````

## Motivation

コストが文書化されていないと，「20分返ってこない」を故障と誤読して
フックを SKIP する判断に流れる．今回それが実際に起きた．
所要時間の表と `pgrep -f rustc` という具体的な切り分け手段があれば，
待つべき場面と調査すべき場面を区別できる．

また release / debug で実行時間が2倍違い，かつ `uv run` が release を
debug で黙って上書きするという挙動は，知らないと「テストが急に遅く
なった」と誤認する類のもので，どこにも書かれていない．

## Alternatives considered

- **`audits/` の Environment notes に書く**．却下: あれは1 run の記録で，
  「テストにどれくらいかかるか」を探す人は開かない．
- **CLAUDE.md § "重いテスト (Rust dfpn)" に併記**．却下: あの節は
  Rust の `[SLOW]` テスト固有の話で，Python 側の全体所要時間とは
  読者の関心が違う．CLAUDE.md の Documentation Links は既に
  `docs/testing-guide.md` を指しているので到達性は足りている．
- **`profile = "release"` に変える**．却下: 本提案の範囲外 (コード変更)．
  暗黙ビルドが31分になり，doc 修正のコミットまで重くなる可能性がある．
  トレードオフの評価が別途必要．

## What this enables

- コミット前に「今のコンテナは温まっているか」を意識でき，20分の待ちを
  最初の1回に寄せられる．
- フックが返らないときに，待つ / 調べるの判断が数秒で付く．

## What this constrains

- 実測値なので，マシンやコンテナ世代が変わればずれる．表に測定日と
  測定環境を明記して「実測値・環境依存」と分かる形にした．
- `profile` を `release` に変えた場合はこの表の前提が崩れる．変更時は
  ここも直す必要がある．

## Rollback plan

`docs/testing-guide.md` に小節を1つ追加するだけなので，該当ブロックを
削除すれば元に戻る．既存記述には触れていない．
