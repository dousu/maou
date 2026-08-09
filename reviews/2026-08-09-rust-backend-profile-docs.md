---
status: pending
applied_in:
date: 2026-08-09
target:
  - docs/rust-backend.md
  - docs/rust-build-optimization.md
risk: low
reversibility: easy
---

# py-ext プロファイル導入に伴う Rust ビルド doc の追随と既存ドリフト訂正

## Trigger

`6cb9248` / `7331986` で Python 拡張のビルド既定を変更した:

- `[tool.maturin] profile`: `dev` → `py-ext` (release 継承 + `lto = false`
  + `debug-assertions`/`overflow-checks` 有効)
- `.cargo/config.toml` から `jobs = 1` を削除 (cargo 既定 = CPU 数)

これにより Rust ビルド系 doc の記述が実態と合わなくなる．
加えて，確認中に**変更とは無関係な既存ドリフト2件**を発見したので併せて
提案する (放置すると読者が誤った前提でビルド設定をいじる)．

`docs/testing-guide.md` 側の所要時間記載は別提案
(`2026-08-09-test-timing-and-implicit-rust-build.md`, applied `830bbdf`) で
済んでいる．本提案はビルド設定そのものの記述を扱う．

## 1. docs/rust-backend.md § "Build Profiles"

### 現状 (201-220行付近)

````markdown
### Build Profiles

**Development (default):**
```bash
uv run maturin develop  # Uses [profile.dev]
# opt-level = 0, codegen-units = 1, incremental = true
```

**Production (optimized):**
```bash
uv run maturin develop --release  # Uses [profile.release]
# opt-level = 3, codegen-units = 1, lto = "thin"
```

**Balanced (memory-optimized):**
```bash
CARGO_PROFILE=mem-opt uv run maturin develop  # Uses [profile.mem-opt]
# opt-level = 2, codegen-units = 1, lto = "thin"
```
````

**既存ドリフト**: dev の説明 `codegen-units = 1, incremental = true` は
両方とも誤り．root `Cargo.toml` の `[profile.dev]` は
`codegen-units = 16`, `incremental = false` (sccache 併用のため) である．

### 提案

````markdown
### Build Profiles

**Python 拡張 (既定):**
```bash
uv run maturin develop  # Uses [profile.py-ext]
# release 継承 + opt-level = 3, lto = false,
# debug-assertions = true, overflow-checks = true
```
明示ビルドと `uv run` の暗黙リビルドが同じプロファイルを使うので，
最適化拡張が別プロファイルの拡張に差し替わることはない．
`lto = false` は反復ビルドのためで，thin LTO は cdylib を再リンクする
たびに依存グラフ全体をやり直し，キャッシュできない (実測: 再ビルドごとに
約 127秒 の固定費)．`debug-assertions`/`overflow-checks` は
`rust/maou_rust` に `#[test]` が無く Python 経由が唯一の実行経路である
ため有効にしている．

**出荷 wheel:**
```bash
uv run maturin develop --release  # Uses [profile.release]
# opt-level = 3, codegen-units = 1, lto = "thin"
```
CI の wheel ビルドはこちら (`--release` を明示)．
**開発ループでは付けないこと** — 反復ビルドが 6秒 → 130秒 になる．

**cargo 直接 (Rust 単体テスト):**
```bash
cargo test -p maou_shogi  # Uses [profile.dev]
# opt-level = 0, codegen-units = 16, incremental = false
```
`[tool.maturin] profile` は maturin 経由のビルドにしか効かない．
`cargo` を直接叩く場合は従来どおり dev/release が選ばれる．

**Balanced (memory-optimized):**
```bash
CARGO_PROFILE=mem-opt uv run maturin develop  # Uses [profile.mem-opt]
# opt-level = 2, codegen-units = 1, lto = "thin"
```
````

## 2. docs/rust-backend.md § "Automatic Optimizations" (174-181行付近)

### 現状

```markdown
**1. Environment Variables (scripts/dev-init.sh):**
```bash
export CARGO_BUILD_JOBS=1              # Single parallel job
export RUSTFLAGS="-C codegen-units=1 -C incremental=1"  # Sequential compilation
```

**2. Build Profiles (Cargo.toml):**
- `codegen-units = 1` for all profiles (dev, release, mem-opt)
- Thin LTO (Link-Time Optimization) for smaller binaries
- Sequential compilation prioritizes memory over build speed
```

**既存ドリフト**: 「`codegen-units = 1` for all profiles」は誤り．
`[profile.dev]` は `16`，`scripts/dev-init.sh` は user config で
release も `16` にしている．また `scripts/dev-init.sh` は
`export CARGO_BUILD_JOBS=1` ではなく user cargo config に
`[build] jobs = 1` を書く実装になっている．

### 提案

```markdown
**1. 並列度 (scripts/dev-init.sh):**
`scripts/dev-init.sh` は user cargo config
(`$CARGO_HOME/config.toml`) に以下を書く:
```toml
[build]
jobs = 1              # メモリ制約環境向けに直列化
```
リポジトリの `.cargo/config.toml` は **`jobs` を設定しない** (cargo 既定
= CPU 数)．tracked config に書くと潤沢な環境にも一律に効いてしまうため，
絞るのは環境ごとの判断として `dev-init.sh` の責務にしている．

並列度はビルド時間の**最大のレバー**で，プロファイル選択より桁で影響が
大きい (実測: py-ext コールドビルドが `jobs=1` で 37分47秒，`jobs=4` で
10分09秒 = 3.7倍)．代償はピーク RSS で，3.8GB → 7.2GB．
8GB 環境では `dev-init.sh` 経由の `jobs = 1` が必要．

**2. Build Profiles (Cargo.toml):**
- `codegen-units`: `dev` は 16 (並列 codegen でピークメモリを下げる)，
  `release` / `mem-opt` / `py-ext` は 1 (メモリ優先)．
  `dev-init.sh` は user config で `release` / `py-ext` も 16 にする
- Thin LTO は `release` / `mem-opt` のみ．`py-ext` は反復ビルドのため無効
- Sequential compilation prioritizes memory over build speed
```

## 3. docs/rust-backend.md § "性能計測時の注意" (535-540行付近)

### 現状

```markdown
`uv run maturin develop` は既定で **dev プロファイル** (`pyproject.toml` の
`[tool.maturin] profile`) でビルドされ，release 比 ~6 倍遅い．Python 経由で
ソルバーの性能を測る場合は `uv run maturin develop --release` でビルドし直すこと
(配布 wheel は CI が `--release` でビルドするため影響しない)．
```

既定が最適化済みになったので，この節の前提が消えた．

### 提案

```markdown
`uv run maturin develop` は既定で **py-ext プロファイル**
(`pyproject.toml` の `[tool.maturin] profile`) でビルドされ，
`opt-level = 3` なので Python 経由の性能計測にそのまま使える
(2026-08-09 以前の既定は dev で release 比 ~6 倍遅く，計測前に
`--release` での再ビルドが必要だった)．

ただし py-ext は `debug-assertions` / `overflow-checks` を有効にしている
ため，出荷 wheel と完全に同じ数値にはならない．dfpn のノード/秒のように
絶対値を出荷構成と揃えたい場合のみ `--release` を使う．
**開発ループでは付けないこと** — 反復ビルドに約 127秒 の固定費が乗る．
```

## 4. docs/rust-build-optimization.md (113行付近)

### 現状

```markdown
- **`--release` は既定では付かない** (`[tool.maturin] profile = "dev"` の
  ため省くと debug ビルド)．
```

### 提案

```markdown
- **`--release` は既定では付かない** (`[tool.maturin] profile = "py-ext"`)．
  py-ext は `opt-level = 3` なので省いても debug ビルドにはならないが，
  thin LTO は無効なので出荷 wheel とは別物．workflow は `--release` を
  明示している．
```

## Motivation

`docs/rust-backend.md` § Build Profiles は「どのコマンドがどのプロファイルを
使うか」の一次情報で，ここが古いと読者は `--release` を付けるべきかを誤る．
今回の変更で `--release` は**開発ループでは有害** (反復 20倍以上) になったが，
同じ doc が従来「性能計測は `--release` で」と勧めていたため，放置すると
矛盾した指示になる．

既存ドリフト2件 (`codegen-units = 1` for all profiles / dev の
`incremental = true`) は今回の変更とは無関係だが，同じ節にあり，
メモリ調整をしようとする読者が誤った現状認識から出発することになる．

## Alternatives considered

- **既存ドリフトは別提案に分ける**．却下: 同じ節の同じ表を触るので，
  2回に分けると片方適用時点で節内が不整合になる．
- **§ 性能計測時の注意 を削除する**．却下: 「py-ext は出荷構成と完全に
  同じ数値ではない」という注意は依然必要で，むしろ内容を差し替えるべき．

## What this enables

- `--release` を開発ループで付けない判断が doc から読み取れる．
- 並列度がプロファイルより効くという実測が記録され，メモリ調整の議論が
  「まず jobs を見る」から始められる．

## What this constrains

- 実測値 (37分47秒 / 10分09秒 / 127秒 / 3.8GB / 7.2GB) を含むので，
  環境が変われば数字がずれる．測定環境を併記する形にした．
- `[tool.maturin] profile` を変えた場合はこの4箇所すべてを直す必要がある．

## Rollback plan

いずれも記述のみ．`docs/rust-backend.md` 3箇所，
`docs/rust-build-optimization.md` 1箇所の差分なので revert は容易．
