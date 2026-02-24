# Rust ビルド高速化 設計ドキュメント

## 目的

Rust 拡張モジュール(`maou._rust`)のビルド時間を短縮する．

## ユースケース

### UC1: Codespaces DevContainer での開発ビルド

- **環境**: 2-core / 4GB RAM の低スペック環境
- **ビルド頻度**: `cargo clean` からのフルビルドは初回1回のみ．以降は Rust コード変更時にリビルド
- **優先度**: Rust コード変更 → ビルド成果物(`.so`)生成までの時間を最小化
- **最適化レベル**: dev プロファイル(最適化不要)
- **備考**: editable install が必須(`src/maou/` の Python コード変更が即座に反映される必要あり)

### UC2: Google Colab での機械学習実験

- **環境**: GPU インスタンス(T4/A100)，Rust ツールチェインなし
- **ビルド頻度**: 環境セットアップ時の1回のみ
- **リードタイム**: 開発環境でコード編集 → Colab で試すフローのため，短いほどよい
- **最適化レベル**: release プロファイル(ML ワークロードの性能が重要)
- **備考**: editable install は不要．wheel でまとめてインストールしてよい

## 方針概要

```
┌──────────────────────────────────────────────────┐
│ GitHub Actions (main ブランチ push 時)            │
│   maturin build --release → wheel 作成           │
│   sccache + GHA Cache でビルドキャッシュ           │
│   GitHub Releases にアップロード                  │
└───────────────┬──────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Colab (UC2)   │
        │ wheel install │
        │ editable 不要 │
        │ release 最適化│
        └───────────────┘

┌────────────────────────────────────────────┐
│ DevContainer (UC1)                         │
│   sccache (ローカルディスクキャッシュ)       │
│   + incremental = false                    │
│   + maturin develop (editable install)     │
│   GitHub Actions の成果物は使わない         │
└────────────────────────────────────────────┘
```

UC1 と UC2 は独立した経路である．DevContainer は自前でローカルビルドを行い，
GitHub Actions の wheel は Colab 専用．

| コンポーネント | 対象ユースケース | 手法 |
|---|---|---|
| GitHub Actions wheel ビルド | UC2 (Colab) | `maturin build --release` → GitHub Releases |
| sccache (GitHub Actions Cache) | UC2 (CI の wheel ビルド高速化) | `maturin-action` の `sccache: 'true'` |
| sccache (ローカルディスク) | UC1 (DevContainer) | `RUSTC_WRAPPER=sccache` + `~/.cache/sccache/` |

---

## 実装レイヤーごとの設計と選択肢比較

### レイヤー1: CI での wheel ビルド

#### 方針: GitHub Actions + `maturin-action` + GitHub Releases

main ブランチへの push をトリガーに wheel をビルドし，GitHub Releases に配置する．

#### 選択肢比較

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **GitHub Actions + Releases** | 無料，GitHub エコシステム内で完結，`gh` CLI で簡単にDL | Release 管理が必要 | **採用** |
| GitHub Actions + Artifacts | Release 不要で手軽 | 90日で有効期限切れ，URL が不安定，ワークフロー外からのDLに `gh` API が必要 | 不採用 |
| 手動ビルド + アップロード | CI 不要 | 再現性なし，属人化 | 不採用 |

#### トリガー設計の選択肢

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **main push 時に毎回ビルド** | 最新の wheel が常に利用可能，Colab へのリードタイム最小 | 不要なビルドが走る可能性 | **採用** |
| タグ push 時のみ | 不要なビルドを回避 | タグ付けの手間，Colab で試したい時にタグを打つのは不自然 | 不採用 |
| Rust ソース変更時のみ (`paths` フィルタ) | 効率的 | Python のみの変更でも Colab では最新 wheel が欲しい場合がある | 不採用 |
| 手動トリガー (`workflow_dispatch`) のみ | 完全にコントロール可能 | 忘れがち，自動化の恩恵なし | 補助的に採用 |

**注意点**: main push 時に毎回ビルドする方針だが，`workflow_dispatch` も併用して手動トリガーも可能にする．

#### wheel のバージョニング

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **latest タグへの上書き** | Colab 側のインストールコマンドが固定で済む | Release 履歴が残りにくい | **採用** |
| コミットSHA ベースのタグ | 追跡可能 | Releases が増殖，Colab 側で最新タグの解決が必要 | 不採用 |
| セマンティックバージョニング | 正式リリースに適切 | 開発中の頻繁な更新には重い | 将来的に検討 |

#### ビルド対象プラットフォーム

| プラットフォーム | 用途 | 判定 |
|---|---|---|
| **`manylinux_2_28` x86_64 + CPython 3.12** | Codespaces (Ubuntu)，Colab (Ubuntu) | **必須** |
| **`manylinux_2_28` x86_64 + CPython 3.11** | Python 3.11 環境 (Colab 等) | **必須** |
| macOS arm64 | ローカル開発 (Apple Silicon) | 当面不要(ローカルは sccache で対応) |
| Windows x86_64 | Windows 開発者 | 当面不要 |

**補足**: `requires-python = ">=3.11,<3.13"` のため，CPython 3.11 と 3.12 の両方の wheel をビルドする．Colab のデフォルト Python バージョンに依存しない安全な選択とする．

#### wheel の内容物

maturin の仕様上，1つの wheel に以下がすべて含まれる:

- Python コード (`src/maou/` 全体)
- コンパイル済み Rust 拡張 (`_rust.so`)
- CLI エントリポイント (`maou` コマンド)

Colab では wheel をそのままインストールして使うため，これで問題ない．
DevContainer では wheel は使わず sccache + `maturin develop` (editable install) を使う．

---

### レイヤー2: Colab 環境でのインストール

#### 方針: `uv pip install` + wheel URL

Colab 環境では Rust ツールチェインをインストールせず，プリビルト wheel をインストールする．

#### 選択肢比較

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **`uv pip install` + wheel URL** | シンプル，1コマンド | wheel URL の管理 | **採用** |
| `uv sync` + `[tool.uv.sources]` で URL 指定 | pyproject.toml で宣言的 | main の pyproject.toml を書き換える必要あり，editable install との競合 | 不採用 |
| `uv sync --no-install-project` + `uv pip install` | 依存を uv sync で，maou を wheel で | 2段階コマンド | 補助的に採用 |
| `[[tool.uv.index]]` flat index | uv のネイティブ機能 | GitHub Releases の HTML が flat index として適切でない場合がある | 不採用 |

#### Colab セットアップの流れ

```python
# Colab セル
# Python バージョンに応じて cp311 または cp312 の wheel を指定
!pip install uv
!uv pip install --system "maou @ https://github.com/{owner}/{repo}/releases/download/latest/maou-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
!uv pip install --system "maou[cuda]"  # GPU 依存の追加インストール
```

**考慮事項**:
- Colab はデフォルトでシステム Python を使うため `--system` フラグが必要
- wheel URL は latest Release を指すようにすることで固定化可能
- GPU 依存(torch CUDA 版等)は wheel に含まれないため，別途インストールが必要

---

### レイヤー3: DevContainer でのビルド高速化 (sccache)

#### 方針: sccache (ローカルディスク) + `incremental = false` + `codegen-units = 16`

`RUSTC_WRAPPER=sccache` を設定し，Cargo の incremental compilation を無効化する．
sccache のローカルディスクキャッシュでクレート単位のキャッシュを行う．

#### sccache の技術的特性

sccache を正しく評価するために，以下の特性を理解する必要がある:

**1. キャッシュ粒度: クレート単位**

sccache は `rustc` の呼び出しをラップする．Cargo は各クレートを1回の `rustc` 呼び出しでコンパイルするため，
キャッシュの粒度は**クレート単位**となる．1つの `.rs` ファイルが変更されると，そのクレート全体が再コンパイルされる．

**2. `cdylib` はキャッシュ対象外**

sccache がキャッシュできるのは `rlib` と `staticlib` のみ．
`bin`, `dylib`, `cdylib`, `proc-macro` はシステムリンクを伴うためキャッシュできない．

本プロジェクトのクレート構成:

| クレート | crate-type | sccache キャッシュ |
|---|---|---|
| `maou_io` | `lib` (rlib) | **可能** |
| `maou_index` | `rlib` | **可能** |
| `maou_rust` | `cdylib` | **不可** |

つまり，最終的な `_rust.so` を生成する `maou_rust` のコンパイル+リンクは**常に実行される**．
sccache で高速化できるのは `maou_io` と `maou_index` の部分のみ．

**3. `incremental = true` とは併用不可**

sccache は incremental compilation のアーティファクトをキャッシュできない．
`incremental = true` の場合，sccache はキャッシュを**スキップ**する(効果なし)．
sccache の効果を得るには `incremental = false` が必要．

**4. Cargo の fingerprint は sccache と独立**

Cargo は `target/` ディレクトリ内の fingerprint でソース変更を検知し，
未変更クレートのコンパイルをスキップする．これは sccache なしでも動作する．
sccache が価値を持つのは `target/` が存在しない場合(クリーンビルド)．

#### `incremental = false` + sccache の挙動

`maou_io/src/arrow_io.rs` を1行変更した場合:

```
1. Cargo fingerprint: maou_io のソース変更を検知
2. maou_io (rlib): sccache ミス → 全体再コンパイル → キャッシュに保存
3. maou_index (rlib): Cargo fingerprint でスキップ (rustc を呼ばない)
4. maou_rust (cdylib): maou_io に依存 → 再コンパイル + リンク (sccache 対象外)
```

`target/` が消えた後に同じコードをビルドした場合:

```
1. maou_io (rlib): sccache ヒット → コンパイルスキップ
2. maou_index (rlib): sccache ヒット → コンパイルスキップ
3. maou_rust (cdylib): sccache 対象外 → 再コンパイル + リンク
4. 依存クレート (polars, arrow 等の rlib): sccache ヒット → コンパイルスキップ
```

#### 選択肢比較

| 選択肢 | コード変更→成果物 | target/ 消失後 | メリット | デメリット | 判定 |
|---|---|---|---|---|---|
| **`incremental = false` + sccache** | クレート全体再コンパイル | rlib はキャッシュヒット，cdylib は再コンパイル | target/ 消失に強い | 変更クレート内の部分コンパイル不可 | **採用** |
| `incremental = true` (sccache なし) | 変更部分のみ再コンパイル | フルリビルド | コード変更時の粒度が細かい | target/ 消失で全喪失 | 不採用 |
| `incremental = true` + sccache | incremental で再コンパイル | sccache がキャッシュしていないためフルリビルド | — | 両方の弱点を合わせ持つ(sccache が効かない) | 不採用 |
| sccache なし，incremental なし | クレート全体再コンパイル | フルリビルド | 設定不要 | 最も遅い | 不採用 |

**判定理由**:

- `incremental = true` はクレート内の部分再コンパイル(サブクレート粒度)が可能だが，本プロジェクトのクレートは小さい(maou_io: 4ファイル，maou_index: 3ファイル，maou_rust: 1ファイル)ため，クレート全体再コンパイルとの差は小さい
- `Cargo.toml` では `codegen-units = 1` を維持 (CI は 7GB RAM で問題なし)．`scripts/dev-init.sh` では `codegen-units = 16` を維持する (理由: polars-core 等の大きな依存クレートのコンパイル時に `codegen-units = 1` ではピークメモリが増大し 4GB 環境で OOM が発生するため．sccache のキャッシュキーは値が固定されていれば安定するため，16 でも問題ない)
- `target/` の消失は DevContainer の再作成，ディスク容量逼迫時の `cargo clean`，`.venv` 再作成時の `uv sync` で頻繁に発生する
- sccache なら `target/` が消えても依存クレート(polars, arrow, pyo3 等)のキャッシュが残り，フルリビルドの大部分をスキップできる

**既知のトレードオフ**:

- sccache はキャッシュミス時に**約 8% のオーバーヘッド**が発生する(ハッシュ計算+キャッシュ書き込み，[NeoSmart ベンチマーク 2024](https://neosmart.net/blog/benchmarking-rust-compilation-speedups-and-slowdowns-from-sccache-and-zthreads/))
- `incremental = false` にすると，コード変更時は変更クレート全体の再コンパイルになるため，`incremental = true` 比で**コード変更→ビルドのサイクルは若干遅くなる**
- 一方，sccache のキャッシュヒット時は**約 80% の高速化**(同ベンチマーク)が報告されており，target/ 消失時のリカバリ効果が大きい
- `maou_rust` (`cdylib`) はいずれの設定でもキャッシュ不可のため，最終リンクステップの時間は変わらない

#### sccache がローカルで効果を発揮するシーン

| シーン | sccache なし | sccache あり | 効果 |
|---|---|---|---|
| 通常のコード変更 + ビルド | Cargo fingerprint でスキップ | 同左 (sccache は関与しない) | **差なし** |
| `target/` が消えた後 | **フルリビルド** (依存含む数分) | **rlib キャッシュヒット** (cdylib のみ再コンパイル) | **大幅短縮** |
| `.venv` 再作成後の `uv sync` | **フルリビルド** | **rlib キャッシュヒット** | **大幅短縮** |
| `cargo clean` 後 | **フルリビルド** | **rlib キャッシュヒット** | **大幅短縮** |
| ブランチ切替後 | 変更クレートのみ再コンパイル | 同左 | **差なし** |

**まとめ**: sccache のローカル効果は**通常の編集サイクルでは現れず，`target/` 消失時のリカバリに限定**される．
ただし DevContainer 環境では `target/` 消失が頻繁に起こるため，この恩恵は実用上大きい．

#### `incremental = false` への変更の影響

現在の設定 (`Cargo.toml` および `scripts/dev-init.sh`) を変更する必要がある:

**Cargo.toml (ワークスペース):**
```toml
[profile.dev]
incremental = false  # sccache との併用のため無効化 (現在は true)
```

**scripts/dev-init.sh:**
```bash
# [profile.dev]
# codegen-units = 16  # Parallel codegen to reduce peak memory (required for 4GB RAM)
# incremental = false
```

---

### レイヤー4: GitHub Actions でのビルドキャッシュ

#### 方針: sccache + GitHub Actions Cache

CI での Rust ビルドを sccache でキャッシュし，wheel ビルド時間を短縮する．

#### 選択肢比較

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **sccache + GHA Cache** | コンパイル単位の細粒度キャッシュ，maturin-action に組み込みサポート | 初回はキャッシュなし | **採用** |
| `Swatinem/rust-cache` | 設定が簡単，`target/` 全体をキャッシュ | sccache より粒度が粗い，maturin-action と組み合わせると二重キャッシュ | 不採用 |
| `actions/cache` で `target/` を手動キャッシュ | 完全にコントロール可能 | キーの設計が複雑，キャッシュの肥大化 | 不採用 |
| キャッシュなし | 設定不要 | 毎回フルビルド(数分) | 不採用 |

**備考**: `PyO3/maturin-action` は `sccache: 'true'` オプションを直接サポートしており，設定が最も簡単．

---

### レイヤー5: リンカーの選択

#### 方針: lld を維持 (現状のまま)

現在 `.cargo/config.toml` で lld を使用しており，これを維持する．
GitHub Actions でも同じく lld を使用する．

#### 背景

GitHub Actions でのビルドは DevContainer (2-core / 4GB RAM) より高スペックなランナーで実行されるため，
リンカーの選択肢が広がる．特に mold はマルチスレッドリンクに優れるため検討した．

ただし，本プロジェクトの出力は `cdylib` (Python 拡張の `.so`) であり，
大規模バイナリ(ゲームエンジン，Web ブラウザ等)と比較してリンク対象が小さい．
そのため，リンカー間の絶対的な時間差は限定的である．

#### 選択肢比較

| リンカー | 速度 (対 GNU ld) | メモリ使用量 | manylinux 互換性 | cdylib/PyO3 互換性 | LTO 対応 | 判定 |
|---|---|---|---|---|---|---|
| **lld (LLVM)** | 5-7x 高速 | 良好 | 良好 (manylinux_2_28 で利用可) | 問題なし | Thin LTO 対応 | **採用 (維持)** |
| mold | 10-25x 高速 | 最小 (lld の約半分) | manylinux_2_28 以降は可，古い manylinux では困難 | 問題なし (想定) | 非対応 (LTO ビルドでは使えない) | 不採用 |
| GNU ld (デフォルト) | 1x (基準) | 大 | 完全互換 (標準装備) | 問題なし | 対応 | 不採用 (遅い) |
| gold (GNU) | 2-3x 高速 | 大 (最も多い) | 標準装備 | 問題なし | 部分的 | 不採用 (非推奨) |

#### 各リンカーの詳細

**lld (LLVM linker)** — 採用

- Rust エコシステムで最も広く使われている代替リンカー
- `rust-lld` として Rust に同梱されつつある ([Rust 1.90.0 (2025年9月) で Linux x86_64 のデフォルトに](https://blog.rust-lang.org/2025/09/01/rust-lld-on-1.90.0-stable/))
- Thin LTO にネイティブ対応 (LLVM ツールチェインの一部であり，LLVM bitcode を直接処理)
- maturin-action の manylinux_2_28 コンテナで利用可能
- 将来的に `.cargo/config.toml` のリンカー設定を削除できる可能性あり (Rust デフォルト化後)
- sccache との組み合わせ: `rustflags` でリンカーを指定するとキャッシュキーに影響する既知の問題あり ([sccache#1755](https://github.com/mozilla/sccache/issues/1755))．ただし sccache は `cdylib` をキャッシュしないため，rlib クレートへの影響のみ

**mold** — 不採用

- 最も高速なリンカーだが，本プロジェクトの cdylib では lld との差が 100ms 未満と推定される
- **LTO に非対応**: LTO 使用時に `lto-wrapper: fatal error` が発生する既知の問題あり ([mold#1226](https://github.com/rui314/mold/issues/1226), [rust#119332](https://github.com/rust-lang/rust/issues/119332))．本プロジェクトの release プロファイルは `lto = "thin"` のため CI wheel ビルドで使えない
- manylinux_2014 コンテナでは古い glibc (2.17) との `.gnu.linkonce` 互換性問題あり．manylinux_2_28 以降なら使用可能
- sccache との組み合わせでキャッシュミスが発生する既知の問題あり ([sccache#1755](https://github.com/mozilla/sccache/issues/1755))
- ライセンス: v2.0 以降 MIT (以前は AGPL)

**GNU ld** — 不採用

- デフォルトリンカーだが lld の 5-7 倍遅い
- メモリ使用量も大きく，2-4GB 環境では不利

**gold (GNU)** — 不採用

- [2025年2月に非推奨化](https://www.phoronix.com/news/GNU-Gold-Linker-Deprecated) (binutils 2.44)
- Rust が `#[used]` 属性で使用する `SHF_GNU_RETAIN` セクションを誤処理する問題あり ([rust#141748](https://github.com/rust-lang/rust/issues/141748))
- メモリ使用量が最も大きい
- 新たなメンテナーが見つからない限り binutils から削除予定

#### DevContainer vs GitHub Actions でリンカーを分ける案

| 環境 | 方針 | 理由 |
|---|---|---|
| DevContainer | lld | メモリ制約あり，LTO 不使用 (dev プロファイル)，lld で十分 |
| GitHub Actions | lld | LTO 使用 (release プロファイル)，mold は LTO 非対応のため不可 |

両環境とも lld で統一できるため，設定の分岐は不要．

---

## 未考慮事項・注意点

### 1. Colab の Python バージョン → 解決済み

CPython 3.11 と 3.12 の両方の wheel をビルドする．
`requires-python = ">=3.11,<3.13"` のため両バージョンのサポートが必要であり，
Colab のデフォルト Python バージョンに依存しない安全な選択とする．
レイヤー1のビルド対象プラットフォーム表に反映済み．

### 2. wheel の extras (torch 等) の取り扱い → 解決済み

wheel には Python 依存パッケージは含まれない(メタデータのみ)．
Colab セットアップの流れにコメントで GPU 依存の追加インストール手順を明記する．
ヘルパースクリプトは作成しない(過剰設計)．
レイヤー2の Colab セットアップコード例にコメントとして反映済み．

### 3. GitHub Releases の latest タグの運用 → 解決済み

`softprops/action-gh-release` で既存 Release を上書きする方式を採用する．
Release ボディにはコミット SHA とビルド日時を記載する．
詳細な変更ログは不要(開発用 wheel であり正式リリースではないため)．

### 4. DevContainer の `dev-init.sh` への sccache 追加 → 解決済み

バイナリ直接ダウンロード方式を採用する(`cargo install` はコンパイルが必要で遅いため)．
以下のサンプルコードの方向性で実装する:

```bash
# バイナリ直接ダウンロードのイメージ
if ! command -v sccache &> /dev/null; then
    echo "Installing sccache..."
    curl -L https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-unknown-linux-musl.tar.gz | tar xz
    mv sccache-*/sccache ~/.cargo/bin/
fi
export RUSTC_WRAPPER="sccache"
```

### 5. DevContainer のディスク容量 → 解決済み

`SCCACHE_CACHE_SIZE="1G"` を設定する．
`debug=1` + `codegen-units=16` 条件での実キャッシュサイズは ~300-500MB であり，1GB で十分なマージンがある．
DevContainer では dev profile のみキャッシュする想定．

| 依存グループ | 推定 rlib サイズ |
|---|---|
| polars (限定 features) | 200-400MB |
| arrow (ipc + pyarrow) | 100-200MB |
| pyo3 + pyo3-polars | 30-50MB |
| その他 (thiserror, walkdir 等) | 10-20MB |
| maou_io + maou_index | 5-10MB |
| **合計** | **~350-680MB** |

なお，dev profile では `debug = 1` (line tables only) のためデバッグ情報が少なく，実際の rlib サイズは上記見積もりの下限寄りになると想定される．

```bash
export SCCACHE_CACHE_SIZE="1G"  # キャッシュ上限を 1GB に制限
```

### 6. sccache のパス依存性

sccache のキャッシュキーにはソースファイルの**絶対パス**が含まれる．
DevContainer のワークスペースパスが変わるとキャッシュがヒットしない可能性がある．
通常の Codespaces 利用では問題にならないが，異なるパスにクローンした場合は注意．
回避策として `--base-dir` オプション(sccache 0.8+)による絶対パスの正規化が可能．

### 7. セキュリティ → 対応方針確定

- GitHub Releases の wheel は公開リポジトリの場合，誰でもダウンロード可能
- wheel にはコンパイル済みバイナリが含まれるため，CI の整合性が重要
- `softprops/action-gh-release` のバージョンをピン留めする(実装タスクに含む)

---

## 将来の最適化候補

### CI wheel の .so を DevContainer で活用

DevContainer 初期化時に CI でビルドした wheel から `.so` ファイルを取得し，
Rust ビルドを完全にスキップする案．以下の理由で今回のスコープからは外し，
将来のフェーズで検討する:

- maturin editable install と wheel `.so` の共存の検証が必要
- release build の `.so` を dev 環境で使う際のデバッグ体験への影響
- sccache で十分な改善が得られる見込み

---

## 実装タスク一覧

### Phase 1: GitHub Actions wheel ビルド (UC2 向け)

1. `.github/workflows/build-wheel.yml` を作成
   - トリガー: `push` (main ブランチ) + `workflow_dispatch`
   - `PyO3/maturin-action` で wheel ビルド (`--release`)
   - ビルド対象: CPython 3.11 + 3.12，`manylinux_2_28` x86_64
   - `sccache: 'true'` で CI キャッシュ有効化
   - `softprops/action-gh-release` で latest Release に配置(バージョンをピン留め)
   - Release ボディにコミット SHA とビルド日時を記載
2. 手動トリガーで動作確認
3. Colab でのインストール手順を検証

### Phase 2: sccache のローカル導入 (UC1 向け)

4. `scripts/dev-init.sh` に sccache バイナリのインストールを追加
5. `RUSTC_WRAPPER=sccache` の設定を追加
6. `SCCACHE_CACHE_SIZE="1G"` の設定を追加
7. `Cargo.toml` の `[profile.dev]` で `incremental = false` に変更
8. `scripts/dev-init.sh` の cargo ユーザー設定で `codegen-units = 16` + `incremental = false` に変更
9. `.devcontainer/devcontainer.json` の `mounts` 設定で `~/.cargo/registry/` を DevContainer の volume で永続化し，`cargo fetch` のキャッシュを有効化
10. DevContainer 再作成でのビルド時間を計測・検証

### Phase 3: ドキュメント更新

11. `docs/rust-backend.md` に Colab でのプリビルト wheel インストール手順を追加
12. `docs/rust-backend.md` に sccache のセットアップ手順を追加

---

## 参考資料

- [PyO3/maturin-action](https://github.com/PyO3/maturin-action) - GitHub Action for maturin
- [mozilla/sccache](https://github.com/mozilla/sccache) - Shared Compilation Cache
- [mozilla-actions/sccache-action](https://github.com/mozilla-actions/sccache-action) - sccache GitHub Action
- [sccache: Known Caveats - Incremental Compilation](https://github.com/mozilla/sccache#known-caveats)
- [sccache: Rust and incremental compilation (Issue #236)](https://github.com/mozilla/sccache/issues/236)
- [Benchmarking Rust compilation with sccache](https://neosmart.net/blog/benchmarking-rust-compilation-speedups-and-slowdowns-from-sccache-and-zthreads/)
- [Fast Rust Builds with sccache and GitHub Actions](https://depot.dev/blog/sccache-in-github-actions)
- [softprops/action-gh-release](https://github.com/softprops/action-gh-release) - GitHub Release Action
- [Maturin Distribution Guide](https://www.maturin.rs/distribution.html)
- [uv: Settings Reference](https://docs.astral.sh/uv/reference/settings/)
- [Rust 1.90.0: Faster linking with rust-lld on Linux](https://blog.rust-lang.org/2025/09/01/rust-lld-on-1.90.0-stable/) - lld のデフォルト化
- [Enabling rust-lld on nightly](https://blog.rust-lang.org/2024/05/17/enabling-rust-lld-on-linux/) - lld nightly 検証結果
- [mold GitHub Repository](https://github.com/rui314/mold) - mold リンカー
- [GNU Gold Linker Is Deprecated (Phoronix)](https://www.phoronix.com/news/GNU-Gold-Linker-Deprecated) - gold 非推奨化
- [mold fails to link with LTO (mold#1226)](https://github.com/rui314/mold/issues/1226) - mold LTO 問題
- [sccache cache miss with mold (sccache#1755)](https://github.com/mozilla/sccache/issues/1755) - リンカーとキャッシュキーの問題
