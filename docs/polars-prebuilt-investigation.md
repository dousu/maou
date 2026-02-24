# Rust ビルド高速化 設計ドキュメント

## 目的

Rust 拡張モジュール(`maou._rust`)のビルド時間を短縮する．

## ユースケース

### UC1: Codespaces DevContainer での開発ビルド

- **環境**: 2-core / 4GB RAM の低スペック環境
- **ビルド頻度**: `cargo clean` からのフルビルドは初回1回のみ．以降はインクリメンタルビルド
- **優先度**: インクリメンタルビルドの高速化 > フルビルドの高速化
- **最適化レベル**: dev プロファイル(最適化不要)
- **備考**: 開発中は editable install が必須(`src/maou/` の Python コード変更が即座に反映される必要あり)

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
        ┌───────┴──────────┐
        ▼                  ▼
┌───────────────┐  ┌────────────────────┐
│ Colab (UC2)   │  │ DevContainer (UC1) │
│ wheel install │  │ sccache (local)    │
│ editable 不要 │  │ + maturin develop  │
│ release 最適化│  │ editable install   │
└───────────────┘  └────────────────────┘
```

| コンポーネント | 対象ユースケース | 手法 |
|---|---|---|
| GitHub Actions wheel ビルド | UC2 (Colab) | `maturin build --release` → GitHub Releases |
| sccache (GitHub Actions Cache) | CI の Rust ビルド高速化 | `Swatinem/rust-cache` or `sccache` + GHA Cache |
| sccache (ローカルディスク) | UC1 (DevContainer) | `RUSTC_WRAPPER=sccache` + `~/.cache/sccache/` |

---

## 実装レイヤーごとの設計と選択肢比較

### レイヤー1: CI での wheel ビルド

#### 決定: GitHub Actions + `maturin-action` + GitHub Releases

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
| `manylinux_2_28` x86_64 + CPython 3.11 | Python 3.11 環境 | 要検討(Colab のデフォルト Python バージョン次第) |
| macOS arm64 | ローカル開発 (Apple Silicon) | 当面不要(ローカルは sccache で対応) |
| Windows x86_64 | Windows 開発者 | 当面不要 |

**考慮事項**: `requires-python = ">=3.11,<3.13"` のため，Python 3.11 用 wheel も必要になる可能性がある．Colab のデフォルト Python バージョンを確認して判断する．

#### wheel の内容物

maturin の仕様上，1つの wheel に以下がすべて含まれる:

- Python コード (`src/maou/` 全体)
- コンパイル済み Rust 拡張 (`_rust.so`)
- CLI エントリポイント (`maou` コマンド)

Colab では wheel をそのままインストールして使うため，これで問題ない．
DevContainer では wheel は使わず sccache + `maturin develop` (editable install) を使う．

---

### レイヤー2: Colab 環境でのインストール

#### 決定: `gh release download` + `uv pip install`

Colab 環境では Rust ツールチェインをインストールせず，プリビルト wheel をインストールする．

#### 選択肢比較

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **`uv pip install` + wheel URL** | シンプル，1コマンド | wheel URL の管理 | **採用** |
| `uv sync` + `[tool.uv.sources]` で URL 指定 | pyproject.toml で宣言的 | main の pyproject.toml を書き換える必要あり，editable install との競合 | 不採用 |
| `uv sync --no-install-project` + `uv pip install` | 依存を uv sync で，maou を wheel で | 2段階コマンド | 補助的に採用 |
| `[[tool.uv.index]]` flat index | uv のネイティブ機能 | GitHub Releases の HTML がflat index として適切でない場合がある | 不採用 |

#### Colab セットアップの流れ

```python
# Colab セル
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

#### 決定: sccache + ローカルディスクキャッシュ

`RUSTC_WRAPPER=sccache` を設定し，コンパイル結果を `~/.cache/sccache/` にキャッシュする．

#### 選択肢比較

| 選択肢 | メリット | デメリット | 判定 |
|---|---|---|---|
| **sccache (ローカルディスク)** | 設定最小限，外部依存なし，venv/target 再作成後も有効 | 開発者間で共有不可(個人キャッシュ) | **採用** |
| sccache (S3 バックエンド) | 開発者間でキャッシュ共有可能 | AWS 認証が全員に必要，コスト発生 | 不採用(過剰) |
| sccache (GitHub Actions Cache) | 無料 | ローカルからアクセス不可(CI 内限定) | CI のみ採用 |
| `cache-keys` (uv 設定) | pyproject.toml だけで設定可能 | uv のキャッシュが破損するとフルリビルド，sccache ほど粒度が細かくない | 併用を検討 |
| Docker イメージに `.so` を事前ビルド | DevContainer 起動即使える | Docker イメージの管理コスト，更新の手間 | 不採用(運用負荷) |
| cargo のインクリメンタルビルドのみ | 追加設定不要 | `target/` 削除やvenv再作成で無効化 | 現状のまま維持 |

#### sccache がローカルで効果を発揮するシーン

| シーン | cargo incremental のみ | sccache あり |
|---|---|---|
| 通常の `maturin develop` | キャッシュヒット (速い) | キャッシュヒット (速い) |
| `target/` が消えた後 | **フルリビルド** | **キャッシュヒット** |
| `.venv` 再作成後の `uv sync` | **フルリビルド** | **キャッシュヒット** |
| ブランチ切替後 | 部分リビルド | より多くキャッシュヒット |
| `cargo clean` 後 | **フルリビルド** | **キャッシュヒット** |

最も効果が大きいのは **`target/` や `.venv` が消えた後の再ビルド** である．
DevContainer の再作成時やディスク容量の都合で `cargo clean` した際にフルリビルドを回避できる．

#### sccache のインクリメンタルビルドとの関係

**重要な注意点**: sccache は `incremental = true` との併用をサポートしていない．

- `incremental = true` の場合，sccache はキャッシュを**スキップ**する(コンパイル結果をキャッシュしない)
- つまり，sccache の効果を最大化するには `incremental = false` にする必要がある

しかし，UC1 (DevContainer) では **インクリメンタルビルドの高速化が最優先** であるため:

| 設定 | インクリメンタルビルド | クリーンビルド | 判定 |
|---|---|---|---|
| `incremental = true` (sccache なし) | 速い | 遅い | 現状 |
| `incremental = false` + sccache | 遅い(毎回フルコンパイル) | 速い(キャッシュヒット) | UC1 に不向き |
| **`incremental = true` + sccache** | **速い(cargo キャッシュ)** | **速い(sccache キャッシュ)** | **推奨** |

`incremental = true` + sccache の組み合わせでは:
- **通常のインクリメンタルビルド**: cargo 自体のインクリメンタルコンパイルが担当(sccache は関与しない)
- **クリーンビルド時**: sccache がキャッシュを提供(**ただし，incremental = true のコンパイル結果はキャッシュされていないため，効果が限定的**)

**結論**: sccache のローカルディスクキャッシュは `incremental = true` 環境では **効果が限定的** である．
主な恩恵は CI (GitHub Actions) 側で `incremental = false` + sccache を使うケースに限られる．
DevContainer では現状の cargo incremental compilation が最も効果的であり，sccache の追加効果は小さい．

#### 代替検討: `cache-keys` による uv キャッシュ最適化

sccache のローカル効果が限定的であるため，`cache-keys` の導入も検討する:

```toml
[tool.uv]
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "rust/**/*.rs" },
]
```

これにより `uv sync` は Rust ソースに変更がない場合にリビルドをスキップする．
ただし `uv sync` 経由での再ビルドのみに効果があり，`maturin develop` を直接実行する場合は関係ない．

---

### レイヤー4: GitHub Actions でのビルドキャッシュ

#### 決定: sccache + GitHub Actions Cache

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

## 未考慮事項・注意点

### 1. Colab の Python バージョン

Colab のデフォルト Python が 3.11 か 3.12 かで，ビルドする wheel の ABI タグが変わる．
両バージョン用の wheel をビルドするか，Colab 側で Python バージョンを指定するか決める必要がある．

### 2. wheel の extras (torch 等) の取り扱い

wheel には Python 依存パッケージは含まれない(メタデータのみ)．
Colab で `uv pip install maou-*.whl` した後，`torch[cuda]` 等は別途インストールが必要．
インストール手順のドキュメントまたはヘルパースクリプトが必要．

### 3. GitHub Releases の latest タグの運用

`latest` タグを上書きする場合，`softprops/action-gh-release` の設定で以下を考慮:
- 既存の Release を更新するか，削除して再作成するか
- Release のボディ(変更ログ)をどう管理するか

### 4. DevContainer の `dev-init.sh` への sccache 追加

sccache のインストールを `dev-init.sh` に含める場合:
- インストール方法: `cargo install sccache` はコンパイルが必要で時間がかかる → `cargo-binstall` またはバイナリ直接ダウンロードが望ましい
- 環境変数の設定タイミング: `~/.cargo/config.toml` に `rustc-wrapper` を設定するか，`~/.bashrc` に `export RUSTC_WRAPPER=sccache` を追加するか

### 5. DevContainer のディスク容量

sccache のローカルキャッシュはデフォルトで最大 10GB を使用する．
DevContainer のディスク容量によってはキャッシュサイズの上限を設定する必要がある:

```bash
export SCCACHE_CACHE_SIZE="2G"  # キャッシュ上限を 2GB に制限
```

### 6. `dev-init.sh` の変更範囲

現在の `dev-init.sh` は Rust ツールチェインのインストールから `uv sync` まで一貫して行う．
sccache の導入は既存フローへの追加であり，破壊的変更はない:

```bash
# 追加部分のイメージ
if ! command -v sccache &> /dev/null; then
    echo "Installing sccache..."
    # バイナリを直接ダウンロード (cargo install より速い)
    curl -L https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-unknown-linux-musl.tar.gz | tar xz
    mv sccache-*/sccache ~/.cargo/bin/
fi
export RUSTC_WRAPPER="sccache"
```

### 7. `incremental = true` と sccache の競合 (再掲)

レイヤー3で述べた通り，`incremental = true` 環境では sccache のキャッシュ効果が限定的である．
CI (`incremental = false` 相当のクリーンビルド) では効果が大きいが，ローカル開発では cargo 自体のインクリメンタルコンパイルが主力となる．
sccache をローカルに入れるコストは小さいため導入自体はするが，**過度な期待は禁物**．

### 8. セキュリティ

- GitHub Releases の wheel は公開リポジトリの場合，誰でもダウンロード可能
- wheel にはコンパイル済みバイナリが含まれるため，CI の整合性が重要
- `softprops/action-gh-release` のバージョンをピン留めすること

---

## 実装タスク一覧

### Phase 1: GitHub Actions wheel ビルド

1. `.github/workflows/build-wheel.yml` を作成
   - トリガー: `push` (main ブランチ) + `workflow_dispatch`
   - `PyO3/maturin-action` で wheel ビルド
   - `sccache: 'true'` で CI キャッシュ有効化
   - `softprops/action-gh-release` で latest Release に配置
2. 手動トリガーで動作確認
3. Colab でのインストール手順を検証

### Phase 2: sccache のローカル導入

4. `scripts/dev-init.sh` に sccache インストールを追加
5. `RUSTC_WRAPPER=sccache` の設定
6. DevContainer 再作成でのビルド時間を計測・検証

### Phase 3: ドキュメント更新

7. `docs/rust-backend.md` に Colab でのプリビルト wheel インストール手順を追加
8. `docs/rust-backend.md` に sccache のセットアップ手順を追加

---

## 参考資料

- [PyO3/maturin-action](https://github.com/PyO3/maturin-action) - GitHub Action for maturin
- [mozilla/sccache](https://github.com/mozilla/sccache) - Shared Compilation Cache
- [mozilla-actions/sccache-action](https://github.com/mozilla-actions/sccache-action) - sccache GitHub Action
- [sccache: Known Caveats - Incremental Compilation](https://github.com/mozilla/sccache#known-caveats)
- [softprops/action-gh-release](https://github.com/softprops/action-gh-release) - GitHub Release Action
- [Maturin Distribution Guide](https://www.maturin.rs/distribution.html)
- [uv: Settings Reference](https://docs.astral.sh/uv/reference/settings/)
