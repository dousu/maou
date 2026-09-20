---
status: applied          # pending | approved | applied | rejected
applied_in: a23bcba8
date: 2026-09-20
target: [docs/colab-cli-notes.md, scripts/colab_adopt_session.py, .claude/skills/colab-operator/SKILL.md, CLAUDE.md]
risk: low
reversibility: trivial
---

# `docs/colab-cli-notes.md` を maou 固有の Colab 運用規約にする

## Trigger

ユーザが `colab` CLI (googlecolab/google-colab-cli) の汎用メモを
`docs/colab-cli-notes.md` に置き (未コミット)，このプロジェクト向けに
ブラッシュアップするよう明示的に依頼した (2026-09-20)．追加すべき規約として
ユーザが挙げたもの:

1. Colab インスタンスの起動停止オーバーヘッドを減らすため計画的に使う
2. Google Drive への書き込みは Colab で Drive をマウントして VM 内で行う
3. Drive は `MyDrive/shogi` 内だけを操作する
4. まず VM ローカルに出力し，作業のまとまりで Drive にコピーする
5. レイアウト: hcpe → `shogi/hcpe`，preprocess → `shogi/preprocess`，
   search_values → `shogi/search_values`，models → `shogi/maou_test/models`，
   TensorBoard logs → `shogi/maou_test/logs`
6. 再利用しうるデータ (算出に時間のかかるものを含む) は Drive に残す
7. VM で実行するスクリプトは `scripts/` に置いて再現性を持たせる
8. 時間のかかる検証は，ユーザ側でブラウザからセッションを作ってから CLI で接続し，
   投入したら `/checkpoint-context` する (セッション維持はユーザの担当)

## Part A — `docs/colab-cli-notes.md` の書き換え (依頼により適用済)

ユーザの明示的な依頼を承認とみなし，同セッションで本文を書き換えた．
コミット時に本提案の `applied_in` を埋める．

書き換えの要点:

- **CLI 汎用部 (§0–§5) の訂正・補強** (PyPI 0.6.0 / GitHub v0.7.1 を実機で検証):
  - **PyPI 0.6.0 は素の状態で壊れている** (upstream `jupyter-kernel-client` 1.0.2 を拾い
    `exec` が `AttributeError`)．導入手順を GitHub v0.7.1 一本にした．
  - `--auth` の既定は `oauth2` (README は `adc` と書くがコードの既定値は
    `AuthProvider.OAUTH2`)．元メモの「`--auth=oauth2` を毎回付ける」は逆で，
    ADC を使う場合に `--auth=adc` を毎回付ける．refresh token は
    `~/.config/colab-cli/token.json` にキャッシュされる．
  - `colab drivemount` は OAuth 同意待ち (600 秒) の人間必須コマンド．マウントは
    FUSE で VM 全体に効く．
  - keep-alive デーモンを再生成するコマンドは無い．`sessions.json` は Codespaces の
    停止→再開では残るが container rebuild で消える (token も)．消えても
    `colab_adopt_session.py` で再登録できる (デーモンは戻らない)．
  - `exec` が例外でも exit 0 になること，未知 GPU 名が A100 にフォールバックすることを
    ソースで裏取り．
  - 句読点を `，` `．` 半角括弧に統一．
- **maou 固有の運用規約 (§6–§10, MUST) を新設**:
  - §6 wheel 導入: VM でビルドせず Release `latest` (main push ごとに上書き) を使う．
    extras の使い分け，TensorRT の導入手順は `docs/design/position-search/benchmarking.md`
    を正として複製しない．
  - §7 Drive 規約: 上記 2–6 を規則化．ローカル側を `/content/shogi/` として Drive と
    同レイアウトにする (コピーが 1 行になる)．
  - §8 計画的利用: 上記 1 を「`colab new` 前に全ステップを書き出す」手順にした．
  - §9 `scripts/`: 上記 7．argparse・flush・番兵行・worklog 記録項目．
  - §10 ハンドオフ: 上記 8 をそのままの向きで規則化．CLI に attach コマンドは無いが，
    `colab sessions` の一覧に runtime proxy の token / url が含まれ，`colab new` が保存する
    のと同じ 3 つなので，**`scripts/colab_adopt_session.py` (新規) でローカル登録すれば
    ブラウザ起動のランタイムに `exec` できる** (CPU ランタイムで実機確認)．登録した
    セッションを `colab stop` しない (VM ごと消える) こと，投入直後の `/checkpoint-context`
    と記録項目，Codespaces 復帰・container rebuild からの再登録手順を明記．
    `colab url` でブラウザを attach する向きは，そのタブが keep-alive を担うか未検証
    なので代替案にせず，「長時間ジョブなら `stop` してブラウザで作り直す」とした．

追加提案としてユーザの列挙に無い規則を入れ，2026-09-20 にユーザが判断した:

- §7.2 `shogi/` 直下に新しいトップレベルフォルダを増やすときはユーザ確認 — 採用
- §7.4 / §7.5 一時物の扱い — **修正して採用**: Drive に残す基準は「再生成に 10 分以上」．
  TensorRT engine cache は数分で再生成できるので Drive に置かない (当初案の
  「`maou_test/` 配下に置く」は却下)
- §7.5 Drive 上の削除・上書きはユーザ確認 (ローカルは自由) — 採用
- §8 計画外の途中 `stop` はユーザ確認 — 採用

同時に決まった運用値:

- logs の置き場は `shogi/maou_test/logs` (models と同階層)
- VM ローカルは `/content/shogi/` で Drive と同レイアウト (一旦この規則で運用)
- サブフォルダ命名は既存 Drive の実例に合わせて `<種別>_<YYYYMMDD>`
  (`hcpe/hcpe_20260805/` 等．同日複数は `_<HHMM>`)．由来・条件は worklog へ
- GPU: 本番学習は G4，ONNX 推論と学習のテスト段階は L4，T4 は使わない
- 登録したセッションは `colab stop` しない (ユーザがブラウザで削除)
- `scripts/colab_adopt_session.py` は手動の疎通確認のみでテスト無し (`colab_cli` が
  `uv tool` の仮想環境にしか無い)
- 公式 skill `.claude/skills/colab-operator/SKILL.md` を `colab skill` から生成して導入．
  公式 skill の記述 (既定 `adc`・`-c` 必須・PyPI 導入・T4 フォールバック) は検証結果と
  食い違うため，frontmatter 直後に "maou project overrides" ブロックを足して本メモを優先

## Part B — `CLAUDE.md` への参照追加 (2026-09-20 ユーザ承認・適用)

このメモは自動では読み込まれないので，Colab 作業時に §6–§10 が binding になるよう
`CLAUDE.md` に 2 箇所追加する．

**1. `## Critical Rules (MUST)` の末尾に節を追加**:

```
### Google Colab (GPU 検証)
- MUST follow [docs/colab-cli-notes.md](docs/colab-cli-notes.md) §6–§10 when using
  the `colab` CLI: wheel from Release `latest`, Drive writes only via mount inside
  the VM and only under `MyDrive/shogi`, local-first output, `scripts/` for
  reproducibility, `/checkpoint-context` right after launching a long job
```

**2. `## Documentation Links` の表に行を追加** (`| CLI Commands |` 行の直後):

```
| Colab CLI 運用 | [docs/colab-cli-notes.md](docs/colab-cli-notes.md) |
```

## Why

- 上記 8 項目は「会話で決めた運用」であり，`docs/` に置かない限り次のセッションで
  再発明される (memory-architecture の趣旨)．
- 元メモの認証記述は逆だったため，そのまま採用すると毎回 `--auth=oauth2` を付ける
  無駄と，ADC 利用時のフラグ落ちによる再認証プロンプトが起きる．
- 「ブラウザで作ってから CLI で接続」は CLI の公式コマンドでは不可能だが，登録スクリプト
  1 本で成立する．スクリプトを `scripts/` に置くことで §9 の規則とも整合する．

## Risk

low．新規ドキュメント + 開発マシン側の小スクリプト 1 本 (`src/` には触れないので
版数 bump 不要) + CLAUDE.md への参照 2 行．
