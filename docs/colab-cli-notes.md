# colab CLI 運用メモ (Claude Code による maou の自律 GPU 検証)

対象: Google 公式 `colab` CLI = [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
検証日: 2026-09-19 (CLI 挙動)，2026-09-20 (PyPI 0.6.0 / GitHub v0.7.1 を実機で検証，
maou 固有の運用規約を追加)

このメモは 2 部構成:

- **§0–§5: CLI の落とし穴** — 公式同梱 skill (`colab-operator`) に書かれていない／誤解しやすい
  挙動だけをまとめる．基本的なコマンド体系は公式 skill 側に任せる．
- **§6–§10: maou 固有の運用規約 (MUST)** — Colab インスタンス，Google Drive，`scripts/`，
  長時間検証のハンドオフについて，このリポジトリで Claude Code が守るべき規約．
  §11 に判断フローの要約を置く．

---

## 0. 人間が 1 回だけやるセットアップ

### (a) CLI インストール (Python >= 3.12 必須 / Linux・macOS のみ)

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli@v0.7.1   # この DevContainer に導入済
colab version
```

> **PyPI 版 (0.6.0 = 2026-06-16) は使わない．** CLI は Google のフォーク
> `github.com/googlecolab/jupyter-kernel-client` を必要とするが，PyPI 版はピン無しの
> `jupyter-kernel-client` を宣言しているため upstream 1.0.2 が入り，`colab exec` が
> `AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'` で
> **素の状態で壊れる** (2026-09-20 実機確認)．GitHub 版は `[tool.uv.sources]` でフォークを
> 指定しているので `uv tool install git+...` で正しく解決される．
> GitHub タグは v0.7.1 = 2026-09-15．0.7.x だけにあるのは `ssh`，`--env KEY=VALUE`，`--high-mem`．

### (b) 公式 skill を Claude Code に入れる

```bash
mkdir -p .claude/skills/colab-operator && colab skill > .claude/skills/colab-operator/SKILL.md
```

リポジトリの `.claude/skills/colab-operator/SKILL.md` は上記の出力に **"maou project overrides"**
ブロック (frontmatter 直後) を足したもの．公式 skill は「既定は `adc`」「oauth2 には `-c` が要る」
「PyPI で導入」「T4 にフォールバック」と書いているが，いずれも本メモの検証結果と食い違うので，
skill と本メモが衝突したら本メモが勝つ．CLI を更新して skill を再生成したら，このブロックを
付け直す．

### (c) 認証 (ここが唯一の「人間必須」ポイント)

**既定は `oauth2`** (README には `adc` と書いてあるが，コードの既定値は `AuthProvider.OAUTH2`．
v0.7.1 で確認)．Codespaces のようなブラウザなし環境では次のどちらか．

```bash
# 案1 (推奨): oauth2 — gcloud 不要．URL が出るのでブラウザで開き，コードを貼り戻す
colab sessions

# 案2: ADC (gcloud がある場合)．以降 **毎回** `--auth=adc` を付ける必要がある
gcloud auth application-default login --no-launch-browser \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions
```

- oauth2 はローカルコールバック不要の**コピペ方式**なので Codespaces でそのまま通る．
  OAuth クライアント設定は CLI に同梱済 (`-c` は不要)．refresh token は
  `~/.config/colab-cli/token.json` にキャッシュされ，**初回ログイン 1 回で以後フラグ不要**．
- `--auth` の選択は設定ファイルに保存されない．ADC を選ぶなら毎回 `--auth=adc`．
- 認可コードの貼り戻しは `input()` 待ちになるので，Claude Code の `!` プレフィックスでは
  `Aborted.` になる．**VS Code の通常ターミナルで実行する**．
- 確認: `colab sessions` (読み取りのみ)，`colab whoami` (隠しコマンド．email・スコープ・有効期限)．
- `colab auth` は **VM 側の GCP 認証**であり CLI 認証とは無関係．401/403 の対処に使わない．

**Claude Code へ**: 上記が未完了なら自力で突破しようとせず，人間に (c) の実行を依頼して停止すること．

### (d) Google Drive のマウント (セッションごとに人間必須)

`colab drivemount -s <name>` は VM 上で `google.colab.drive.mount('/content/drive')` を走らせ，
**ブラウザでの OAuth 同意を待つ** (600 秒)．`auth` と同じく人間必須 (§5)．
マウントは FUSE で VM 全体に効くので，一度マウントすれば CLI の `exec` からも
`/content/drive/MyDrive/shogi` が見える (§7)．

- 人間はセッション登録後に通常ターミナルで `colab drivemount -s <name>` を実行する．
  §10 のようにブラウザで起動したランタイムなら，そのノートブックのセルで
  `from google.colab import drive; drive.mount('/content/drive')` を実行するほうが早い
  (Colab の UI が同意ダイアログを出す)．
- Claude Code は `colab exec -s <name>` で `os.path.ismount('/content/drive')` または
  `ls /content/drive/MyDrive/shogi` を確認してから Drive を扱う．未マウントなら人間に依頼して停止．

---

## 1. 最重要: `colab exec` / `colab run` の `--timeout` は既定 30 秒

公式 README・skill に記載がないが，ソース上の既定値は `30.0` 秒
(`src/colab_cli/commands/execution.py`, `run.py`．0.6.0 の `--help` にも `[default: 30.0]` と出る)．
学習や重いベンチはこれで確実に落ちる．

```bash
colab run --gpu L4 --timeout 3600 verify.py
colab exec -s s1 --timeout 3600 -f verify.py
```

- タイムアウトは「無出力が続いた時間」に近い挙動 (kernel から出力イベントが届き続ける限り
  延命される実装)．→ **スクリプト側で定期的に `print(..., flush=True)` を出す**とタイムアウト
  耐性が上がる．
- それでも数時間規模のジョブを `exec` で同期的に回すのは不安定．§4 のバックグラウンド方式を使う．

## 2. `colab exec` は Python 例外でも exit 0

`exec` は kernel のエラー出力を表示するだけで，終了コードには反映しない (成功と区別できない)．
一方 `colab run` は CPython 準拠で伝播する: 未捕捉例外 → 1，`sys.exit(N)` → N，`sys.exit(0)` → 0．

> **結論: 合否判定が必要な自動検証は `colab run` を第一選択にする．**
> `exec` を使う場合は `print("VERIFY_OK")` のような番兵を出力し，標準出力を grep して判定する．

`colab run` はさらに，

- VM を新規確保 → 実行 → **自動で破棄** (`--keep` で保持)．消し忘れによる課金事故が起きない．
- `[colab] ...` の進捗は **stderr**，スクリプト出力は **stdout** に分離される
  (`colab run job.py > out.txt` で純粋な結果だけ取れる)．
- スクリプトが存在しない場合は VM を確保する前に非ゼロ終了する．
- ただし **毎回 VM 確保 + 環境構築をやり直す**ので，maou の wheel 導入 (§6) を伴う検証は
  `run` を連発せず，§8 の計画に従って `new` + `exec` で 1 セッションにまとめる．

## 3. keep-alive デーモンは「開発マシン側」に常駐する

`colab new` は Codespaces 上にデタッチしたバックグラウンドプロセスを立て，60 秒ごとに ping して
Colab 標準の約 90 分アイドル回収を防いでいる (`docs/01_session_management.md`)．

- **Codespaces が停止・再起動すると ping が止まり，VM は放置後に回収される．**
  デーモンを再生成するコマンドは無い (`new` / `run` 時にしか spawn されない)．
- デーモンは **24 時間で自動終了** (ゾンビ防止の安全弁)．24 時間超のジョブは成立しない前提で設計する．
- ローカル状態 (`~/.config/colab-cli/sessions.json`: session 名・endpoint・token) は Codespaces の
  **停止→再開では残る**ので，VM が生きていれば `colab exec -s <name>` は再開後も通る．
  **container rebuild では消える** (認証 token も)．VM が生きていれば §0(c) の再認証のあと
  `scripts/colab_adopt_session.py` で登録し直せる (§10) が，keep-alive デーモンは戻らないので，
  `new` で作ったセッションのジョブ中に rebuild しない．
- §10 の手順で登録した (ブラウザ起動の) セッションには**そもそもデーモンが無い**．
  keep-alive はブラウザのタブが担う．
- checkpoint は VM ローカルに書き，作業のまとまりごとに Drive へ退避する (§7)．
- Codespaces に依存せず VM を維持したい場合の手順は §10．

## 4. 長時間ジョブの推奨パターン (公式未文書・こちらの推奨)

`exec` は kernel と同期するためタイムアウトを持つ．**VM 上で detach させれば `exec` は即座に返る**ので，
以降はポーリングで状態を見る．

```bash
colab new -s gpu --gpu L4

# 投入 (即 return する)
cat <<'EOF' | colab exec -s gpu
import subprocess
subprocess.Popen("nohup python /content/job.py > /content/job.log 2>&1 &", shell=True)
EOF

# ポーリング
echo "print(open('/content/job.log').read()[-2000:])" | colab exec -s gpu
colab status -s gpu

# 回収と後始末 (大きな成果物は §7 のとおり Drive へ．download は小物のみ)
colab download -s gpu /content/summary.json ./summary.json
colab stop -s gpu
```

- `colab exec` 間で **kernel の状態 (変数・import) は保持される** (同じ kernel に再接続する仕様)．
  リセットは `colab restart-kernel` か `colab stop`．
- 作業ディレクトリは常に `/content`．ファイル操作は絶対パスで書く．
- 終了判定は log の番兵行 (`VERIFY_OK` / `JOB_DONE exit=0`) か，ジョブ側で書く
  `/content/job.exit` のような終了コードファイルで行う (§2 の理由で `exec` の exit code は使えない)．

## 5. その他の実務メモ

- **GPU 指定**: `T4` / `L4` / `G4` / `H100` / `A100` (TPU は `v5e1` / `v6e1`)．
  **未知の値は黙って A100 にフォールバック**する (`mapping.get(gpu.lower(), Accelerator.A100)`)
  のでタイポ注意．確保失敗 (400/412) はアカウントの枠不足なので，§8 の目安に従って 1 段下
  (G4 → L4) に落とす．T4 は古いので使わない．CPU は wheel の疎通確認にしか使えない．
- **秘密情報**: `--env KEY=VALUE` (`exec` / `run` 共通)．値はセッション生存中のみ有効．
  スクリプトや worklog に書かない．
- **ノートブック**: `colab exec -f nb.ipynb` は全コードセルを実行し，`nb_output.ipynb` を入力の隣に書き出す．
- **エージェントが呼んではいけないコマンド**: `repl` / `console` / `auth` / `drivemount` は TTY 前提．
  `repl` / `console` はパイプ入力なら使えるが，`auth` / `drivemount` は人間必須．
- **ファイル転送**: `colab upload` / `download` はスクリプトや小さな結果ファイル用．
  データ・モデルの受け渡しは Drive 経由 (§7)．
- **並列実行の分離**: `colab --config /tmp/agent.json new -s job` でセッション状態ファイルを分けると，
  人間の手元セッションと衝突しない．この DevContainer では人間も同じ `~/.config/colab-cli/` を
  使う前提なので，通常は分けない (分けると §10 で登録したセッションが別ファイルに入る)．
- **`colab sessions` の `[?]`**: ローカル未登録のランタイム (ブラウザで起動したもの)．2 列目は
  セッション名ではなく backend の **endpoint ID**．`-s` には渡せないが，
  `scripts/colab_adopt_session.py` で名前を付けて登録すれば `exec` / `status` / `upload` が
  使える (§10)．登録したものへの `colab stop` は VM を unassign する (ブラウザ側も消える) ので，
  ローカル登録だけ外すときは `--forget`．
- **失敗時の一次情報**: `colab log -s <name> -n 20` (keep-alive エラーの生レスポンスまで出る)，
  `colab sessions`，`colab status`．§10 で登録したセッションには keep-alive ログは無い．
- **404 が出たら**: バックエンド側で VM が消えている．ローカル状態は自動で掃除されるので
  `colab new` (または §10 の再登録) からやり直す．Drive に退避済みの中間物から再開する (§7)．
  **401** は runtime proxy token の失効の可能性もある — `colab sessions` にまだ見えるなら
  `scripts/colab_adopt_session.py <name> --force` で token を取り直してから判断する．
- **コスト**: `colab new` した VM は `stop` するまで課金対象．エージェントは必ず `stop`
  (もしくは `colab run`) で終わること．例外は §10 の「ユーザが保持するセッション」．

---

## 6. maou 固有: VM への環境構築は Release `latest` の wheel で行う

VM 上で Rust をビルドしない．`main` への push ごとに GitHub Actions (`build-wheel.yml`) が
`manylinux_2_28` x86_64 の CPython 3.12 / 3.13 用 wheel を Release **`latest`** に上書き公開する．
アセット名は `maou-<version>-cp312-cp312-manylinux_2_28_x86_64.whl` (cp313 も同様)．

- 検証したいコードは **先に `main` にマージし，Build Wheel が完了してから** VM を確保する
  (§8 の「計画」の一部)．`latest` は最後に走った build で上書きされるので，実験中は他の
  `main` push に注意する．
- 用途別の extras: 学習 (`learn-model`) は `maou[cuda]`，ONNX GPU 推論 (`search` / `selfplay` /
  `utility search-values` / `floodgate` / `usi`) は `maou[tensorrt-infer]` + `ldconfig` 手順，
  Drive 以外のクラウド I/O は `[gcp]` / `[aws]`．**`tensorrt-infer` の導入セル (wheel URL 解決 +
  `/etc/ld.so.conf.d/maou.conf` + `ldconfig`) は
  [docs/design/position-search/benchmarking.md](design/position-search/benchmarking.md)
  § "Colab (GPU)" を正とし，ここに複製しない．**
- 同じ版数の wheel を差し替えるときは `pip install --force-reinstall --no-deps`
  (版数が同じだと再インストールされない．`docs/design/usi-engine/verification.md`)．
- `maou[cuda]` は `torch>=2.6.0,<=2.11.0` を要求する．Colab 同梱の torch が範囲外だと
  再インストールが走って時間を食うので，`pip show torch` を先に見て計画に織り込む．
- 導入後は `pip show maou` の版数と，その wheel を作った `main` の commit SHA を
  worklog に記録する (§9)．

wheel URL の解決は Python で (`colab exec -s gpu` に流す):

```python
import json, subprocess, sys, urllib.request
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
rel = json.load(urllib.request.urlopen(
    "https://api.github.com/repos/dousu/maou/releases/tags/latest"))
whl = next(a["browser_download_url"] for a in rel["assets"] if tag in a["name"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"maou[cuda] @ {whl}"])
print("INSTALL_OK", whl, flush=True)
```

## 7. maou 固有: Google Drive 規約 (MUST)

### 7.1 書き込みは Colab 内でマウント経由のみ

Drive への書き込み (作成・更新・削除) は **Colab VM で Drive をマウントし，VM 内のプロセスが行う**．
Codespaces 側の Google Drive MCP ツール，Drive API，`colab upload` などで Drive に書かない．
理由: 書き手を VM に限定すれば，成果物は必ず「VM ローカル → Drive」の 1 経路を通り，
何がどこから来たかが `worklog/` から追える．

### 7.2 操作範囲は `MyDrive/shogi` の中だけ

VM 上のパスで `/content/drive/MyDrive/shogi/` 配下だけを読み書きする．それ以外の Drive 領域は
一覧すら取らない．`shogi/` 直下に新しいトップレベルフォルダを増やす場合はユーザに確認する
(§7.4 のレイアウトが規約)．

### 7.3 まず VM ローカルに出力し，まとまりごとに Drive へコピー

Drive (FUSE) に直接書くと，細かい書き込みが遅く，`utility search-values` の `pending_*` shard
flush や TensorBoard の event ファイルのような高頻度更新で不安定になる．

- 出力先は常に VM ローカル．**ローカル側は `/content/shogi/` を Drive の `MyDrive/shogi/` と
  同じレイアウトで切る** (§7.4)．コピーが `rsync -a /content/shogi/X/ /content/drive/MyDrive/shogi/X/`
  の 1 行になり，取り違えが起きない．
- **読み取りも同様に，Drive から `/content/shogi/` へコピーしてから使う**．学習の入力
  (HCPE / 前処理済) だけでなく，TensorBoard のイベントファイルやモデルを**解析目的で
  読む場合も含む** (例: `EventAccumulator` で基準 run の LR 曲線を読む)．FUSE 直読みは
  遅く，DataLoader の並列読みで不安定になるうえ，多数の小ファイルを舐める読み取りは
  kernel の接続断を招きうる．`rsync -a /content/drive/MyDrive/shogi/X/ /content/shogi/X/`
  で取り，ローカル側を読む．Drive 上で許されるのは `ls` 相当の一覧取得だけ．
- Drive へのコピーは「作業のまとまり」単位: 1 ステージ完了時 (hcpe → preprocess → …)，
  epoch 数回ごとの checkpoint，`search-values` の shard がいくつか確定したとき，
  および **`colab stop` の直前**．コピー後に `ls -l` でサイズを突き合わせてから次へ進む．
- **次の段階を始める前に，済んだ段階の出力を退避する (MUST)**．特に 1 時間級の
  `pre-process` 出力は `learn-model` に入る前に Drive へコピーする．learn 中の周期退避
  (models / logs) だけでは，VM を失ったとき前処理からやり直しになる
  (2026-09-21 の Arm 0: learn 中の切断で `preprocess_20260921` を失った)．
  `scripts/colab_arm0_job.py` は preprocess 段階の完了直後に退避する．
- 途中で VM を失っても Drive にある中間物から再開できる状態を保つ (`search-values --resume` は
  shard ディレクトリを Drive から戻せばそのまま続きから走る)．

### 7.4 レイアウト

| データ | 生成コマンド | Drive (`MyDrive/shogi/`) | VM ローカル (`/content/shogi/`) |
|---|---|---|---|
| HCPE | `maou hcpe-convert --output-dir` | `hcpe/` | `hcpe/` |
| 前処理済 | `maou pre-process --output-dir` | `preprocess/` | `preprocess/` |
| 探索値 (shard dir) | `maou utility search-values --output-path` | `search_values/` | `search_values/` |
| モデル (`.onnx` / `.pt`) | `maou learn-model --model-dir` | `maou_test/models/` | `maou_test/models/` |
| TensorBoard ログ | `maou learn-model --log-dir` | `maou_test/logs/` | `maou_test/logs/` |

- 各フォルダの下は **`<種別>_<YYYYMMDD>`** (種別 = 親フォルダ名，日付 = 生成日) のサブフォルダを
  切る．既存の実例: `hcpe/hcpe_20260805/`，`preprocess/preprocess_20260901/`，
  `search_values/search_values_20260806/`．モデルとログも同じ形 (`maou_test/models/models_20260920/`，
  `maou_test/logs/logs_20260920/`)．同日に複数作るときは `_<HHMM>` を足す
  (例: `models_20260920_1430`)．由来や条件はフォルダ名に詰め込まず worklog に書く (§9)．
  `learn-model` の出力名 (`model_{id}_{tag}_{epoch}.onnx`) は実験条件を含まないので，
  モデルは実験 (= 1 回の `learn-model`) ごとに必ずサブフォルダを分ける．
- `learn-model` の `--model-dir` / `--log-dir` の既定は cwd (`/content`) 相対の `./models` / `./logs`
  なので **必ず明示する**．
- 上表にない一時物は Drive に置かない．TensorRT engine cache (`--trt-cache-dir`) は数分で
  再生成できるので VM ローカルに置くだけでよい．Drive に残す基準は §7.5．

### 7.5 再利用しうるデータは Drive に残す

再利用の可能性があるもの，および **再生成に 10 分以上かかるもの**は，実験が終わっても Drive に
置いたままにし，再現しやすい状態を維持する．該当: `search_values/` (1M 局面で約 22 時間)，
`preprocess/`，`hcpe/`，採用した/比較対象にしたモデル，その TensorBoard ログ．
10 分未満で再生成できるもの (TensorRT engine cache 等) は Drive に置かない．
Drive 上の削除・上書きは復旧に手間がかかるので，`shogi/` 内であっても **事前にユーザへ確認**する
(ローカル `/content/shogi/` は自由に消してよい)．

## 8. maou 固有: Colab インスタンスは計画してから起動する (MUST)

VM の確保・環境構築・データ転送はそれぞれ分単位のオーバーヘッドがあり，`new`/`stop` を
繰り返すほど無駄になる．**`colab new` の前に，そのセッションでやることを最後まで書き出す**．

計画に含める項目:

1. 前提: 検証対象の commit が `main` にあり Build Wheel が完了している (§6)．
   `scripts/` の実行スクリプトがコミット済 (§9)．
2. GPU 種別と，そのセッションで回す **すべてのステップ** (環境構築 → Drive マウント依頼 →
   入力コピー → ジョブ 1 → Drive 退避 → ジョブ 2 → … → 最終退避 → `stop`)．
   関連する検証は 1 セッションにまとめ，同じ環境構築を繰り返さない．
3. 各ステップの所要時間見込みと合計 (24 時間の壁: §3)．
4. `stop` する条件．計画どおり完了したら `stop`．計画外に途中で `stop` する場合
   (中断・方針変更) はユーザに確認する．

GPU の目安: 本番の学習 (`learn-model`) は **`G4`**，ONNX 推論系 (`search-values` / `selfplay` /
`floodgate`，TensorRT で batch 64 が実測最適) と **学習のテスト段階** (少 epoch・小データで
配線を確かめる) は **`L4`**．`T4` は古いので使わない．枠が取れなければ 1 段下に落として
計画を作り直す．ブラウザで起動するとき (§10) はユーザがこの目安で種別を選ぶ．

## 9. maou 固有: VM で走らせるものは `scripts/` に置く (MUST)

Colab で実行するスクリプトは `scripts/` に置き，コミットしてから使う．heredoc で `colab exec` に
流すのは，ファイル確認・プロセス起動・ログ tail のような**ワンライナー**に限る．

- `uv run python scripts/<name>.py ...` で手元でも動く形にする (既存の
  `scripts/measure_calibration.py`，`scripts/soften_result_value.py` が手本)．
  argparse で入出力パスを引数に取り，Drive のパスを埋め込まない (呼び出し側が
  `/content/shogi/...` を渡す)．
- 進捗を `print(..., flush=True)` で出し (§1)，終わりに番兵行 (`VERIFY_OK` / `JOB_DONE exit=N`)
  を出す (§2)．
- `maou` パッケージを import するので，wheel の版数とスクリプトの commit を揃える (§6)．
- VM へは `colab upload -s gpu scripts/<name>.py /content/<name>.py` で送る．
  `scripts/soften_result_value.py` のように 1 ファイルで完結させておくと運びやすい．
- 実行後，worklog に **再現に必要な全部** を残す: commit SHA，wheel 版数，GPU 種別，
  完全なコマンドライン，入力の Drive パス，出力の Drive パス，所要時間，番兵行の結果．
- 例外: `scripts/colab_adopt_session.py` は VM ではなく **開発マシン側** で，`colab` CLI と
  同じ仮想環境の python (`"$(uv tool dir)/google-colab-cli/bin/python"`) で実行する．

## 10. maou 固有: 時間のかかる検証は Codespaces から切り離してハンドオフする (MUST)

§3 のとおり，CLI が作ったセッションは Codespaces が落ちると回収される．時間のかかる検証は
**ユーザがブラウザで起動したランタイムを保持**し，Claude Code はそこに接続して
「投入して checkpoint を残す」役に徹する (colab セッションの維持はユーザの担当)．

CLI に attach コマンドは無いが，`colab sessions` が取る一覧に runtime proxy の token / url が
含まれており，`colab new` が保存するのも同じ 3 つ (token / url / endpoint) なので，
`scripts/colab_adopt_session.py` でローカル登録すれば同等に扱える (2026-09-20 に CPU
ランタイムで `exec` まで実機確認)．

```bash
# 1. ユーザ: ブラウザ (colab.research.google.com) で GPU 種別を選んでランタイムを起動し，
#    タブを開いたままにする．必要なら同じノートブックのセルで drive.mount も済ませる (§0(d))．
#    さらに `scripts/colab_keepalive_cell.py` の中身を新しいセルに貼って実行し，
#    ジョブの間ずっと「実行中」にしておく (下記)
colab sessions                              # 2. `[?] <endpoint> | Hardware: L4 ...` に見える
PY="$(uv tool dir)/google-colab-cli/bin/python"
"$PY" scripts/colab_adopt_session.py gpu    # 3. 未登録が 1 つならそれを gpu として登録 (複数なら --endpoint)
colab status -s gpu                         #    [gpu] <endpoint> | Hardware: L4 | Status: IDLE
# 4. Claude Code が環境構築 (§6) → Drive マウント確認 (§0(d)) → 入力コピー (§7.3) → §4 の nohup で投入
# 5. 投入を確認したら **すぐに /checkpoint-context** (下記の記録項目)
```

- 登録したセッションには keep-alive デーモンが無い (§3)．**ブラウザのタブを開いている
  だけでは足りない — ノートブックでセルを実行していないと数時間でアイドル切断される**
  (2026-09-21: Arm 0 の learn 開始 2 時間後，VM 起動から約 3.5〜4 時間で切断)．
  **`scripts/colab_keepalive_cell.py` を新しいセルに貼って実行し，ジョブの間ずっと
  実行中のままにする (MUST)**．24 時間走り，5 分ごとに `/content/job.log` と
  `/content/shogi/arm0_*/STATUS` の末尾を表示し直す (進捗もここで見える)．
  `colab exec` で流すものではない (ブラウザが attach している kernel で走らせる)．
  Colab 側の上限 (最長実行時間) はユーザのプラン次第で，ユーザが把握する．
- **登録したセッションを `colab stop` しない．** `stop` は VM を unassign するので，ユーザが
  ブラウザで見ているランタイムごと消える．終了はユーザに報告し，ランタイムの削除は
  ユーザがブラウザ側で行う．ローカル登録だけ外すなら `--forget`．
- **完走したら VM 側が自分で unassign する (MUST)**．長時間ジョブの driver には
  `google.colab.runtime.unassign()` と同じ `POST http://$TBE_RUNTIME_ADDR/unassign` を
  rc=0 の終わりに入れる (`scripts/colab_arm0_job.py --unassign-on-done`．環境変数は
  kernel の子プロセスに継承されるので nohup からでも通る)．CLI 側から `stop` しない規約は
  そのまま — 止めるのは VM 上のジョブ自身．猶予 (`--unassign-grace-min`，既定 60 分) の間に
  `colab download` で小物 (value-best の `_fp16.onnx` 等) を取る．猶予を過ぎて VM が消えた
  あとは Drive にある (§7.3 で退避済み) ので，ユーザに CPU ランタイム + `drive.mount` を
  依頼して `colab download` で取る．**失敗 (rc≠0) のときは段階ログを見られるよう VM を
  残す**ので，診断後にユーザがブラウザで削除する．
- 一覧の token には `tokenExpiresInSeconds: 3600` が付くが CLI は token を更新しない
  (`new` で作ったセッションも 1 つの token を最長 24h 使う設計)．401 が出たら
  `colab_adopt_session.py gpu --force` で最新 token に置き換える．
- `/checkpoint-context` は dirty-tree gate を持つ (`src/` / `rust/` が未コミットなら拒否)．
  §6 で先に `main` へ入れているので通常は通る．
- checkpoint (worklog + `scratchpad/current.md`) に残す項目: ローカル名・endpoint
  (`colab status -s gpu`)・GPU・投入コマンドとログのパス (`/content/job.log`)・終了判定の番兵・
  完了予想時刻・出力のローカルパスと Drive 退避先・残りステップ (退避 → ユーザへ報告)．
- Codespaces が落ちた後: 新セッションで `/resume-context` → `colab status -s gpu`
  (`sessions.json` は Codespaces 停止→再開で残る．404 なら VM は消えた．Drive の中間物から
  §7.3 で再開) → ログをポーリング → 完了したら Drive へ退避 → `ls -l` で確認 → ユーザに報告．
- container rebuild は `sessions.json` と token を消す (§3)．消えても VM はブラウザ側で
  生きているので，`colab sessions` → `colab_adopt_session.py gpu` で登録し直せば復帰できる
  (この点は `new` で作ったセッションより強い)．

**`colab url` は代替にならない (未検証)．** `colab url -s gpu` の URL をブラウザで開くと
Colab フロントエンドは既存 VM に attach するが，そのタブが通常のランタイムと同じ keep-alive を
送るかは確認していない (`dbu` / `datalabBackendUrl` は開発用フラグ)．Codespaces 側デーモンが
落ちた時点で VM が回収されるリスクを残すので，長時間ジョブでは使わない．
Claude Code が先に `colab new` してしまった場合は，短時間で終わるものだけそのまま回し，
長時間ジョブは `colab stop` してユーザにブラウザで作り直してもらう (VM 再確保 1 回分の
オーバーヘッドのほうが，回収されて成果を失うより安い)．

---

## 11. Claude Code の判断フロー (要約)

1. 認証確認 → `colab sessions` が通らなければ人間に §0(c) を依頼して停止．
2. 計画を書く (§8): 対象 commit は `main` + wheel あり (§6)，スクリプトは `scripts/` にコミット済 (§9)，
   GPU 種別，全ステップ，所要時間，`stop` 条件．
3. **単発検証 (〜数十分，wheel 不要)** → `colab run --gpu L4 --timeout <十分な秒数> script.py`
   で exit code を判定に使う．
4. **反復検証 (同一環境を使い回す)** → `colab new -s <name>` → §6 で wheel → 人間に
   `drivemount` を依頼 → 入力を `/content/shogi/` へ → `colab exec --timeout ...` →
   出力を Drive の所定フォルダへ (§7.4) → **終了時に必ず `colab stop -s <name>`**
   (自分で `new` したセッションに限る)．
5. **時間のかかる検証** → §10: ユーザがブラウザで起動 → `colab sessions` →
   `scripts/colab_adopt_session.py gpu` で登録 → nohup 投入 → 即 `/checkpoint-context`．
   Codespaces 復帰後は `/resume-context` からポーリング・退避・報告 (**`stop` しない**)．
6. 失敗したら `colab log` / `colab status` を見てから再試行．GPU が取れない場合は 1 段下に
   フォールバックし，計画を作り直す．
