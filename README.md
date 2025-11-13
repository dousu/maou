# maou
将棋AIを作ってみる

## 開発環境

基本的にはdevcontainerを前提にする．
以下のスクリプトを実行する．

```bash
# devcontainerの場合は以下をインストールしておく
# bash scripts/devcontainer.sh
bash scripts/dev-init.sh
# ここで確認できるpythonのパスをVScodeのインタプリタとして設定する
poetry env info --path
# pre-commit系の設定
poetry run bash scripts/pre-commit.sh
```

ここでシェルスクリプトを実行するような構成になっているのは，
devcontainerのfeaturesになるべくインストールを任せたいため．

featuresはインストール順序としては最後になるためPython等に依存しているとDockerfileにインストール処理を書けない．

### Pythonアップデート方法

```bash
# pythonのバージョンが新しくなっていることを確認する
python --version
# 今使っている古いpython環境を確認する
poetry env list
poetry env remove ${古いpython環境}
poetry env use python
bash scripts/dev-init.sh
```

### poetry cache削除

poetryのcacheがたまってストレージ容量を圧迫している場合は以下のコマンドでpoetryのcacheを消せる．
GitHub Codespacesを使っている場合等，ストレージ容量をなるべく削減したいときに利用する．

```bash
poetry cache clear --all .
```

### GCPを使う場合

以下のコマンドを実行してGCPへの認証をしておくと，テストやプログラム中でGCPを利用できる．

```bash
gcloud auth application-default login
# gcloud projects listで設定可能なプロジェクトを確認できる
gcloud config set project "your-project-id"
gcloud auth application-default set-quota-project "your-project-id"
```

なお，GCPを使ったテストをするときは以下のように行う．

```bash
TEST_GCP=true poetry run pytest
```

### AWSを使う場合

```bash
aws configure sso --use-device-code --profile default
# アクセストークンが切れたら以下のように再認証する
# aws sso login --use-device-code --profile default
```

なお，AWSを使ったテストをするときは以下のように行う．

```bash
TEST_AWS=true poetry run pytest
```

## Masked Autoencoderの事前学習コマンド例

`maou pretrain`コマンドでMasked Autoencoderの事前学習を行う．
事前学習で保存したstate_dictは`maou learn-model --resume-from`オプションで読み込める．

```bash
poetry run maou pretrain \
  --input-dir data/preprocessed \
  --input-format preprocess \
  --output-path artifacts/masked-autoencoder.pt \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --mask-ratio 0.75 \
  --hidden-dim 512
```

## Preprocessingデータのメモリマップ方式

前処理済みの`.npy`ファイルはデフォルトでコピーオンライト(`mmap_mode="c"`)
としてメモリマップされます。これにより`numpy.memmap`が
`writeable=True`のまま保持され、`torch.from_numpy()`によるゼロコピー変換が
可能になります。ファイルを汚染することなくテンソルから値を更新できるため、
訓練用データセットをGPUへ転送する際のコピーを削減できます。

ファイルシステム／オブジェクトストレージ／BigQueryの各データソースには
`preprocessing_mmap_mode`引数を追加しており、必要に応じてコピーオンライト以外の
モードへ切り替えることもできます。
