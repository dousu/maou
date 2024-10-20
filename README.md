# maou
将棋AIを作ってみる

## 開発環境

基本的にはdevcontainerを前提にする．
以下のスクリプトを実行する．

```bash
bash scripts/dev-init.sh
# ここで確認できるpythonのパスをVScodeのインタプリタとして設定する
poetry env info
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
