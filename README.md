# maou
将棋AIを作ってみる

## 開発環境

基本的にはdevcontainerを前提にする．
以下のスクリプトを実行する．

```bash
bash scripts/dev-init.sh
```

ここでシェルスクリプトを実行するような構成になっているのは，
devcontainerのfeaturesになるべくインストールを任せたいため．

featuresはインストール順序としては最後になるためPython等に依存しているとDockerfileにインストール処理を書けない．
