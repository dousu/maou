# boto3でのAWS S3 Sync相当の実装とAWS CLI v2との比較

## 概要

AWS CLI v2の`aws s3 sync`コマンドは非常に高速で、特に小さなファイルが大量に存在する場合でも効率的に同期できます。一方で、boto3を用いて同等の処理を自作した場合、処理時間が著しく遅くなるケースがあります。

本レポートでは、AWS CLI v2が高速な理由と、boto3での実装でパフォーマンスを改善する方法について調査・整理しました。

---

## 前提条件

- 対象ファイルサイズ：100KB前後
- ファイル数：約1,000,000個
- boto3では`download_file()`を用いた並列処理を実施

---

## 比較表：AWS CLI v2 vs boto3

| 項目 | AWS CLI v2 (`aws s3 sync`) | boto3 + `download_file()` |
|------|-----------------------------|----------------------------|
| 並列処理 | 高度に最適化された並列処理（スレッド、キュー、非同期制御） | 自前実装が必要 |
| 接続管理 | HTTPセッションとTCP接続を自動再利用 | セッションの再利用が困難でコスト増 |
| ファイル取得 | `TransferManager` による高速ダウンロード | 単純な `GetObject` API 呼び出し |
| 小さなファイルの最適化 | 効率的なプリフェッチとバッチ処理 | ファイルごとに処理が分離し遅延発生 |
| 再試行・リトライ戦略 | バックオフ含む堅牢な再試行ロジック | 明示的に制御しないと弱い |
| スレッド／リソース管理 | 自動調整（最大並列数10+） | 設定しないと非効率 |

---

## パフォーマンスが遅くなる主な原因（boto3側）

1. **`download_file()` は逐次的で、1ファイルずつ HTTP リクエストが発生する**
2. **並列処理のスレッド数がボトルネックになる**
3. **接続の再利用がされず、毎回 TLS ハンドシェイクが発生する**
4. **S3 `ListObjectsV2` の処理も逐次的になりがち**

---

## 推奨対策：CLIレベルのパフォーマンスをboto3で再現するには

### ✅ `TransferManager` の使用

```python
import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer

s3 = boto3.client('s3')

config = TransferConfig(
    max_concurrency=16,
    multipart_threshold=8 * 1024 * 1024,  # 小さいファイルは単一リクエストで
)

transfer = S3Transfer(client=s3, config=config)

def download_file(bucket, key, dest):
    transfer.download_file(bucket, key, dest)
```

### ✅ ThreadPoolExecutor などで並列実行

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=16) as executor:
    for key in keys:
        executor.submit(download_file, bucket, key, f'./downloads/{key}')
```

### ✅ セッション再利用（オプション）

```python
import botocore.session

session = botocore.session.get_session()
s3 = session.create_client('s3')
```

## まとめ

AWS CLI v2の`aws s3 sync`が高速な理由は、以下の技術的最適化によるものです：

- TransferManagerの活用と最適なスレッド制御
- セッション・TCP接続の再利用によるレイテンシ削減
- 小ファイル向けに設計された効率的な内部キューとバッファ制御
- boto3で同等の速度を実現するには、TransferManagerを活用しつつ、並列制御・接続最適化・リスト操作などの部分を明示的に設計・調整する必要があります。

## 補足

- `aioboto3` による非同期並列処理
- ファイルリストの事前取得とスケジューリングの導入
- ダウンロード対象の条件フィルタリング（プレフィックス、更新時刻など）
