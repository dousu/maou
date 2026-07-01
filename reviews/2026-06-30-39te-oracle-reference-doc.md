---
status: applied
applied_in: 03f635a
target: docs/design/tsume-solver/39te-oracle.md, docs/design/tsume-solver/index.md
date: 2026-06-30
---

# 39手詰 Oracle リファレンスドキュメントの新設

## 背景

39te ベンチ問題の oracle (正解手数・正解 PV・サブ局面の手数) は user のみが
絶対 oracle であり ([[feedback_confirm_before_diverging]])，分岐が多岐にわたる．
session 中に piecemeal で user に確認した oracle 事実が散逸するのを防ぎ，
user がドキュメントを介して maou の解と照合できるようにする．

## 提案

`docs/design/tsume-solver/39te-oracle.md` を新設し，以下を記録する:

- root 問題 (SFEN / 正解39手 / 正解 PV)．
- Oracle 検証の規則 (OR node=明らかに短い手順のみ bug / AND node=明らかに長い受けのみ bug;
  無駄合いは数えない; 乖離報告形式) — user 2026-06-30 指示．
- 2c3d 失着変化 (post-2c3d=27手 / total 35; 外逃げ受けは正当)．
- 確認済みサブ局面表 (king-3二=15)．
- PENDING (maou findings; user 確認待ち): P1=9c8d後P*5四を maou が偽 disproof 疑い,
  P2=k=4 で 9c8d 回避．

`index.md` にリンク行を 1 行追加．

## 承認

user が「詰将棋 oracle は多岐にわたるのでドキュメントに記載; user が絶対 oracle として
ドキュメント経由で確認できるように」と明示要求 (2026-06-30) → approved．

## 適用

- `docs/design/tsume-solver/39te-oracle.md` 新規作成．
- `docs/design/tsume-solver/index.md` に 1 行追加．
- (commit は oracle 内容が一段落してから; PENDING 解消で随時更新する living doc)．
