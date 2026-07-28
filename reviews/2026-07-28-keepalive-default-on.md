---
title: keep-alive 空行の既定を on (5000ms) にする — 未決 2 の決着
date: 2026-07-28
status: approved
target:
  - docs/design/usi-engine/index.md
  - docs/commands/usi.md
risk: low
reversibility: easy
---

# 提案: `KeepAlive` の既定を off → on (5000ms) にする

## Trigger

`docs/design/usi-engine/index.md:289` の未決事項 2 が「GUI 実機待ち」の
まま残っていた最後の 1 件だった．判定条件は
[verification.md §8](../docs/design/usi-engine/verification.md) が
明示している:

> **`KeepAlive` の空行を GUI が無害に無視するか** — 無視するなら既定を
> on にできる (**未決 2 の判断はこれだけが根拠になる**)．壊れる GUI が
> あるなら既定 off のまま，該当 GUI 名を docs に残す．

2026-07-28 に Windows / **ShogiHome** の実機で確認した．

## 実証

`KeepAlive 200` を `setoption` で指定した GUI のエンジン通信ログ:

```
▶ isready
◀
◀
◀ readyok
▶ usinewgame
▶ quit
◼ closed: close=0 signal=null
```

- **空行が 2 行流れた** (機構が発火している — 非空性の確認)
- **GUI はそれを無害に無視した** — エラーを出さず `usinewgame` へ進み，
  対局に入れた．終了も `close=0 signal=null` で正常

発火と無害性の**両方**を観測している点が重要．空行が 0 行のまま「GUI が
壊れなかった」を見ても，無視されたのか発火していないのかが区別できず
判定にならない (この罠を踏みかけた — 手元の `isready` は 0.3 秒で
`KeepAlive 500` では 1 行も出なかった)．

## 変更

- コード既定 `keep_alive_ms` を **0 → 5000** に:
  `rust/maou_usi/src/agent.rs` (`EngineConfig::default`) /
  `src/maou/infra/console/usi.py` (`--keep-alive-ms`) /
  `src/maou/interface/usi.py` / `src/maou/app/usi/run.py`
- `docs/design/usi-engine/index.md` §12 の表 2 行目と冒頭の注記を決着へ
- `docs/commands/usi.md` の CLI 表と USI option 表の既定値・説明

値を **5000ms** にする根拠: KeepAlive の目的は TensorRT の初回エンジン
ビルド (数十秒) が GUI の `readyok` タイムアウトを超える構成の延命なので，
`isready` が速い環境で 1 行も出ないのは正常な挙動．細かく刻む理由がない．

## VETO との関係

compass の VETO「レバーは『より強い』ことを A/B で確認してから既定化
(既定 off 実装 → 発火量 → 棋力 A/B の順)」は形式上「既定 off → on」に
該当するが，**KeepAlive は探索に一切触れず棋力に影響しない**ため VETO の
趣旨 (探索を変えるトグルを A/B 抜きで既定化しない) には抵触しない．
user 確認済み (2026-07-28)．

なお VETO の要求のうち「発火量の確認」は上記ログ (空行 2 行) で満たして
いる．棋力 A/B は対象が棋力に影響しないため不要．

## Risks

- **確認したのは ShogiHome のみ**: 中 / 互換性 / 将棋所・ShogiGUI・
  各種サーバは未確認．空行で壊れる実装があれば既定 on が回帰になる．
  緩和: `setoption name KeepAlive value 0` で即座に無効化できる．
  壊れる GUI が見つかったら docs に名前を残し既定を戻す．
- **速い環境では無音**: 低 / 運用 / 「設定したのに何も出ない」を不具合と
  誤認し得る．docs とヘルプに「無音が正常」と明記した．
