# 39手詰問題 — Oracle リファレンス

このドキュメントは **39手詰 (39te) ベンチ問題**の oracle (正解手数・正解手順) を記録する．
**詰将棋の最短手数・正解 PV の絶対 oracle は user のみ**であり (KH/ぴよ将棋等のソルバーは参照に過ぎない)，
user がこのドキュメントを介して maou の解と oracle を照合できるようにする目的を持つ．

各エントリは **CONFIRMED (user 確認済)** / **PENDING (maou findings; user 確認待ち)** を明示する．
新たな分岐を確認したら追記し，PENDING を CONFIRMED へ昇格する．

---

## Oracle 検証の規則 (user 2026-06-30)

- 将棋には**同手数の詰みが複数存在し得る**．node 種別で bug 判定が異なる:
  - **OR node (攻め方手番)**: maou が同手数 or **より長い**詰みを出すのは探索に影響なし (bug でない)．
    **明らかに短い**詰みが見つかった場合のみ，探索結果をそれに合わせる必要がある = bug．
  - **AND node (受け方手番)**: **より長い**受け (手順) が見つかれば探索もそれに合わせる必要がある = bug
    (maou が受けの長さを短く見積もっている)．
- → oracle 確認は **「OR node で明らかに短い手順が出る」** か **「AND node で明らかに長い手順が出る」**
  の形に絞る．それ以外 (OR で長め / AND で短め) は bug でない可能性が高い．
- **無駄合いは手数に数えない** (探索は全合駒を生成するが，pv.rs post-pass が長さ集計から除外する)．
- 乖離報告の形式: `局面 SFEN → maou の手 M (→L手) vs oracle の手 M' (→L'手); maou は LONGER/SHORTER by D`．

---

## 問題 (root)

- **root SFEN**: `9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1`
- **正解**: **39手** (受け方最善)．**[CONFIRMED]**
- 正解 PV (USI; 受け方は 8手目に正着 2c3b):
  ```
  7b6b 5b4c 8b9c 4c3d 1b2c 3d2c N*1e 2c3b N*2d 3b2b 2d1b+ 2b3b 1b2b 3b2b 4f1c 2b1c
  9c3c 1c1d 3c2c 1d1e P*1f 1e1f P*1g 1f1g 5g6f 1g1h 2c2g 1h1i 8g8i S*6i 8i6i 6h6i+
  S*2h 1i2i 2h3g 2i3i 2g2h 3i4i 2h4h
  ```

---

## 2c3d 変化 (8手目の受け方失着)

受け方が 8手目に **2c3d (3四玉)** を選ぶと正着 2c3b (→39) より**短く**詰む → 失着．

- **2c3d は全体 35手で詰む** (39 より短い)．→ 受け方は 2c3d を選ばない．**[CONFIRMED]**
- **post-2c3d 局面**: `9/3+N1P3/+R5p2/6k2/8N/5+B3/1R2S4/3p5/9 b NPb4g3sn4l14p 9`
  - post-2c3d から **27手** (= total 35 − 先手から8手)．**[CONFIRMED]**
- 受け方は**外へ逃げて最大抵抗**する．4d5c (即5三逃げ)・4c3b (3二逃げ) は正当な長手数の受け．**[CONFIRMED]**

### Oracle line (Branch A; 即 4d5c 逃げ; total 35)

```
post-2c3d → 8g8d P*4d 8d4d 3d4d 9c8d 4d5c 8d6d 5c4b P*4c 4b4c N*3e 4c3b → [king-3二 = 15手]
```

### 確認済みサブ局面 (user oracle)

| ラベル | SFEN | 手数 | 到達 (post-2c3d から) |
|---|---|---|---|
| **king-3二** | `9/3+N2k2/6p2/3+R5/6N1N/5+B3/4S4/3p5/9 b Prb4g3sn4l15p 21` | **15** | `...9c8d 4d5c 8d6d 5c4b P*4c 4b4c N*3e 4c3b` |

---

## 2c3d の正解 = 39手 (N*3f 経由) — **[CONFIRMED]**

正着 attacker は k=4 で **N*3f** (9c8d ではない; 下記 P1)．post-2c3d = **31手** (total 39)．

### Oracle line (N*3f; total 39)
```
post-2c3d → 8g8d P*4d 8d4d 3d4d N*3f 4d3d 9c8d 3d4c 8d4d 4c3b 3f2d 3b2a P*2b 2a2b
            P*2c 2b1a 2d1b+ 1a1b 4d1d 1b2a 2c2b+ 2a2b 1e2c+ 2b3a 4f1c 3a4b 1d4d
            P*4c 2c3c 4b4a 4d4c
```

### (P1 解決) k=4 の 9c8d は P*5四 で refute される

- `9c8d` 直後の `P*5四` 中合い局面 `9/3+N1P3/6p2/1+R2pk3/8N/5+B3/4S4/3p5/9 b N2Prb4g3sn4l12p 15`
  は **不詰**: maou (pn=∞), KomoringHeights (nomate) **両者一致**，かつ user の正解 line が
  `9c8d` でなく **N*3f** を採用することで確認された．→ **k=4 では N*3f が正着**．maou は正しく N*3f を選ぶ．

## 確認済みサブ局面 (N*3f line) — **[CONFIRMED]**

| ノード (N*3f line) | oracle | maou standalone |
|---|---|---|
| after 22 plies `9/3+N1P1k1/6p2/8+R/8N/5+B3/4S4/3p5/9 b ...31` | 9 | 9 ✓ |
| after 24 plies | 7 | 7 ✓ |
| after 26 plies (1d4d 直前) `9/3+N1k3/6p+N+B/8+R/9/9/4S4/3p5/9 b rb4g3s2n4l16p 35` | 5 | 5 ✓ (P*4c→4d4c) |
| **S*4c 局面** `9/3+N1k3/5sp+N+B/5+R3/9/9/4S4/3p5/9 b rb4g2s2n4l16p 37` | 3 | **3** ✓ (2c3c 4b4a 4d4c; 銀を取って詰み) |

## 解決済の過去乖離 (記録)

かつて maou は 39te root を 43 手 / post-2c3d を 35 手と過大評価していた．真因は
**find_shortest の len 予算が無駄合い込みの raw ply を数える units 不整合** (受け方が透過中合いで
raw 長を膨らませると len-cutoff が短い詰みを偽 disproof する) であり，**無駄合い-free len credit**
([aigoma §8.4](aigoma-optimization.md)) で解消した．現在は root=**39** / post-2c3d=**31** /
29te=**29** が user oracle と一致し，STRICT verify + canonical anchor (`tests.rs`) で回帰を
ガードしている．乖離の局所化には `test_39te_divergence_probe` (正解手順を進めながら各局面の
残手数を期待値と照合) を用いる．

---

## 更新履歴

- 2026-06-30: 初版．king-3二=15・post-2c3d=27(total 35)・2c3d 失着を CONFIRMED 記録．
  P1 (9c8d P*5四 偽 disproof 疑い)・P2 (k=4 で 9c8d 回避) を PENDING 記録．
- 2026-07-02: P1 解決 (P*5四 中合いは不詰; N*3f が正着)・post-2c3d=31 (total 39) を CONFIRMED 化．
  root 43/post-2c3d 35 の過大評価は無駄合い len credit で解消済み — 経緯を「解決済の過去乖離」に集約．
