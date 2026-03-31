# 詰将棋ソルバー設計ドキュメント

## 目次

1. [概要](#1-概要)
2. [探索アーキテクチャ](#2-探索アーキテクチャ)
3. [閾値制御](#3-閾値制御)
4. [証明数・反証数の計算](#4-証明数反証数の計算)
5. [初期値ヒューリスティック](#5-初期値ヒューリスティック)
6. [転置表管理](#6-転置表管理)
7. [ループ・GHI 対策](#7-ループghi-対策)
8. [合駒最適化](#8-合駒最適化)
9. [手順改善](#9-手順改善)
10. [既知の課題とベンチマーク](#10-既知の課題とベンチマーク)
11. [最適化案の評価](#11-最適化案の評価)
12. [参考文献](#12-参考文献)

---

## 1. 概要

maou_shogi の詰将棋ソルバーは Df-Pn (Depth-First Proof-Number Search, Nagai 2002) を
基盤とし，Best-First PNS と IDS-dfpn(Frontier Variant 統合)の
2フェーズ探索を採用する．

### 設計目標

1. **cshogi が解けない問題をカバーする**: 片玉局面，合駒(中合い)の正確な探索，最短手順保証
2. **どんな詰将棋問題も解ける**: リソースパラメータ(`depth`, `nodes`, `timeout`)の増加で対応
3. **高度な枝刈りによる効率化**: cshogi が取り入れていない手法を積極的に導入

### 実装ファイル

`rust/maou_shogi/src/dfpn/` モジュールに全ての探索ロジックを実装(合計約 6,800 行，テスト除く)．

| ファイル | 行数 | 内容 |
|---------|------|------|
| `solver.rs` | ~2,950 | `DfPnSolver` 構造体，`mid()` 関数，child init，MID ループ |
| `pns.rs` | ~2,540 | 手生成，PNS メインループ，IDS-dfpn，Frontier Variant，PV 復元 |
| `tt.rs` | ~720 | フラットハッシュテーブル型転置表 |
| `mod.rs` | ~380 | 定数，ユーティリティ関数(SNDA, hand\_gte, DFPN-E 等) |
| `entry.rs` | ~80 | `DfPnEntry`, `PnsNode` データ構造 |
| `profile.rs` | ~190 | プロファイリングマクロ・統計 |

### 実装済み手法一覧

| # | 手法 | 出典 | 節 | 導入版 |
|---|------|------|-----|--------|
| 1 | Df-Pn | Nagai 2002 | §2.1 | v0.1.0 |
| 2 | Best-First PNS | Seo, Iida & Uiterwijk 2001 (PN*) | §2.2 | v0.18.0 |
| 3 | IDS-dfpn | Seo et al. 2001; Nagai 2002 | §2.3 | v0.16.0 |
| 4 | 1+ε トリック | Pawlewicz & Lew 2007 | §3.1 | v0.11.0 |
| 5 | TCA | Kishimoto & Müller 2008; Kishimoto 2010 | §3.2 | v0.14.0 |
| 6 | WPN | Ueda et al. 2008 | §4.1 | v0.17.0 |
| 7 | CD-WPN | maou 独自 | §4.2 | v0.20.0 |
| 8 | VPN | Saito et al. 2006 | §4.3 | v0.15.0 |
| 9 | SNDA | Kishimoto 2010 | §4.4 | v0.11.0 |
| 10 | df-pn+ ヒューリスティック初期化 | KomoringHeights v0.4.0; GPW 2004 | §5.1 | v0.12.0 |
| 11 | DFPN-E | Kishimoto et al., NeurIPS 2019 | §5.2 | v0.13.0 |
| 12 | Deep df-pn | Song Zhang et al. 2017 | §5.3 | v0.12.0 |
| 13 | インライン詰み検出 | — | §5.4 | v0.12.0 |
| 14 | 持ち駒優越 | Nagai 2002 | §6.1 | v0.1.0 |
| 15 | 前方チェーン補填 | maou 独自 | §6.2 | v0.20.0 |
| 16 | Pareto frontier 管理 | Breuker et al. 1994 | §6.3 | v0.12.0 |
| 17 | TT GC | — | §6.4 | v0.12.0 |
| 18 | TT Best Move | KomoringHeights v0.4.0 | §6.5 | v0.16.0 |
| 19 | 経路依存フラグ付き GHI 対策 | Kishimoto & Müller 2004/2005 | §7.1 | v0.15.0 |
| 20 | NM Remaining 伝播 | — | §7.2 | v0.16.0 |
| 21 | Futile/Chain 合駒分類 | — | §8.1 | v0.12.0 |
| 22 | チェーンドロップ3カテゴリ制限 | — | §8.2 | v0.12.0 |
| 23 | 合駒遅延展開 | KomoringHeights v0.5.0 | §8.3 | v0.12.0 |
| 24 | TT ベース合駒プレフィルタ | — | §8.4 | v0.16.0 |
| 25 | 同一マス証明転用 | — | §8.5 | v0.18.0 |
| 26 | 合駒 DN バイアス | — | §8.6 | v0.12.0 |
| 27 | チェーンマス内→外順序 | — | §8.7 | v0.20.0 |
| 28 | チェーン深さ DN スケーリング | — | §8.8 | v0.20.0 |
| 29 | TT Best Move 動的手順改善 | — | §9.1 | v0.16.0 |
| 30 | Killer Move | — | §9.2 | v0.18.3 |
| 31 | 捨て駒ブースト | — | §9.3 | v0.13.0 |
| 32 | Frontier Variant (PNS→局所MID) | maou 独自 | §11.7 | v0.20.33 |
| 33 | TT フラットハッシュテーブル | — | §6.6, §11.4 | v0.20.34 |
| 34 | チェーン AND pn\_floor boost | maou 独自 | §3.3 | v0.20.35 |
| 35 | PN\_UNIT 統一スケーリング | maou 独自 | §3.5 | v0.20.36 |

---

## 2. 探索アーキテクチャ

### 2.1 Df-Pn (Nagai 2002)

**出典:** Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002);
Nagai, "Df-pn Algorithm for Searching AND/OR Trees" (Ph.D. Dissertation, UTokyo, 2002)

AND/OR 木を証明数(pn)・反証数(dn)に基づいて深さ優先で探索するアルゴリズム．
各ノードに pn/dn の閾値を設定し，閾値を超えた時点で親に復帰する．

- **OR ノード**(攻め方手番): `pn = min(child_pn)`, `dn = sum(child_dn)`
- **AND ノード**(守備方手番): `pn = sum(child_pn)`, `dn = min(child_dn)`

```
    OR node (attacker)              AND node (defender)
    pn=1, dn=5                      pn=5, dn=1
   /    |    \                     /    |    \
 AND   AND   AND                 OR    OR    OR
 pn=3  pn=1  pn=2              pn=2  pn=1  pn=2
 dn=2  dn=1  dn=2              dn=1  dn=3  dn=2
  |     ^                             ^
  |   select                        select
  |  (min pn)                      (min dn)

 pn = min(3,1,2) = 1            pn = sum(2,1,2) = 5
 dn = sum(2,1,2) = 5            dn = min(1,3,2) = 1
```

OR ノードでは pn が最小の子を選択して再帰(最も証明しやすい手を優先)．
AND ノードでは dn が最小の子を選択して再帰(最も反証しやすい応手を優先)．

**実装:** `mid()` 関数 (`solver.rs`)．
MID (Multiple Iterative Deepening) ループにより，選択→展開→バックアップを
閾値に達するまで繰り返す．

### 2.2 Best-First PNS (Phase 1)

**出典:** Seo, Iida & Uiterwijk, "The PN*-search algorithm" (AI 129, 2001);
Allis, "Searching for Solutions in Games and Artificial Intelligence" (1994)

明示的な探索木(アリーナ)上でグローバル最適なノード選択を行う best-first 探索．
Df-Pn の深さ優先制約を緩和し，thrashing(同一ノードの再展開)を回避する．

**実装:** `pns_main()` 関数 (`pns.rs`)．

- **アリーナ**: `Vec<PnsNode>` (`entry.rs`)，上限 `PNS_MAX_ARENA_NODES = 5,000,000`
- **ノード予算**: `PNS_BUDGET_CAP = 150,000` ノード (全体の 1/4，上限 150K)
- **停滞検出**: `PNS_STAGNATION_LIMIT = 500,000` イテレーション
- **合駒遅延展開**: AND ノードの合駒(drop)を `deferred_drops` に格納し，逐次活性化
- **TT 転写**: `pns_store_to_tt()` で証明済み(pn=0)・反証済み(dn=0)ノードのみを TT に保存．
  中間ノードは保存しない(Phase 2 の MID が中間値に束縛されるのを防止)

**出典との差異:**
- PN* は RBFS ベースの反復深化型だが，maou_shogi は明示的アリーナの best-first 方式
- PNS のバックアップは標準 OR/AND 公式だが，AND ノードに WPN (§4.1) を適用

### 2.3 IDS-dfpn (Phase 2)

**出典:** Seo et al. 2001 (PN*); Nagai 2002 (df-pn の反復深化)

探索深さ制限を段階的に増加させ，浅い証明を TT に蓄積しながら深い探索を実行する．
PNS で未解決の場合に自動的に Phase 2 として呼び出される．

**実装:** `mid_fallback()` 関数 (`pns.rs`)．

- **深さ進行**: 倍増ステップ

```
  depth=41 (long):  2 -> 4 -> 8 -> 16 -> 32 -> 41
                    |    |    |     |     |      |
                    +--->+--->+---->+---->+----->+
                    retain proofs between each step

  depth=31 (short): 2 -> 4 -> 31
                    |    |     |
                    +--->+---->+  (skip 8,16: jump to full depth)
```

  - `depth ≤ 31` の場合: 2 → 4 → depth (中間ステップを省略)
- **予算配分**: 各浅い反復に `remaining_budget / (remaining_steps + 1)` を割り当て，
  最終反復にノードを温存
- **反復間 TT 清掃**:
  - `remove_path_dependent_disproofs()`: 経路依存の反証を除去
  - `remove_stale_for_ids()`: 浅い深さの仮反証(remaining=0)を除去
- **NM 昇格**: 反復終了後に `depth_limit_all_checks_refutable()` で全王手が
  反駁可能と確認できれば，NM を `REMAINING_INFINITE` に昇格

**出典との差異:**
- 論文の IDS-dfpn は単純な深さ制限増加だが，maou_shogi では倍増ステップ +
  適応的予算配分 + 反復間 TT 清掃を組み合わせた独自方式
- MID 呼び出し時の閾値は `INF-1`(事実上無制限)で，深さ制限のみで探索範囲を制御

### 2.4 全体フロー

```mermaid
flowchart TD
    A[solve] --> B[Phase 1: PNS]
    B -->|budget: min\nmax_nodes/4, 150K| C{root proved?}
    C -->|yes| D[extract PV from arena]
    C -->|no| E[pns_store_to_tt: proof/disproof only]
    E --> F[Phase 2: IDS-dfpn]
    F --> G[IDS loop: depth 2,4,8,...,full]
    G -->|shallow depth| H[MID: depth-first search]
    G -->|full depth| I[MID 1/2 budget]
    H --> J{proved/disproved?}
    J -->|yes| K[extract PV from TT]
    J -->|no| L[TT cleanup + NM promotion]
    L -->|next depth| G
    I --> M{proved?}
    M -->|yes| K
    M -->|no| N[Frontier: PNS→MID cycle\nremaining 1/2 budget]
    N --> O{proved?}
    O -->|yes| K
    O -->|no| P[return Unknown]
    D --> Q[return Checkmate]
    K --> Q
```

### 2.5 Phase 1 → Phase 2 の連携

```mermaid
flowchart LR
    subgraph Phase1[Phase 1: PNS]
        AR[Arena tree]
        AR -->|proved pn=0| S[pns_store_to_tt]
        AR -->|disproved dn=0| S
        AR -.-x|intermediate\npn>0,dn>0| X[discard]
    end

    subgraph TT[Transposition Table]
        P[proof entries]
        D[disproof entries]
    end

    subgraph Phase2[Phase 2: IDS-MID]
        M[mid] -->|lookup| TT
        M -->|try_capture_tt_proof| P
        M -->|prefilter| P
    end

    S --> P
    S --> D
```

#### PNS が TT に与える影響

Phase 1 の PNS 終了時に `pns_store_to_tt()` が呼ばれ，
アリーナ上の**証明済み(pn=0)・反証済み(dn=0)ノードのみ**を TT に転写する．
中間ノード(pn>0 かつ dn>0)は転写しない．

この設計には2つの意図がある:

1. **Phase 2 への情報伝達**: PNS が発見した浅い証明(例: 1手詰み，3手詰み)が
   TT に蓄積されるため，Phase 2 の IDS-MID が同じ局面に到達した際に
   TT ヒットで即座にスキップできる．特に OR ノードの子初期化時に
   `try_capture_tt_proof` (§8.4) が PNS 由来の証明を参照し，
   合駒の展開自体を回避できるケースが増える

2. **Phase 2 の自由度確保**: 中間ノードの pn/dn を転写すると，
   MID がそれらの値に束縛されて閾値配分が歪む．PNS の pn/dn は
   best-first 的な評価値であり，depth-first の MID とは閾値体系が異なるため，
   中間値の混在は MID の探索効率を低下させる

#### IDS 単体との差異

IDS のみのソルバー(Phase 1 なし)と比較した PNS → IDS の利点:

| 観点 | IDS 単体 | PNS → IDS |
|------|---------|-----------|
| 浅い詰みの発見 | 浅い IDS 反復で発見(再探索コストあり) | PNS が1回で発見(thrashing なし) |
| TT の初期状態 | 空 | PNS 由来の証明/反証が存在 |
| 合駒プレフィルタ | 浅い反復の蓄積を待つ必要あり | PNS 由来の証明で即座に機能 |
| グローバル最適性 | 各反復は depth-first(局所的) | PNS が大域的に最有望ノードを選択 |
| メモリ消費 | TT のみ | TT + アリーナ(最大 5M ノード) |

29手詰めテスト (`test_tsume_6_29te_no_pns`) では PNS なしの IDS のみでも
解けることが確認されているが，これは IDS-MID 自体のロバストネスを示すものであり，
PNS の寄与が不要であることを意味しない．
一般に PNS は浅い証明の高速発見と TT のウォームアップに寄与し，
特にチェーン合駒問題(§8)では PNS 由来の証明が
プレフィルタ(§8.4)のヒット率を大幅に向上させる．

### 2.6 IDS フルデプスステップ: MID + Frontier Variant

IDS の最終反復(フルデプス)では，MID 先行 + Frontier フォールバックの
2段構成を採用する．

```mermaid
flowchart TD
    IDS[IDS フルデプス反復] --> MID[MID: 1/4 チャンク × 最大4回]
    MID --> TTchk{TT 成長?}
    TTchk -->|yes| MID
    TTchk -->|no / 予算上限| C1{証明?}
    C1 -->|yes| Done[完了]
    C1 -->|no| FV[Frontier Variant: 残り予算]
    FV --> PNS[PNS: フロンティア特定]
    PNS --> MID2[MID: 局所探索]
    MID2 --> RP[retain_proofs\n中間エントリ除去]
    RP -->|未解決 & 予算残| PNS
    MID2 -->|証明| Done
    RP -->|予算終了| Unknown[UNKNOWN]
```

#### 設計根拠

MID を先行させる理由: 閾値飢餓が発生しない部分木は MID が効率的に処理する
(NPS が Frontier の ~1.6 倍)．

v0.21.0 では動的予算配分(§10.2 方針B)を導入: MID の最大予算(全体の 1/2)を
1/4 チャンクに分割し，各チャンク後に TT エントリ数の成長を確認する．
TT が成長しなくなった時点で閾値飢餓による停滞と判定し，
残り予算を Frontier に動的移行する．これにより MID が進捗可能な範囲を
効率的に処理した後，速やかに Frontier の閾値飢餓回避機構に切り替える．

#### MID → Frontier 遷移

MID と Frontier 間で **TT 清掃は行わない**．
MID が蓄積した証明・反証・中間エントリを Frontier がそのまま活用する．
これは Phase 1 → Phase 2 の連携(§2.5)とは異なる設計判断である:

- **Phase 1 → Phase 2**: `retain_proofs_only()`(証明のみ保持)．
  PNS の中間値は best-first 評価であり，depth-first の MID とは閾値体系が異なる
- **MID → Frontier**: 清掃なし．MID 直後の中間値は最新の探索状態を反映しており，
  Frontier 内の PNS/MID で再利用可能

#### Frontier サイクル内 TT 清掃

Frontier Variant の各 PNS→MID サイクル間では `retain_proofs()` を実行する．
MID が各サイクルで蓄積した中間エントリが次の PNS サイクルの
フロンティア選択を汚染するのを防止する．
証明(pn=0)と確定反証(dn=0, 非経路依存)は保持される．

---

## 3. 閾値制御

### 3.1 1+ε トリック (Pawlewicz & Lew 2007)

**出典:** Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)

標準 df-pn では子 c1 の pn 閾値を `min(parent_th, pn2 + 1)` で設定する
(`pn2` は2番目に小さい pn)．c1 の pn が pn2 を1超えた瞬間に
他の子に切り替わり，seesaw effect(スラッシング)が発生する．

```
  Standard df-pn threshold (seesaw effect):

  OR node (pn_th=100)
   |
   +-- c1: pn=10  <-- selected (min pn)
   +-- c2: pn=11  (pn2 = 11)
   |
   child_pn_th = min(100, 11+1) = 12
   --> c1 explores until pn reaches 12, then switches to c2
   --> c2 explores until pn reaches 13, then back to c1
   --> rapid switching = thrashing

  1+epsilon trick:

  OR node (pn_th=100)
   |
   +-- c1: pn=10  <-- selected
   +-- c2: pn=11  (pn2 = 11)
   |
   epsilon = 11/4 + 1 = 3
   child_pn_th = min(100, 11+3) = 14
   --> c1 gets 4 more units of exploration before switching
   --> deeper search per visit, less thrashing
```

1+ε トリックは `+1` を乗算型に変更:

```
pn_threshold(c1) = min(parent_th, ceil(pn2 * (1 + ε)))
```

pn が小さい時は小さな増分(細かい制御)，pn が大きい時は大きな増分(深い探索を許容)．

**実装:** solver.rs (child threshold computation)

```
// v0.21.0: 自然精度 epsilon (§3.5 方針A)
epsilon = second_best / 4 + PN_UNIT
sibling_based = second_best + epsilon ≈ second_best * 5/4 + PN_UNIT
```

OR ノード: `child_pn_th = min(eff_pn_th, second_best + epsilon)`
AND ノード: `child_dn_th = min(eff_dn_th, second_best + epsilon)`

PN\_UNIT=16 では `second_best = 3S = 48` のとき `epsilon = 12 + 16 = 28`，
`sibling_based = 76`(4.75S)となり，PN\_UNIT=1 の 4S に対し ~19% の閾値余裕が
各 OR/AND レベルで得られる．12 レベルの累積で `1.19^12 ≈ 8 倍` の余裕．

**出典との差異:**
- 論文は `ceil(pn2 * (1 + ε))` (純粋な乗算)だが，maou\_shogi では
  `second_best + second_best / 4 + PN_UNIT` で乗算を近似
- `min` キャップは論文どおり適用し，全域で乗算型の性質を維持

### 3.2 TCA (Kishimoto & Müller 2008; Kishimoto 2010)

**出典:** Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008);
Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation" (AAAI 2010)

巡回グラフ(DCG)上での pn/dn **過小評価**を修正するアルゴリズム．
ループ検出により子ノードが `(INF, 0)` を返すと，兄弟ノードの pn/dn が過小評価される．
TCA は OR ノードでループ子が存在する場合に閾値を拡張し，兄弟の深い探索を促す．

df-pn は有限 DCG 上で不完全だが，TCA を加えると完全になる．

**実装:** mod.rs (`TCA_EXTEND_DENOM`), solver.rs (MID ループ)

- **拡張量**: `threshold / TCA_EXTEND_DENOM + 1` (`TCA_EXTEND_DENOM = 4`，約25%の加算)
- **適用条件**: OR ノードでループ子(`path` 上の子)が存在する場合
- AND ノードではループ子が即時反証を引き起こすため拡張不要

**出典との差異:**
- 論文は乗算的拡張(2×)を提案するが，再帰で指数的に増大する問題がある
- maou_shogi では加算的拡張(約25%)を採用し，各レベルで独立に適用されるため膨張を抑制

### 3.3 閾値フロア

MID ループ内で閾値が過度に縮小するのを防ぐフロア値を設定する．

**実装:** solver.rs (child threshold computation)

- **PN フロア(通常)**: `pn_floor = eff_pn_th / 2`
- **PN フロア(チェーン AND)**: `pn_floor = max(DN_FLOOR, eff_pn_th / 2)` (v0.20.35)
- **DN フロア(OR)**: `dn_floor_or = DN_FLOOR`
- **DN フロア(通常)**: `dn_floor = DN_FLOOR`

`DN_FLOOR = 100 * PN_UNIT`（§3.5 参照）．

チェーン合駒構造では閾値が深いネストで指数的に枯渇するため，
フロアにより最低限の探索予算を保証する．

チェーン AND ノードでは DN_FLOOR(=100) を PN フロアにも適用し，
OR 親の sibling_based(2〜5)に制約されず子 OR に十分な pn 予算を
伝播する(dn のチェーン用キャップ外しと同じ発想)．
backward 解析で ply 24 サブ問題が 1M→397K ノードに改善(§10.2)．

### 3.4 停滞検出

MID ループ内で pn/dn が改善しない場合に早期終了する．

**実装:** solver.rs (`ZERO_PROGRESS_LIMIT`, `STAGNATION_LIMIT`)

- `ZERO_PROGRESS_LIMIT = 16`: 子 `mid()` が消費するノード数が 0 の回数が連続16回で進展なしと判定
- `STAGNATION_LIMIT = 4`: best child の pn/dn と閾値が連続4回不変で MID ループを終了

### 3.5 PN\_UNIT 統一スケーリング

pn/dn の 1 単位を定数 `PN_UNIT`(mod.rs)で表現し，全てのスケーリング対象を
明示する仕組み．PN\_UNIT=16(v0.21.0)で閾値飢餓を緩和し，
1+ε 閾値の余裕を確保する（KomoringHeights の初期 pn=10-80 に相当）．
PN\_UNIT=1 で従来動作と等価であり，スケーリング漏れの検証に使用する．

**設計原理:**

全ての pn/dn を完全にスケーリングすればソルバーの挙動は一致する．
逆に言えば，PN\_UNIT を変更して挙動が変わるならスケーリング漏れがある．
この原理を用いて PN\_UNIT=1 と PN\_UNIT=64 の結果を比較し，
漏れを機械的に特定・修正した．

**スケーリング対象:**

| 区分 | 具体例 |
|------|--------|
| 初期値 | TT ミスの pn=1/dn=1，heuristic\_or\_pn/heuristic\_and\_pn 返り値 |
| 加算定数 | edge\_cost\_or/and，sacrifice\_check\_boost，epsilon の +1，progress\_floor の +1，TCA の +1 |
| フロア・バイアス | DN\_FLOOR，INTERPOSE\_DN\_BIAS，`.max(N)` のリテラル |
| WPN の加算分 | `(unproven_count - 1) * PN_UNIT`（盤面カウントを pn 単位に変換） |
| TT ミス判定 | `cpn == PN_UNIT && cdn == PN_UNIT`（heuristic 初期化の条件） |

**スケーリング不要:**

| 区分 | 理由 |
|------|------|
| 終端値 (INF, 0) | 証明/反証のセンチネル |
| 盤面状態の比較 (safe\_escapes >= 4 等) | 手数・マス数であり pn/dn 値ではない |
| ループカウンタ (ZERO\_PROGRESS\_LIMIT 等) | イテレーション回数 |

**除算の丸め等価性:**

除算を含む計算は「PN\_UNIT=1 相当に戻してから除算し再スケール」する
(divide-at-unit-scale パターン):

```
// 等価パターン: PN_UNIT=1 と同じ丸めを再現(スケーリング漏れ検証用)
let epsilon = second_best / PN_UNIT / 4 * PN_UNIT + PN_UNIT;
```

例: `second_best = 3 * PN_UNIT` のとき
- PN\_UNIT=1: `3 / 4 + 1 = 0 + 1 = 1`
- PN\_UNIT=64 (等価): `192 / 64 / 4 * 64 + 64 = 0 + 64 = 64` (= 1 × 64) ✓

適用箇所: epsilon (`/4`)，pn\_floor (`/2`)，TCA (`/TCA_EXTEND_DENOM`)，
Deep df-pn (`/DEEP_DFPN_R`)．

**自然精度パターン:**

divide-at-unit-scale はスケーリング漏れの検証には不可欠だが，
PN\_UNIT > 1 の本来の利点である**除算の解像度向上**を殺してしまう．
閾値飢餓の改善には，除算の自然精度をそのまま活かすパターンが有効:

```
// 自然精度パターン (v0.21.0 本番): PN_UNIT > 1 で epsilon が増大する
let epsilon = second_best / 4 + PN_UNIT;
```

例: `second_best = 3 * PN_UNIT` のとき

| | PN\_UNIT=1 | PN\_UNIT=16 (等価) | PN\_UNIT=16 (自然精度) |
|--|----------|------------------|---------------------|
| second\_best | 3 | 48 | 48 |
| epsilon | 1 | 16 | 12 + 16 = 28 |
| sibling\_based | 4 | 64 | 76 |
| PN\_UNIT 単位 | 4.0 | 4.0 | **4.75** |

自然精度では PN\_UNIT=1 の整数切り捨て `3/4 = 0` が
`48/4 = 12` として正確に計算され，epsilon が 75% 増加する．
これにより子 AND に渡る pn 閾値が増大し，閾値飢餓が緩和される．

heuristic\_or\_pn が S〜3S の範囲で中間値(例: 1.5S)を返す場合，
second\_best の分布がさらに広がり，自然精度の恩恵が増す．

**使い分け:**

- **等価パターン**: スケーリング漏れの検証，回帰テスト
- **自然精度パターン**: 閾値飢餓の改善（本番運用）

**pn/dn 値の全体マップ (PN\_UNIT = S):**

全ての pn/dn 初期値・バイアス・フロアの相対関係を示す．
S = PN\_UNIT（v0.21.0: 16）．

*初期値(子ノード展開時に TT に格納される値):*

| 値 | S 倍率 | 適用対象 | 条件 | 節 |
|---|-------|---------|------|---|
| S | 1 | OR 子 pn | 標準局面，逃げ場 0〜1 | §5.1 |
| S + S/4 | 1.25 | OR 子 pn | 逃げ場=2 | §5.1 |
| S + S/4 〜 S + S/2 | 1.25〜1.5 | OR 子 pn | 逃げ場 4〜5 | §5.1 |
| 2S + S/4 〜 3S | 2.25〜3 | OR 子 pn | 王手少＋逃げ場多，開放空間 | §5.1 |
| n×S + S/4 〜 n×S + e×S/2 | 〜n+e/2 | AND 子 pn | 応手数 n，逃げ場 e(0.67n〜n+e/2 に調整) | §5.1 |
| S | 1 | AND/OR 子 dn | 全子共通 | — |

*加算コスト(初期値に上乗せ):*

| 値 | S 倍率 | 適用対象 | 条件 | 節 |
|---|-------|---------|------|---|
| 0 | 0 | pn 加算 | 成・取王手(edge\_cost\_or) / 合駒(edge\_cost\_and) | §5.2 |
| S | 1 | pn 加算 | 近い静か王手(距離≤2) / 玉逃げ | §5.2 |
| 2S | 2 | pn 加算 | 遠い静か王手(距離≥3) / 駒取り応手 / 全捨て駒 | §5.2, §9.3 |

*閾値制御パラメータ:*

| 値 | S 倍率 | 用途 | 節 |
|---|-------|------|---|
| ε ≈ second\_best/4 + S | ~1.25倍 | OR/AND の 1+ε 手切替 | §3.1 |
| pn\_floor = eff\_pn\_th/2 | 親の50% | AND 子 pn 閾値の最低保証 | §3.3 |
| pn\_floor(チェーン AND) = 100S | 100 | チェーン AND 子 pn 閾値の最低保証 | §3.3 |
| DN\_FLOOR = 100S | 100 | AND 子 dn / OR 子 dn の最低保証 | §3.3 |
| progress\_floor = best\_pn + S | +1 | 子 pn 閾値のゼロ進捗防止 | §3.3 |
| TCA 拡張 ≈ threshold/4 + S | +25% | ループ検出時の閾値拡張 | §3.2 |

*dn バイアス(AND ノードの応手選択順序):*

| 値 | S 倍率 | 適用対象 | 節 |
|---|-------|---------|---|
| 0 | 0 | チェーン AND: 内側ドロップ(距離1) | §8.7 |
| (d−1)×S | 1〜5+ | チェーン AND: 外側ドロップ(距離 d) | §8.8 |
| 8S | 8 | 非チェーン AND: ドロップ(合駒後回し) | §8.6 |
| 8S | 8 | チェーン AND: 非ドロップ(玉逃げ後回し) | §8.6 |

*WPN/CD-WPN の加算分(AND の current\_pn 計算):*

| 値 | S 倍率 | 用途 | 節 |
|---|-------|------|---|
| (n−1)×S | n−1 | WPN: 未証明子 n 個の加算分 | §4.1 |
| (g−1)×S | g−1 | CD-WPN: グループ数 g の加算分 | §4.2 |

*Deep df-pn バイアス(TT ミス時の深い ply):*

| 値 | S 倍率 | 条件 | 節 |
|---|-------|------|---|
| S | 1 | ply ≤ depth/2 | §5.3 |
| S + ⌊(ply − depth/2)/4⌋×S | 1〜数倍 | ply > depth/2 | §5.3 |

**相対関係の読み方:**

OR 子 pn(1S〜3S)と AND 子 dn(1S)の比が探索の OR/AND バランスを決める．
DN\_FLOOR(100S)は OR 子 pn の 33〜100 倍であり，
dn 閾値が枯渇しにくいことを保証する一方，pn 側にはこの水準のフロアがない
（チェーン AND を除く）．これが閾値飢餓の構造的要因である(§10.2)．

INTERPOSE\_DN\_BIAS(8S)は OR 子 pn(1〜3S)の 3〜8 倍であり，
合駒を玉逃げ・駒取りの後に探索させる効果が十分に働いている．

PN\_UNIT を拡大すると，上記の全ての値が比例してスケールされる．
改善の余地は「S 倍率が整数に丸められている箇所」にあり，
PN\_UNIT > 1 で中間値(1.5S 等)を設定することで
heuristic の解像度を上げられる(§10.2 方針 A)．

**検証結果:**

PN\_UNIT=1 と PN\_UNIT=64 で 126 テスト全通過（pass/fail 完全一致）．
backward 解析では ply 24 まで完全一致(396,636 ノード，343,999 TT エントリ)．
ply 22 以降の微差(TT 7 エントリ = 0.005%)は予算上限到達後の
TT クラスタ衝突パターンのみ．

**スケーリング漏れの特定に至った経緯:**

| 発見した漏れ | 症状 | 特定方法 |
|------------|------|---------|
| WPN `(unproven_count - 1)` に PN\_UNIT 未適用 | PN\_UNIT=64 で 2 テスト FAIL | テスト結果の比較 |
| TT ミス判定 `cpn == 1` | 同上 | 同上 |
| depth 制限超過時の初期 pn=1u32 | ply 24 以降で探索パターン乖離 | backward 解析の diff |
| 除算の丸め精度差 | P\*4g で 1.5%のノード数差 | 4M クラスタ TT での比較 |

---

## 4. 証明数・反証数の計算

### 4.1 WPN: Weak Proof Number (Ueda et al. 2008)

**出典:** Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)

証明数の**二重計数問題**(double-counting problem)に対処する手法．
DAG 構造の探索木において，共有ノードが複数の親から重複してカウントされ
証明数が過大評価される問題を，分岐係数を組み込んだ推定量で解決する．

```
  Standard AND node             WPN AND node
  pn = sum(child_pn)            pn = max(child_pn) + (count - 1)

      AND                           AND
     / | \                         / | \
   OR  OR  OR                    OR  OR  OR
  pn=3 pn=5 pn=2               pn=3 pn=5 pn=2

  pn = 3+5+2 = 10              pn = max(3,5,2) + (3-1) = 7
```

標準: `pn(AND) = sum(child_pn)`
WPN: `pn(AND) = max(child_pn) + (unproven_count - 1)`

**実装:** solver.rs (AND ノード collect)

AND ノードの pn 合計を `max(cpn) + (unproven_count - 1)` で計算．
VPN (§4.3) による証明済み子の除外，SNDA (§4.4) による DAG 合流補正と併用．

**出典との差異:**
- 論文は OR/AND 両ノードに WPN を適用するが，maou_shogi では AND ノードのみに適用
- SNDA との併用時に過剰補正が発生する問題を v0.20.24 で修正:
  SNDA 控除後の pn が `max(child_pn)` を下回らないようフロアを設定

### 4.2 CD-WPN: Chain-Drop Weak Proof Number

**出典:** maou 独自手法

チェーン合駒(§8)に特化した WPN の変種．
チェーン合駒では同一マスへの異なる駒種の drop が子ノードとなるが，
これらは同一マスへの合駒として意味的にグループ化できる．

CD-WPN はドロップを `to_sq` でグループ化し，グループ数を `unproven_count` とする:

```
grouped_count = チェーン合駒の到達マス数(駒種ではなくマス数)
pn(AND) = max(child_pn) + (grouped_count - 1)
```

**実装:** solver.rs (AND ノード collect)

- `chain_king_sq` が `Some` の場合(チェーン AND ノード)に CD-WPN を適用
- `chain_king_sq` が `None` の場合は標準 WPN を使用

### 4.3 VPN: Virtual Proof Number (Saito et al. 2006)

**出典:** Saito et al. 2006

AND ノードの pn 計算で証明済み子(cpn=0)を除外する．
証明済み子は pn=0 で sum に影響しないが，子選択ループからのスキップにより
SNDA ペア収集と子選択の効率化に寄与する．

**実装:** solver.rs (AND ノード collect)

AND ノードの子収集ループで `cpn == 0` の子を `continue` で除外．

### 4.4 SNDA: Source Node Detection Algorithm (Kishimoto 2010)

**出典:** Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation" (AAAI 2010)

DAG(転置)による pn/dn の**過大評価**を検出・修正する．
同一のリーフノードが複数の子を通じて重複カウントされる場合，
source ハッシュに基づくグループ化で重複分を控除する．

```
  Without SNDA (overcounting):     With SNDA (corrected):

      OR (dn = 3+5 = 8)               OR (dn = max(3,5) = 5)
     / \                              / \
   AND  AND                         AND  AND
   dn=3 dn=5                       dn=3 dn=5
     \  /                            \  /
      \/                              \/
     LEAF  <-- same source           LEAF  <-- grouped by source
     dn=?                           deduction = (3+5) - max(3,5) = 3
                                    corrected dn = 8 - 3 = 5
```

**実装:** mod.rs (`snda_dedup`), solver.rs (OR/AND collect)

TT エントリに `source: u64` フィールドを追加．
`(source, value)` ペアをソートし，同一 source グループで:

```
deduction = sum(group) - max(group)
```

控除後: `pn' = raw_sum - total_deduction` (最低値 1)

- OR ノード: `(source, dn)` ペアで dn を補正
- AND ノード: `(source, pn)` ペアで pn を補正

**出典との差異:**
- 論文は親ポインタ追跡による共通祖先検出を提案するが，
  maou_shogi では source ハッシュ(リーフ位置キー)によるグループ化で近似
- 積極的 max 集約方式を採用: グループ内で最大値のみを残す
  (保守的方式 v0.11.0 → 積極的方式 v0.15.0 に移行)
- AND ノードでの SNDA + WPN 併用時の過剰補正を v0.20.24 で修正:
  SNDA 控除後の pn が `max(child_pn)` を下回らないようにクランプ

---

## 5. 初期値ヒューリスティック

### 5.1 df-pn+ ヒューリスティック初期化 (GPW 2004; KomoringHeights v0.4.0)

**出典:** Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004);
KomoringHeights v0.4.0

標準 df-pn は全リーフを `(pn=1, dn=1)` で初期化するが，
df-pn+ では局面の特徴に基づいて初期 pn/dn を設定する．
玉の逃げ場が少ない局面ほど pn を小さく(詰みやすい)，
王手手段が多い局面ほど dn を大きく(反証しにくい)する．

**実装:**

#### `heuristic_or_pn` (solver.rs)

OR 子(攻め方局面)の初期 pn．王手数と玉の安全な逃げ場で調整:

| 条件 | 初期 pn (S = PN\_UNIT) | v0.20.x 互換 |
|------|----------------------|-------------|
| 逃げ場なし | S | S |
| 逃げ場=1 | S | S |
| 逃げ場=2 | S + S/4 | S |
| 王手≤2 かつ 逃げ場=3 | 2S + S/4 | 2S |
| 王手≤2 かつ 逃げ場=4 | 2S + S/2 | 3S |
| 王手≤2 かつ 逃げ場≥5 | 3S (キャップ) | 3S |
| 逃げ場=4 | S + S/4 | S + S/3 |
| 逃げ場=5 | S + S/2 | S + S/3 |
| 逃げ場≥4，隣接≥5，圧迫0 | 3S (開放空間) | 3S |

v0.21.0 で PN\_UNIT=16 による高解像度化を実施(§10.2 方針A)．
safe\_escapes に応じた連続スケーリングにより，従来の 3 段階(S, 2S, 3S)から
S/4 刻みの 7 段階に解像度が向上し，second\_best の分布が広がることで
自然精度 epsilon(§3.5)の恩恵が増加する．

#### `heuristic_and_pn` (solver.rs)

AND 子(守備方局面)の初期 pn．応手数と玉の安全な逃げ場で調整:

| 条件 | 初期 pn (S = PN\_UNIT) | v0.20.x 互換 |
|------|----------------------|-------------|
| 逃げ場なし | `n * 2/3 * S` | `n * 2/3 * S` |
| 逃げ場=1 | `n * S + S/4` | `n * S` |
| 逃げ場=2 | `n * S + S/2` | `n * S` |
| 逃げ場=3 | `n * S + 3S/2` | `(n+1) * S` |
| 逃げ場≥4 | `n * S + e*S/2 + S/4` | `(n + e/2) * S` |

n = num\_defenses, e = safe\_escapes．v0.21.0 で逃げ場 1〜2 にも
中間値を返すことで閾値配分の精度を向上させた．

### 5.2 DFPN-E エッジコスト型 (NeurIPS 2019)

**出典:** "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)

リーフ(ノード)ではなくエッジ(親→子遷移の手)にヒューリスティックコストを付与する．
展開済みノードではエッジコストがゼロになるため，実質的には初期 pn への加算として機能する．

**実装:** mod.rs

#### `edge_cost_or` (OR ノードの王手): mod.rs

| 手の種類 | コスト |
|---------|--------|
| 成王手 / 取王手 | 0 (最有力) |
| 近い静か王手 (距離≤2) | 1 |
| 遠い静か王手 (距離≥3) | 2 |

#### `edge_cost_and` (AND ノードの応手): mod.rs

| 応手の種類 | コスト |
|-----------|--------|
| 合駒 (drop) | 0 (攻め方が取り進んで有利) |
| 玉の逃げ / 駒移動 | 1 |
| 駒取り | 2 (攻め駒除去で攻め方不利) |

**出典との差異:**
- 論文のコスト関数はドメイン非依存の汎用設計だが，
  maou_shogi では将棋の詰みに特化したドメイン知識(成/取/距離/合駒)を組み込み

### 5.3 Deep df-pn (Song Zhang et al. 2017)

**出典:** Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)

深い位置ほど初期 dn を高く設定し，浅い解を優先する．
論文推奨値: `dn_init = max(1, ceil(R * depth))` (R=0.4, Othello/Hex)．

**実装:** mod.rs (`DEEP_DFPN_R`), solver.rs (`look_up_pn_dn`)

TT ミス時(pn=1, dn=1, source=0)に深さバイアスを適用:

```
if ply > depth / 2:
    biased_pn = 1 + (ply - depth/2) / DEEP_DFPN_R    (DEEP_DFPN_R = 4)
```

浅い ply (depth の前半) は標準 df-pn と同じ pn=1 を維持．

**出典との差異:**
- 論文は dn にバイアスを適用するが，maou_shogi では **pn にバイアス**を適用
- 論文の R=0.4 (小さいほど積極的) に対し，maou_shogi では `R=4` (整数除算)
- 深い ply の未探索子の pn を上げることで，探索済みの浅い子を優先する効果
- バイアス適用は depth の後半のみ(前半は標準 pn=1 で不詰検出を維持)

### 5.4 インライン詰み検出

child_init フェーズ(子ノードの TT 初回参照時)で，
MID の再帰呼び出しなしに1手・3手の詰み/不詰を即座に判定する．

**実装:** solver.rs (child init)

#### AND 子ノード(OR 局面)の検出: solver.rs (child init, `or_node` ブランチ)

1. `generate_defense_moves(board)` で全応手を生成
2. 応手なし → 即詰み確定(pn=0, dn=INF)
3. `ply + 2 < depth` なら3手詰め判定:
   - 各応手を実行し `has_mate_in_1_with(board, checks)` で全応手に1手詰みがあるか確認
   - 全応手に対して1手詰みが存在 → 即詰み(pn=0)

#### OR 子ノード(AND 局面)の検出: solver.rs (child init, `!or_node` ブランチ)

1. `generate_check_moves(board)` で全王手を生成
2. 王手なし → 即不詰(pn=INF, dn=0)
3. `ply + 2 < depth` なら:
   - `has_mate_in_1_with(board, checks)` で1手詰み判定
   - `try_capture_tt_proof(board, checks, remaining)` で TT 参照の即証明

#### `has_mate_in_1_with` ヘルパー: solver.rs

`board.mate_move_in_1ply(checks, us)` で1手詰みを検出．
詰み発見時は詰み局面を TT に記録し，将来の探索で再利用可能にする．

**設計判断:** 5手以上のインライン検出は MID の枝刈り(閾値制御・TT 参照)なしの
網羅探索となり，MID 自体より非効率になるため実装しない．
過去に実装した budget 付き N 手詰め検出(static_mate)は TT 汚染と
探索効率の悪化を招いたため v0.20.24 で削除した．

---

## 6. 転置表管理

### 6.1 持ち駒優越 (Nagai 2002)

**出典:** Nagai 2002

盤面が同一で持ち駒が異なる局面間の包含関係を利用した TT 再利用:

```
  Proof reuse (pn=0):              Disproof reuse (dn=0):

  TT: hand={P,G} -> pn=0          TT: hand={P,P,G} -> dn=0

  query: hand={P,P,G}             query: hand={P}
  {P,P,G} >= {P,G} ? YES          {P} <= {P,P,G} ? YES
  -> reuse proof                   -> reuse disproof
  (more pieces = easier to mate)  (fewer pieces = harder to mate)
```

- **証明(pn=0)**: 攻め方の持ち駒が TT エントリ以上 → 再利用可(持ち駒が多いほど詰ませやすい)
- **反証(dn=0)**: 攻め方の持ち駒が TT エントリ以下 → 再利用可(持ち駒が少ないほど詰ませにくい)

**実装:** mod.rs (`hand_gte`), tt.rs (`look_up`)

- TT キー: `position_key(board)` = 盤面ハッシュ(持ち駒を**含まない**)
- TT 値: 同一クラスタ内に同一 `pos_key` の複数エントリを保持(§6.6)
- Lookup 時: クラスタ内で証明エントリを先に走査(証明優先)，その後反証エントリを走査

### 6.2 前方チェーン補填 (maou 独自)

**出典:** maou 独自手法

持ち駒優越の拡張として，歩 ≤ 香 ≤ 飛のカスケード補填を実装する．
チェーン合駒の文脈で，攻め方が合駒を取った後の持ち駒構成が異なっても，
前方利き系の駒種間で代替関係を認める．

**実装:** mod.rs (`hand_gte_forward_chain`)

代替関係:
- 歩の不足 → 香で代替可能
- 香の不足 → 飛で代替可能
- 歩の不足 → 飛で代替可能(カスケード)

桂・銀・金・角は独立判定(利きの方向が異なるため代替不可)．

### 6.3 Pareto Frontier 管理

**出典:** Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)

同一盤面に対する複数の TT エントリを Pareto frontier で管理する．
持ち駒とエントリの支配関係に基づき，冗長なエントリを排除する．

**実装:** tt.rs (`store_impl`)

- **最大エントリ数**: `CLUSTER_SIZE = 6`(v0.20.34〜，旧 `MAX_TT_ENTRIES_PER_POSITION = 16`)
- **証明エントリ(pn=0)**: 最小持ち駒のエントリを保持(少ない持ち駒で証明できるほど汎用的)
- **反証エントリ(dn=0)**: 最大持ち駒のエントリを保持(多い持ち駒で反証できるほど汎用的)
- **支配判定**: `hand_gte_forward_chain` (§6.2) による拡張支配関係を使用
- **容量超過時**: 異なる `pos_key` のエントリを優先的に置換．
  証明/確定反証を保護しつつ，`|pn - dn|` が最小の中間エントリを犠牲にする

#### 反証挿入時の中間エントリ除去

フラットテーブルでは反証(dn=0)エントリ挿入時に，
同一 `pos_key` の中間エントリ(pn>0, dn>0)を積極的に除去してスロットを確保する．
旧 HashMap 版では Vec に追加するだけで中間エントリは保護されていた
(`remaining` の不一致で将来必要になりうるため)．

フラットテーブルではクラスタの 6 スロットが厳密な上限であるため，
反証の挿入を保証するには中間エントリの除去が不可避である．
中間エントリは探索の再訪時に再計算可能であり，
反証(確定結果)の保持を優先する設計判断である．

#### 置換スコア `|pn - dn|` の設計根拠

中間エントリの置換優先度にはスコア `|pn - dn|` を使用する．
この値が小さいエントリ(pn ≈ dn)は証明にも反証にも近くない「均衡状態」であり，
探索方向が未確定のため情報価値が相対的に低い．
一方，`|pn - dn|` が大きいエントリは証明(pn << dn)または反証(pn >> dn)に
偏っており，探索の進行方向を示す有用な情報を持つ．

**既知の限界:**

`pn = dn = 100,000`（深く探索済みの均衡局面）と `pn = dn = 1`（新規エントリ）は
同一スコア 0 となり，深い探索結果が新規エントリと同等に退避される．
探索量ベースのスコア(例: `pn + dn`)を組み合わせることで改善できる可能性があるが，
以下の理由で現時点では採用しない:

1. 1M クラスタでの TT 利用率は ~18%(10M ノード探索)であり，
   置換の影響は限定的
2. `pn + dn` スコアでは `pn=1, dn=1` の新規エントリが最優先で退避され，
   IDS の浅い反復で蓄積した初期値が過保護になるリスクがある
3. 置換ポリシーの変更は探索パターン全体に波及し，退行テストが必要

### 6.4 TT ガベージコレクション

**実装:** tt.rs (`gc`, `gc_shallow_entries`), solver.rs (periodic GC)

- **周期的 GC**: 100K ノードごとにサイズチェック，閾値超過時に容量75%まで回収
- **IDS 間清掃**:
  - `retain_proofs()`: 証明エントリのみを保持
  - `gc_shallow_entries()`: 浅い remaining のエントリを除去
  - `remove_stale_for_ids()`: remaining=0 の反証を除去

### 6.5 TT Best Move 保存 (KomoringHeights v0.4.0)

**出典:** KomoringHeights v0.4.0

TT エントリに最善手(`best_move: u16`)を保存し，動的手順改善(§9.1)に使用する．

**実装:** entry.rs (`DfPnEntry`), tt.rs (`store_with_best_move`, `look_up_best_move`)

- `store_with_best_move`: MID ループの中間結果保存時に最善子の Move16 を記録
- `look_up_best_move`: TT ヒット時に最善手を取得し，手順の先頭にスワップ

### 6.6 TT データ構造

**実装:** `entry.rs` (`DfPnEntry`), `tt.rs` (`TranspositionTable`)

```rust
#[repr(C)]
struct DfPnEntry {
    source: u64,               // SNDA ソースハッシュ (§4.4)
    pn: u32,                   // 証明数
    dn: u32,                   // 反証数
    hand: [u8; HAND_KINDS],   // 攻め方の持ち駒 (7 bytes)
    path_dependent: bool,      // GHI フラグ (§7.1)
    remaining: u16,            // 深さ制約 (0..depth or REMAINING_INFINITE)
    best_move: u16,            // 最善手 (Move16 エンコーディング)
    amount: u16,               // 探索投資量 (GC/置換の保護優先度)
}  // 32 bytes (#[repr(C)] + field order optimization)
```

**TT 全体構造:** フラットハッシュテーブル (v0.20.34 〜)

v0.20.34 で `FxHashMap<u64, Vec<DfPnEntry>>` から固定サイズのフラット配列に置換(§11.4)．
`CLUSTER_SIZE = 6` エントリ/クラスタ，デフォルト 2M クラスタ(≈ 480 MB)．
`pos_key & (num_clusters - 1)` によるダイレクトインデクシングで O(1) アクセス．

```
TTFlatEntry = { pos_key: u64, entry: DfPnEntry }  // 40 bytes
Cluster     = [TTFlatEntry; 6]                     // 240 bytes
Table       = Vec<Cluster>                         // 2M clusters ≈ 480 MB
```

**置換ポリシー:** クラスタ満杯時は異なる `pos_key` のエントリを優先的に置換し，
同一 `pos_key` の証明(pn=0)・確定反証(dn=0, REMAINING\_INFINITE)は保護する．
パレートフロンティア管理(§6.3)，前方チェーン比較(§6.2)，
経路依存フラグ(§7.1)のセマンティクスは完全に維持．

反証エントリの挿入時，クラスタが foreign protected エントリ(別 pos\_key の
proof/confirmed disproof)で埋まっている場合，`replace_weakest_for_disproof`
により foreign depth-limited disproof → foreign confirmed disproof → foreign proof
の優先順で1スロットを犠牲にして挿入する(v0.21.1)．

#### クラスタ衝突と暗黙的置換

フラットハッシュテーブルでは TT GC(§6.4)とは独立に，
**ハッシュ衝突による暗黙的置換**が発生する．

異なる `pos_key` が `pos_key % num_clusters` で同一クラスタにマッピングされると，
6 スロットを複数の局面が共有する．スロットが満杯になると `replace_weakest` が
異なる `pos_key` の中間エントリを上書きする．
これは TT GC のように明示的に発動するのではなく，
`store_impl` の通常動作として常に発生する．

#### クラスタサイズの決定根拠 (v0.21.1)

**CLUSTER\_SIZE = 6 の理由:**

将棋の持ち駒は 7 種(歩・香・桂・銀・金・角・飛)であり，
同一盤面に異なる持ち駒で到達する**転置**(transposition)が頻繁に発生する．
TT は pos\_key(盤面ハッシュ)でクラスタを索引し，hand(持ち駒)で
エントリを区別する．同一 pos\_key に必要な hand バリアント数は:

| 状況 | 必要バリアント数 | 根拠 |
|------|---------------|------|
| 典型的な中盤 | 2-3 | proof + intermediate(1-2 hand) |
| 合駒チェーン | 4-6 | 駒種ごとの合駒で異なる hand |
| depth boundary 付近 | 3-5 | depth-limited NM × 複数 hand |

CLUSTER\_SIZE=4 と CLUSTER\_SIZE=6 の比較(4M クラスタ):

| CLUSTER\_SIZE | TT max (29手詰め) | 結果 |
|--------------|------------------|------|
| 4 | 8.3M | NOT PROVED (120M nodes) |
| **6** | **12.5M** | **SOLVED (109M nodes)** |

CLUSTER\_SIZE=4 では hand バリアントがクラスタに収まらず TT が飽和する．
CLUSTER\_SIZE=6 は hand バリアント(典型3-5) + foreign 衝突分(1-2)で必要十分．

#### クラスタ数の決定根拠 (v0.21.1)

**DEFAULT\_NUM\_CLUSTERS = 2M (1<<21) の理由:**

Poisson 近似による overflow 確率分析:
`N` 個のエントリを `C` 個のクラスタに格納するとき，
各クラスタの期待エントリ数は `λ = N/C`．
クラスタが溢れる(CLUSTER\_SIZE 以上のエントリ)確率は
`P(X >= 6) for X ~ Poisson(λ)`．

| C(クラスタ) | スロット | メモリ | λ @5M | P(overflow) @5M | λ @10M | P(overflow) @10M |
|------------|---------|--------|-------|----------------|--------|-----------------|
| 1M | 6M | 240 MB | 5.0 | 38% | >容量 | — |
| **2M** | **12M** | **480 MB** | **2.5** | **3.5%** | **5.0** | **38%** |
| 4M | 24M | 960 MB | 1.25 | 0.2% | 2.5 | 3.5% |

**実測ベンチマーク(29手詰め no\_pns):**

| クラスタ | TT max | 結果 | ノード | 時間 | メモリ |
|---------|--------|------|--------|------|--------|
| 1M | 3.1M | NOT PROVED | >120M | — | 240 MB |
| **2M** | **6.3M** | **SOLVED** | **59M** | **201s** | **480 MB** |
| 4M | 12.5M | SOLVED | 109M | 556s | 960 MB |

2M クラスタが最速である理由:
- **キャッシュ効率**: 480 MB は L3 キャッシュに近い範囲．960 MB は完全にキャッシュ外
- **NPS 向上**: メモリアクセスの局所性が高く，実効 NPS が向上
- **TT 充足性**: 6.3M エントリは29手詰めの探索空間に十分

1M クラスタでは TT が 3.1M で飽和(ハッシュ衝突限界)し，問題が要求する
~6M の固有エントリを保持できないため解決不能．

**Periodic GC:** TT が capacity の 80% に達した場合に amount ベースの GC を実行:
Phase 1 で amount=0 の中間エントリを除去，Phase 2 で全中間エントリを除去．
ただしクラスタレベルの飽和(全体 52% でも特定クラスタが満杯)では
capacity 閾値に達しないため発動しない(§6.6.1)．

**amount フィールド (v0.22.0):** KomoringHeights の `amount_` に相当する
探索投資量メトリック．中間エントリは更新のたびに +1，proof に +100，
disproof に +25-50 のボーナスを加算．`replace_weakest` は最小 amount の
エントリを優先置換し，探索投資量の大きいエントリを保護する．
フィールド順序の最適化(`#[repr(C)]` + source u64 先頭配置)により
エントリサイズ 40 bytes を維持(amount 追加前と同サイズ)．

#### 6.6.1 クラスタ飽和問題

v0.21.1 の 29 手詰め診断で特定された TT の構造的課題．

**症状:** TT がグローバル容量の 52% しか使用していないにもかかわらず，
新規エントリの挿入が失敗し，探索が停滞する．

**原因:** 将棋の持ち駒転置により同一 `pos_key` に対して複数の hand バリアントが
必要だが，固定サイズクラスタ(6 エントリ)では以下が同時に発生する:

1. **hand バリアント飽和**: 同一盤面に 4-6 種の hand バリアント(proof + disproof +
   intermediate × 複数 hand)がクラスタを占有
2. **foreign protected 占有**: 別の `pos_key` の proof/confirmed disproof が
   クラスタの全スロットを占有し，新規エントリが挿入不能
3. **NM 非伝搬**: depth-limited NM を store しても同一 hand のエントリが
   クラスタに存在しないため look\_up でヒットしない

**影響:** MID ループが同じ子を繰り返し選択し pn/dn が不変のまま
ノードを浪費する(sc\_loop\_hang，MID ループ停滞)．

**現在の対策 (v0.22.0):**

| 対策 | 機構 | 効果 |
|------|------|------|
| `replace_weakest_for_disproof` | foreign protected の段階的犠牲 | NM 挿入可能に |
| amount ベース置換 | 低 amount エントリを優先淘汰 | 高価値エントリの保護 |
| single-child 停滞検出 | SC\_STAGNATION\_LIMIT=4 | 無限ループ防止 |
| MID ループ停滞検出 | pn/dn 不変で脱出 | depth boundary 空振り防止 |
| MID チャンク 1M 固定 | TT 成長チェック頻度向上 | Frontier 早期移行 |

**根本的な限界:** クラスタ方式では hand バリアント数がクラスタサイズを超える
問題に対して構造的に脆弱であり，対症療法(停滞検出)で補っている．

**TT 構造の変遷と設計トレードオフ:**

| 版 | 構造 | NPS | TT 容量 | 問題 |
|----|------|-----|---------|------|
| v0.20.32 | FxHashMap + Vec | ~227K | 14.3M | ヒープ確保 + ポインタ追跡 |
| v0.20.34 | フラットクラスタ (1M×6) | **~868K** | 1.3M | クラスタ飽和 |
| v0.22.0 | フラットクラスタ (2M×6) | ~253K | 6.3M | クラスタ飽和(緩和) |

HashMap→フラット化で NPS が 3.83× 向上した一方，TT の実効容量が
14.3M → 1.3M に激減した．v0.22.0 でクラスタ数を 2M に拡大し実効容量を
6.3M に改善したが，キャッシュ効率の低下で NPS は ~253K に低下．

KomoringHeights はリニアプロービング + amount ベース置換を採用しており，
クラスタサイズの制約がない．ただし maou\_shogi では HashMap/リニアプロービングの
NPS が低い(~227K)ため，クラスタ方式の NPS 優位(~253K-868K)を維持しつつ
クラスタ飽和を緩和する方向で改善を進めている(方針E §10.2)．

#### 6.6.2 NPS 最適化の分析 (v0.22.0)

29 手詰め no\_pns (74.2M ノード) のプロファイル結果:

| 操作 | 時間割合 | 呼出回数 | 平均(ns) | 備考 |
|------|---------|---------|---------|------|
| **child\_init 合計** | **56.3%** | **73M** | **2604** | 子ノード初期化 |
| └ ci\_lookup | 18.2% | 342M | 424 | TT クラスタスキャン |
| └ ci\_fastpath | 11.4% | 342M | 266 | depth limit チェック + store |
| └ ci\_do/undo\_move | 7.6% | 683M | 89 | 盤面状態変更 |
| └ ci\_resolve | 2.5% | 233M | 86 | 初期化後の解決チェック |
| └ ci\_inline | 2.0% | 342M | 45 | TT ミス時 heuristic 計算 |
| **movegen\_check** | **16.1%** | **28M** | **1975** | 王手生成 |
| **movegen\_defense** | **12.6%** | **46M** | **919** | 応手生成 |
| **main\_loop\_collect** | **7.2%** | **93M** | **259** | MID ループの pn/dn 収集 |
| tt\_store | 2.3% | 93M | 82 | TT 書き込み |
| do\_move/undo\_move | 2.7% | 148M | 60 | MID ループの盤面操作 |
| tt\_lookup | 1.3% | 74M | 60 | mid() エントリ時の TT 参照 |

**ボトルネック:** `child_init` が全体の 56% を占める．内訳では `ci_lookup`
(TT クラスタスキャン，424ns/回)と `ci_fastpath`(266ns/回)が支配的．
各 mid() 呼び出しで平均 4.7 個の子ノードに対して do\_move → TT lookup →
heuristic → undo\_move を実行する．

**実施した最適化と結果:**

| 最適化 | 時間 | 改善 | 採否 |
|--------|------|------|------|
| fastpath/lookup 重複排除 + depth limit フラグ化 | 287s | **-3%** | **採用** |
| 1-pass lookup (proof/disproof/exact 統合) | 306s | +7% 悪化 | 不採用 |
| child\_cache 差分更新 (main\_loop\_collect 削減) | >1350s | >4x 悪化 | 不採用 |

1-pass lookup は proof の early return を喪失して悪化．
child\_cache はスタック上の ArrayVec<593> が分岐予測を破壊し NPS が壊滅．

**NPS 改善の限界:** `child_init` の主要コストは do\_move/undo\_move(盤面操作)と
TT クラスタスキャン(hand 比較)であり，これらはアルゴリズムレベルの最適化では
削減困難．大きな NPS 改善には手生成(movegen, 29%)やビットボード操作レベルの
最適化が必要であり，TT 構造や df-pn アルゴリズムの改善とは直交する．

---

## 7. ループ・GHI 対策

### 7.1 経路依存フラグ付き GHI 対策 (Kishimoto & Müller 2004/2005)

**出典:** Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)

GHI (Graph History Interaction) は，同一局面が異なる探索経路で
異なる結果を持つ問題．千日手のような繰り返し検出は経路に依存するため，
ある経路で得た反証が別の経路では無効になりうる．

KomoringHeights は dual TT (base/twin) で経路依存/非依存の不詰を区別する．

**実装:** solver.rs (`path: FxHashSet`，ループ検出，GHI 伝播)

maou_shogi では dual TT の代わりに経路依存フラグ方式を採用:

1. **ループ検出**: `path: FxHashSet<u64>` で現在の探索パス上の全ノードハッシュを保持．
   子ノードが path 上に存在すれば循環と判定し，即座に `(INF, 0)` を返す
2. **経路依存反証**: ループ検出に由来する反証を `path_dependent = true` で TT に保存
3. **IDS 間清掃**: `remove_path_dependent_disproofs()` で経路依存反証を除去し，
   異なる深さの探索で自動的に再評価
4. **Remaining 免除**: 経路依存エントリは remaining チェックをバイパス
   (`e.remaining >= remaining || e.path_dependent`)

**出典との差異:**
- 論文の dual TT 方式ほど完全ではないが，経路依存の反証が TT を
  永続的に汚染する問題を軽減する実用的な妥協案

### 7.2 NM Remaining 伝播

深さ制限に由来する不詰(NM: Non-Mate)の深さ情報を正確に伝播する．

**実装:** mod.rs (`propagate_nm_remaining`)

```
nm_remaining = min(child_remaining + 1, current_remaining)
```

- 子の NM が `REMAINING_INFINITE` なら親も `REMAINING_INFINITE`
- 有限 remaining の NM は深い IDS 反復で再評価される
- `REMAINING_INFINITE = u16::MAX`: 深さ非依存の真の証明/反証

---

## 8. 合駒最適化

チェーン合駒(連続合い駒)は詰将棋ソルバーの主要なボトルネックである．
飛び駒(飛車・角・香)による遠距離王手に対して，玉と飛び駒の間のマスに
駒を打つ(合駒)防御手のうち，飛び駒がその合駒を取り進むことで再び王手となり，
さらに合駒が可能になる再帰的構造を指す．
n マスのチェーンに対して各マスで k 種の合駒が可能な場合，最悪 O(k^n) の分岐が発生する．

### 8.1 Futile/Chain 合駒分類

合駒マス(between squares)を以下の3カテゴリに分類する．

```
  Rook check along rank (e.g. R on 8g checks King on 1g):

  R        between squares (7g..2g)            K
  8g   7g   6g   5g   4g   3g   2g           1g
  [R]--[C]--[C]--[C]--[C]--[N]--[F]--[K]
        ^                   ^    ^
        |                   |    futile: no defender support,
        |                   |            closer to K than breakpoint
        |                   normal (breakpoint):
        |                   defender has support here
        chain: no defender support,
               farther from K than breakpoint,
               R captures -> re-check -> more drops

  [C] = chain     max 3 drops (fwd/diag/knight)
  [N] = normal    all 7 piece types
  [F] = futile    skipped entirely
```

**実装:** `compute_futile_and_chain_squares` (pns.rs)

#### 通常マス (Normal)

守備側の利きが存在する，または玉に隣接し飛び駒が取り進んだ後に逃げ道がある場合．
全駒種(歩→香→桂→銀→金→角→飛)の合駒を生成する．

#### 無駄合いマス (Futile)

以下のすべてを満たすマス:
- 守備側(玉以外)の利きがない
- 玉に隣接していないか，隣接していても取り進み後に逃げ道がない
- ブレークポイント(通常マス)より玉側にある

無駄合いマスへの駒打ちは完全にスキップされる．

#### チェーンマス (Chain)

Futile の条件を満たすが，ブレークポイントより飛び駒側にあるマス．
飛び駒が取り進んだ後に再び王手となり，さらなる合駒が可能な再帰構造を生む．
チェーンマスへの合駒は3カテゴリの代表駒に限定される(§8.2)．

**補助関数:** `king_can_escape_after_slider_capture` (pns.rs)
飛び駒が合駒マスに取り進んだ状態をシミュレートし，玉の逃げ道を判定する．

### 8.2 チェーンドロップ3カテゴリ制限

チェーンマスへの駒打ちを以下の3カテゴリから各1手に制限する．

**実装:** `generate_chain_drops` (pns.rs)

| カテゴリ | 駒種 | 代表の選択 |
|---------|------|----------|
| 前方利き系 | 歩→香→銀→金→飛 | 最弱の合法駒1つ |
| 斜め利き系 | 角 | 角のみ |
| 跳躍系 | 桂 | 桂のみ |

**根拠:** 前方利き系では弱い駒で詰みが証明できれば強い駒でも証明できる
(攻め方が合駒を取った後，手に入る駒が強いほど詰ませやすい)．
角と桂は利きの方向が異なるため独立カテゴリとなる．

**効果:** 合駒マスあたりの駒打ち数を最大7手から最大3手に削減．

### 8.3 合駒遅延展開 (KomoringHeights v0.5.0)

**出典:** KomoringHeights v0.5.0

AND ノードの合駒(駒打ち)を即座に展開せず `deferred_children` に分離する．

```mermaid
flowchart TD
    A[AND node: generate defenses] --> B{classify moves}
    B -->|king moves, supported| C[children: expand immediately]
    B -->|drops = interpose| D[deferred_children: delay]

    C --> E{all children proved?}
    E -->|no: some child disproved| F[AND node disproved - done]
    E -->|yes| G[activate 1 deferred drop]

    G --> H{prefilter: TT proof?}
    H -->|yes| I[skip - already proved]
    H -->|no| J[add to children, search]
    I --> K{more deferred?}
    J --> K
    K -->|yes| G
    K -->|no| L[AND node proved]
```

**実装:** `mid()` 内の子ノード初期化 (`solver.rs`)，PNS の AND ノード展開 (`pns.rs`)

1. AND ノードの子を分類:
   - 駒移動(玉逃げ・紐付き合駒) → `children`(即座に展開)
   - 駒打ち(合駒) → `deferred_children`(遅延)
2. 非合駒応手を先に探索し，TT に証明を蓄積
3. `children` が空になったら `deferred_children` から1手ずつ活性化:
   弱い駒から順に活性化し，証明済み TT エントリを強い駒の探索で援用

**効果:** 非合駒応手で反証できれば合駒の展開自体を回避．
逐次活性化により不要な分岐を抑制．

### 8.4 TT ベース合駒プレフィルタ

合駒を `deferred_children` に追加する前に TT で証明済みか確認する．

**実装:** `try_prefilter_block` (`solver.rs`)

1. 合駒を盤上で実行
2. 攻め方の合法手から合駒マスへの捕獲かつ王手になる手を探索
3. 捕獲後の局面を TT で参照
4. pn=0(証明済み)なら合駒の OR ノードも証明 → 展開不要

**IDS との相乗効果:** 浅い IDS 反復でチェーン末端の証明が TT に蓄積され，
深い反復では浅いレベルの合駒がプレフィルタで即座にスキップされる．
これによりチェーン合駒がボトムアップに折り畳まれる．

### 8.5 同一マス証明転用

同一マスへの異なる駒種の合駒間で TT エントリを相互利用する．

**実装:** `cross_deduce_children` (`solver.rs`)

同一マス S への合駒 P1, P2, ..., Pn は，攻め方が捕獲した後の盤面(position_key)が
全て同一になる(異なるのは攻め方の持ち駒のみ)．

1. 合駒 i が証明済みになった後，同一マスの未解決合駒 j を列挙
2. 合駒 j の捕獲後の攻め方持ち駒を計算:
   `hand_j = base_hand - solved_piece + piece_j`
3. TT で捕獲後局面を参照: `look_up(pc_pk, &hand_j, pc_remaining)`
4. pn=0 なら合駒 j も証明 → `deferred_children` から除去

### 8.6 合駒 DN バイアス

AND ノードの合駒(駒打ち)の初期 dn にバイアスを加算し，探索優先度を下げる．

**実装:** 定数 `INTERPOSE_DN_BIAS = 8` (mod.rs)

非合駒応手の初期 dn(=1)より十分大きく設定し，
df-pn の自然な閾値制御で king move → drop の順序を実現する．
遅延展開(§8.3)が主要な制御手段であり，DN バイアスは補助的な役割．

### 8.7 チェーンマス内→外順序

チェーンマスの合駒を玉に近い側(内側)から飛び駒に近い側(外側)の順にソートする．

**実装:** `mid()` 内のチェーンドロップ順序付け (`solver.rs`)

```rust
deferred_children.sort_by_key(|(m, _, _, _)| {
    let to = m.to_sq();
    let dr = (to.row() - king_sq.row()).abs();
    let dc = (to.col() - king_sq.col()).abs();
    dr.max(dc)  // チェビシェフ距離: 内側(小)優先
});
```

- S1(内側)の証明で蓄積した TT エントリが S2(外側)の探索で再利用可能
- 短いサブチェーンから順に証明が蓄積され，長いサブチェーンの TT 再利用効率が向上

### 8.8 チェーン深さ DN スケーリング

チェーン合駒のみで構成される AND ノードで，DN バイアスをチェーン内位置に応じてスケーリングする．

**実装:** `mid()` 内の DN バイアス計算 (`solver.rs`)

```rust
let bias = if let Some(ksq) = chain_king_sq {
    let dist = chebyshev_distance(to, ksq);
    INTERPOSE_DN_BIAS * dist
} else {
    INTERPOSE_DN_BIAS
};
```

- 内側マス(d=1): `INTERPOSE_DN_BIAS × 1` — 最小バイアス，優先的に探索
- 外側マス(d=5): `INTERPOSE_DN_BIAS × 5` — 大きなバイアス，後回し
- `chain_king_sq` はチェーン判定時に保持した玉位置(チェーン AND ノードのみ)

§8.7(ソート順)と組み合わせることで相乗効果がある．

---

## 9. 手順改善

### 9.1 TT Best Move 動的手順改善

TT エントリに保存された最善手(§6.5)を利用し，手順の先頭にスワップする．

**実装:** solver.rs (Dynamic Move Ordering)

`look_up_best_move(pos_key, hand)` で TT から Move16 を取得し，
手リストの先頭に配置する．

### 9.2 Killer Move (OR ノード専用)

同一 ply の別の局面で証明に寄与した手を記録し，優先的に探索する．

**実装:** solver.rs (`killer_table`, `record_killer`, `get_killers`)

- **テーブル**: `killer_table: Vec<[u16; 2]>` — 各 ply に2スロット
- **記録タイミング**: OR ノードの証明達成時および閾値超過時
- **適用**: TT Best Move の直後に配置

**手順優先度:** TT Best Move > Killer Move (2スロット/ply) > 静的手順(DFPN-E)

AND ノードでは全子ノードの探索が必要(WPN/SNDA 計算のため)なので適用しない．

### 9.3 捨て駒ブースト

OR ノードで全王手が「支えなし」の捨て駒である場合に pn を加算して
探索優先度を下げる．人間が直感的に「不詰」と見切るのと同様のヒューリスティック．

**実装:** `sacrifice_check_boost` (mod.rs (`sacrifice_check_boost`))

- 各王手の `to_sq` に攻め方の他の駒が利いているか確認(移動元を除外)
- 全王手が捨て駒なら `boost = 2` を返す(pn に加算)
- 支えがある王手が1つでもあれば 0

---

## 10. 既知の課題とベンチマーク

### 10.1 29手詰め問題

```
SFEN: l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1
```

**テスト:** `test_tsume_6_29te` (tests.rs), `test_tsume_6_29te_no_pns` (tests.rs)

| 構成 | ノード数 | 結果 |
|------|---------|------|
| PNS + IDS | ~18.5M | 29手詰め (正解) |
| IDS のみ (PNS なし) | ~18.5M | 29手詰め (正解) |

29手詰めは **PNS なしに IDS のみで解ける**ことが確認されている．
これは IDS-MID 単体のロバストネスを示す重要なベンチマーク．

### 10.2 39手詰め問題

```
SFEN: 9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1
```

**テスト:** `test_tsume_39te_aigoma` (tests.rs, `#[ignore]`)

**現状 (v0.21.1):** 100M ノードでも UNKNOWN(未解決)．

#### 問題の構造

39手詰め問題はチェーン合駒最適化(§8)のメインターゲットである．
ply 24 で銀5g→6fの開き王手(飛車8gの横利き開放)が発生し，
飛車(8g)と玉(1g)の間の5マス(7g, 6g, 5g, 4g, 3g)に
チェーン合駒構造が出現する．

ply 25 AND ノードの 23応手を個別に探索すると**すべて 200K ノード以内で解ける**:

| カテゴリ | 応手 | ノード数 | 結果 |
|---------|------|---------|------|
| 即詰み | P/L/N/S/G/B\*2g | 各1 | Mate(1) |
| 3〜5筋合駒 | P\*3g 〜 N\*5g (9手) | 41〜80K | Mate(7〜15) |
| 6〜7筋合駒 | L\*6g, B\*6g, B\*7g, N\*6g | 71K〜83K | Mate(13〜15) |
| 玉逃げ | 1g1f | 140K | Mate(19) |
| PV(最長) | 1g1h | 103 | Mate(13) |
| 不詰 | P\*7g, N\*7g | 各~200K | NoMate |

最も重い応手は P\*7g / N\*7g の不詰証明(各 ~200K ノード)であり，
23応手の合計は約 **1.5M ノード**に収まる．

#### backward 解析 (v0.20.35, 1M ノード/180秒)

PV を逆順にたどり，各偶数 ply(攻め方手番 = OR ノード)を個別に解いた結果:

| Ply | 残り手数 | ノード数 | TT エントリ | 結果 |
|-----|---------|---------|-----------|------|
| 38 | 1 | 14 | 39 | Mate(1) |
| 36 | 3 | 2 | 8 | Mate(3) |
| 34 | 5 | 12 | 36 | Mate(5) |
| 32 | 7 | 17 | 46 | Mate(7) |
| 30 | 9 | 22 | 56 | Mate(9) |
| 28 | 11 | 89 | 203 | Mate(11) |
| 26 | 13 | 103 | 198 | Mate(13) |
| **24** | **15** | **396,636** | **343,999** | **Mate(21)** |
| **22** | **17** | **1,000,000** | **573,936** | **Unknown** |

ply 24 サブ問題は 397K ノードで解けるにもかかわらず，
**ルート(ply 0)からの統合探索では 100M ノードでも UNKNOWN** である(全体ベンチマーク参照)．
この 250倍以上の乖離が閾値飢餓の核心的な現象であり，
MID がルートから ply 24 に到達するまでに閾値が枯渇し，
ply 24 に十分な探索予算を配分できないことが原因である．

ply 26 → ply 24 でノード数が 103 → 397K に急増する．
原因はチェーン合駒の指数的分岐(ply 24 の開き王手以降)．

**TT 内訳 (tt\_diag):**

| Ply | proven | disproven | intermediate | 備考 |
|-----|--------|-----------|-------------|------|
| 24 (Mate) | 6,108 | 246,130 (72%) | 91,761 | 不詰証明が主要コスト |
| 22 (Unknown) | 16,252 | 557,684 (97%) | **0** | TT スラッシング状態 |

ply 22 で intermediate=0 は，TT が完全に proven/disproven で埋まり
新たな中間局面が生成されていないことを意味する．
ply 23 AND の PV 応手 1f1g が 250K(予算上限)で Unknown であり，
その先のチェーン構造が現在のボトルネック．

#### 閾値飢餓 (Threshold Starvation)

統合探索で MID が深い ply に到達できない根本原因．

**メカニズム:** `heuristic_or_pn` の返り値 1〜3(= 1〜3 PN\_UNIT)により，
OR ノードの 1+ε 閾値が小さくなり，子 AND ノードに渡される pn 予算が不足する．
KomoringHeights は初期 pn=10〜80 を使用しており，この問題が発生しにくい．

**エビデンス — 全体ベンチマーク (v0.20.34, PN\_UNIT=1, depth=41, timeout=600s):**

| 構成 | ノード数 | NPS | TT エントリ | 結果 |
|------|---------|-----|-----------|------|
| MID only | 50M | ~1,021K | 3,339,549 | UNKNOWN |
| MID only | 100M | ~1,327K | **3,339,549** | UNKNOWN |
| MID+Frontier | 50M | ~829K | 4,067,787 | UNKNOWN |
| MID+Frontier | 100M | ~1,105K | **4,427,936** | UNKNOWN |

MID only は 50M→100M で TT エントリが**完全に同一**（完全な停滞状態）．
Frontier Variant(§11.7)では TT が成長し続けるが，NPS が ~20% 低下する．

**なぜ初期 pn の拡大が失敗したか:**

v0.20.35 では pn 初期値の拡大(depth-adaptive heuristic\_or\_pn)，
OR pn フロア，ε の拡大を試みたが，全て NPS 低下や TT 停滞を招いた．
当時は「OR ノードの 1+ε 手切替と閾値飢餓対策は構造的に衝突する」
と結論したが，これは誤りであった．

1+ε トリックは乗算型(§3.1)なので，全ての pn/dn を一様にスケーリング
すればソルバーの挙動は完全に一致するはずである．
にもかかわらず v0.20.35 で失敗した原因は，**pn 初期値のみを拡大し，
他の pn/dn 定数(閾値フロア，バイアス，WPN 加算分，TT ミス判定等)を
スケーリングしなかった**ことにある．部分的なスケーリングは pn/dn の
相対関係を破壊し，1+ε の手切替やフロアの効き方を歪めて停滞を招く．

**PN\_UNIT(§3.5)導入で特定されたスケーリング漏れ:**

| 漏れの箇所 | 影響 |
|-----------|------|
| WPN `(unproven_count - 1)` に PN\_UNIT 未適用 | AND pn 計算が破壊され 2 テスト FAIL |
| TT ミス判定 `cpn == 1` がスケール追随せず | heuristic 初期化が不発 |
| depth 制限超過時の初期 pn=1u32 | 深部の TT に異常に小さい pn が残存 |
| 除算(`/4`, `/2`)の丸め精度が PN\_UNIT で変動 | P\*4g サブ問題で探索パターン乖離 |

これらを全て修正し divide-at-unit-scale パターン(§3.5)を適用した結果，
PN\_UNIT=1 と PN\_UNIT=64 で backward 解析が ply 24 まで完全一致し，
PN\_UNIT 拡大が安全に閾値飢餓を緩和できる基盤が整った．

**採用済み手法:**

| 手法 | 効果 | 備考 | バージョン |
|------|------|------|-----------|
| チェーン AND pn\_floor boost (§3.3) | ply 24 サブ: 1M Unknown → 397K Mate | チェーン AND 内部のみ改善 | v0.20.34 |
| Frontier Variant (§11.7) | MID 停滞後も TT 成長継続 | NPS ~20% 低下 | v0.20.34 |
| PN\_UNIT 統一スケーリング (§3.5) | PN\_UNIT 拡大の基盤整備 | PN\_UNIT=64 で全テスト通過 | v0.20.36 |
| PN\_UNIT=16 + 自然精度 epsilon (§3.1, §3.5) | 各 OR レベル ~19% 閾値余裕 | N\*7g NoMate: 200K→82K | v0.21.0 |
| heuristic 高解像度化 (§5.1) | S/4 刻み 7 段階に解像度向上 | 方針A の一部 | v0.21.0 |
| IDS 動的予算配分 (§2.6) | MID 停滞で Frontier に動的移行 | 方針B の実装 | v0.21.0 |

#### 今後の改善方針

**方針 A: PN\_UNIT 拡大 + heuristic 高解像度化 + 自然精度 epsilon — 採用(v0.21.0)**

PN\_UNIT(§3.5)の基盤を活用した閾値飢餓の直接的な改善．
3つの要素を組み合わせることで OR ノードの sibling\_based を増大させる:

1. **PN\_UNIT の拡大**(例: PN\_UNIT=16): 全 pn/dn 値を 16 倍にスケーリング．
   スケーリング等価性(§3.5)により既存テストは全通過が保証される．

2. **自然精度 epsilon(§3.5)**: divide-at-unit-scale を epsilon 計算で外し，
   除算の自然精度を活かす．`second_best = 3S` のとき:
   - 等価パターン: epsilon = 0 + S = S，sibling\_based = 4S（PN\_UNIT=1 と同じ）
   - 自然精度:    epsilon = 12 + S = 28，sibling\_based = 76（4.75S）

   各 OR レベルで sibling\_based が ~19% 増加し，12レベルで
   累積 `1.19^12 ≈ 8倍` の閾値余裕が得られる可能性がある．

3. **heuristic\_or\_pn の高解像度化**: S〜3S の範囲で中間値を返す．
   現在は safe\_escapes=0→S，=1→S，=2→S，=3→2S，≥4→2S の 3段階だが，
   PN\_UNIT=16 なら以下のような連続スケーリングが可能:

   | safe\_escapes | 現在 (3段階) | 提案 | epsilon 自然精度 |
   |-------------|------------|------|----------------|
   | 0 | S | S | +S |
   | 1 | S | S + S/4 | +S (切り捨て同値) |
   | 2 | S | S + S/2 | +S + S/8 |
   | 3 | 2S | 2S | +S/2 + S |
   | ≥4 | 2S | 2S + S/2 | +S/2 + S + S/8 |

   num\_checks との組み合わせでさらに分化できる．
   heuristic\_and\_pn についても同様に解像度を上げ，
   応手数と逃げ場数の組み合わせをより正確に反映する．

**実装手順:**
1. PN\_UNIT=16 に変更し 126 テスト全通過を確認（等価パターンのまま）
2. epsilon のみ自然精度に切替え，39手詰め backward 解析で効果計測
3. heuristic\_or\_pn を高解像度化し backward 解析で効果計測
4. 50M/100M 全体ベンチマークで閾値飢餓の改善度を確認

**方針 B: IDS フルデプスの動的予算配分 — 採用(v0.21.0)**

MID の最大予算(全体の 1/2)を 1/4 チャンクに分割し，
各チャンク後に TT エントリ数の成長を確認する．
MID が新しい TT エントリを生成しなくなった時点で停滞と判定し，
残り予算を動的に Frontier に回す(§2.6 参照)．

#### 方針A+B 実装結果 (v0.21.0)

v0.21.0 にて方針A(3要素)と方針Bを実装し，backward 解析で効果を計測した．

**実装内容:**

1. **PN\_UNIT=16**: 全 pn/dn を 16 倍スケーリング．55 テスト全通過(等価性確認)．
2. **自然精度 epsilon**: `second_best / 4 + PN_UNIT`(divide-at-unit-scale 除去)．
3. **heuristic 高解像度化**: safe\_escapes=1→+S/4，=2→+S/2，=3→S\*3/2，
   ≥4→S\*2+S/4 の連続スケーリング(heuristic\_or\_pn / heuristic\_and\_pn 両方)．
4. **IDS 動的予算配分(方針B)**: MID を 1/4 チャンクに分割し TT 成長を監視．
   TT エントリが増加しなくなった時点で MID を停止し残り予算を Frontier に移行．

**backward 解析 (v0.21.0, 1M ノード/180秒):**

| Ply | 残り手数 | ノード数 | TT エントリ | 結果 | v0.20.35 比 |
|-----|---------|---------|-----------|------|------------|
| 38 | 1 | 14 | 39 | Mate(1) | 同一 |
| 36 | 3 | 2 | 8 | Mate(3) | 同一 |
| 34 | 5 | 12 | 36 | Mate(5) | 同一 |
| 32 | 7 | 17 | 46 | Mate(7) | 同一 |
| 30 | 9 | 22 | 56 | Mate(9) | 同一 |
| 28 | 11 | 92 | 203 | Mate(11) | 89→92 |
| 26 | 13 | 165 | 202 | Mate(13) | 103→165 |
| **24** | **15** | **1,000,000** | **628,370** | **Unknown** | 397K Mate→1M Unknown |

ply 24 サブ問題は v0.20.35 で 397K Mate だったが，v0.21.0 では 1M で Unknown に退行した．
原因は自然精度 epsilon による各 OR レベルでの分岐探索量の ~19% 増加．
個別応手の NoMate 証明は改善(N\*7g: 200K→82K)するが，
全応手の合計コストが増加し 1M 予算内で収まらなくなった．

**ply 25 AND 応手コスト比較 (v0.21.0 vs v0.20.35):**

| 応手 | v0.20.35 | v0.21.0 | 変化 |
|------|---------|---------|------|
| 1g1f (Mate) | 140K | 130K | **改善** |
| P\*7g (NoMate) | ~200K | 149K | **大幅改善** |
| N\*7g (NoMate) | ~200K | 82K | **大幅改善** |
| L\*6g (Mate) | 71K | 127K | 悪化(手数増 13→17) |
| B\*7g (Mate) | 75K | 105K | 悪化(手数増 13→15) |

**TT 内訳 (ply 24, tt\_diag):**

| 指標 | v0.20.35 | v0.21.0 | 備考 |
|------|---------|---------|------|
| proven | 6,108 | 13,936 | 2.3倍に増加 |
| disproven | 246,130 | 614,434 | 2.5倍に増加 |
| intermediate | 91,761 | **0** | TT スラッシング |
| 合計 | 343,999 | 628,370 | 1.8倍に成長 |

TT エントリ数は 344K→628K に 1.8 倍成長しており，
自然精度 epsilon が各 OR レベルでの探索範囲を拡大していることを示す．
ただし intermediate=0 は ply 24 単体でも TT スラッシング状態であり，
閾値飢餓の根本的な解消には至っていない．

統合探索(ルート→ply 24)での効果は，全体ベンチマーク(50M/100M)で
TT エントリ数が予算に応じて成長するかを確認する必要がある．

#### v0.21.1 の追加改善と 29 手詰めベンチマーク

v0.21.1 では方針A+B に加え以下の改善を実施:

1. **heuristic\_or\_pn 拡張 (S→8S)**: safe\_escapes と num\_checks の二次元スケーリング．
   KomoringHeights の pn=10-80 に相当する範囲(16-128)．
2. **epsilon 増加 (second\_best/4→/3)**: OR/AND 各レベルの閾値マージンを 25%→33% に．
3. **single-child ループ停滞検出**: pn/dn 不変時に SC\_STAGNATION\_LIMIT=4 で脱出．
4. **MID ループ停滞検出改善**: nodes\_used>1 スキップを廃止し常に pn/dn 変化を確認．
5. **pn\_floor 自然精度 + 比率 2/3**: AND レベルの閾値減衰を (1/2)^N → (2/3)^N に改善．
6. **TT クラスタ飽和対策**: replace\_weakest\_for\_disproof で foreign protected 置換．
7. **TT 2M クラスタ化**: ハッシュ衝突緩和 + キャッシュ効率の最適バランス(§6.6)．
8. **MID チャンク 1M 固定**: 停滞の早期検出のため 15M→1M に縮小．
9. **Periodic GC 容量ベース化**: 50M/60M のハードコード → capacity×80%/90% に自動適応．

**backward 解析 (v0.21.1, 1M ノード/180秒):**

| Ply | 残り手数 | ノード数 | TT エントリ | 結果 | v0.20.35 比 |
|-----|---------|---------|-----------|------|------------|
| 38 | 1 | 14 | 39 | Mate(1) | 同一 |
| 26 | 13 | 100 | 189 | Mate(13) | 103→100 |
| **24** | **15** | **332,630** | **277,661** | **Mate(21)** | **397K→333K (改善)** |
| **22** | **17** | **1,000,000** | **927,195** | **Unknown** | 574K→927K (TT 成長) |

ply 24 サブ問題が v0.20.35 の 397K から 333K に改善．
TT エントリは 344K→278K に圧縮されつつ Mate(21) を発見．

**29 手詰めベンチマーク (v0.21.1, no\_pns, depth=31, timeout=1200s):**

| 構成 | ノード | 時間 | TT max | 結果 |
|------|--------|------|--------|------|
| v0.20.36 (1M clusters) | >120M | — | 3.1M | UNKNOWN |
| v0.21.1 (1M clusters) | >120M | — | 3.1M | UNKNOWN |
| v0.21.1 (2M clusters) | **58.8M** | **201s** | **6.3M** | **Mate(29)** |
| v0.21.1 (4M clusters) | 109M | 556s | 12.5M | Mate(29) |

29 手詰め解決の主要因は TT クラスタ数の拡大(1M→2M)とクラスタ飽和対策．
2M が 4M より高速な理由はキャッシュ効率(480MB vs 960MB)．

**探索停滞の診断で特定された問題と修正:**

| 問題 | 症状 | 原因 | 修正 |
|------|------|------|------|
| sc\_loop\_hang | single-child ループ無限反復 | pn/dn 不変でも脱出条件なし | SC\_STAGNATION\_LIMIT=4 |
| ply 31 集中 | 93% のノードが depth boundary | nodes\_used>1 で停滞リセット | 常に pn/dn 変化を確認 |
| TT 凍結 | 新規 proof/disproof 挿入ゼロ | foreign protected がクラスタ占有 | replace\_weakest\_for\_disproof |
| TT ハッシュ衝突 | 全体 52% なのに飽和 | 1M クラスタでの衝突限界 | 2M クラスタ化 |

**方針 C: Frontier Variant の軽量化**

PNS アリーナ確保のオーバーヘッドが NPS を ~27% 低下させる(§11.7)．
アリーナの再利用や増分更新で NPS 低下を抑えれば，
Frontier フェーズの実効探索力が向上する．

**方針 E: TT 構造の改善 (リニアプロービング + amount ベース置換)**

v0.21.1 の調査で特定された TT の構造的問題:

1. **クラスタ飽和**: 固定 6 エントリ/クラスタでは同一盤面の hand バリアントが
   溢れて store 失敗が発生する．df-pn の TT はキャッシュとして設計されており，
   エントリ喪失は再計算で対応可能だが，maou\_shogi では store 失敗が
   single-child / MID ループの停滞を引き起こす（対症療法として停滞検出を実装済み）．
2. **置換ポリシー**: `|pn-dn|` スコアベースの置換は探索量を考慮しない．
   KomoringHeights は `amount_` フィールドで探索投資量を記録し，
   深い探索で得た価値の高いエントリを GC/置換時に保護する．

改善候補:
- **リニアプロービング**: クラスタサイズ制約を解消し hand バリアントを
  必要なだけ保持．KomoringHeights が採用．
- **amount ベース置換**: エントリの探索投資量を記録し，置換時に保護優先度を判定．
- **store 失敗時の fallback**: store 成否を呼び出し元に返し，
  look\_up に依存せず直接値を使う．

**方針 D: 並列 df-pn (§11.6)**

マルチスレッドでの探索は構造的にスケールし，
閾値制御の非効率性を計算資源で補償できる．
長期的だが最大のスケーラビリティ．

### 10.3 ミクロコスモス(1525手詰)の解法比較

| ソルバー | 解答時間 | 主要手法 |
|---------|---------|---------|
| 脊尾詰 (1997) | ~20時間 | PN*，~188M 局面 |
| KomoringHeights | ~10分 | df-pn+, SNDA, 証明駒/反証駒, GHI 対策, 合駒遅延展開 |
| shtsume | ~1分 | 不明(最速ソルバー) |
| やねうら王 | 解けない | TT GC 未実装(~3.5TB 必要) |
| maou_shogi | 未挑戦 | — |

---

## 11. 最適化案の評価

v0.20.33〜v0.20.34 にて §11.1〜11.7 (§11.6 を除く) の最適化案を実装・計測し，
採用/不採用を決定した．

**計測環境:** DevContainer (2 vCPU)，`--release` ビルド．
**計測問題:** 39手詰め合駒問題 (§10.2), depth=41, max\_nodes=10M, timeout=120s．
**ベースライン:** §11.1〜11.3 は v0.20.32 HashMap TT (NPS ~227K)，
§11.5 は v0.20.34 フラット TT (NPS ~868K) をベースラインとする．

### 11.1 History Heuristic — 不採用

**出典:** Schaeffer, "The History Heuristic and Alpha-Beta Search Enhancements in Practice" (1989)

OR ノードの手順序を改善する手法．
手がカットオフを引き起こした回数を `[piece][to_sq]` テーブルに記録し，
Killer Move の後の手順序に利用する．

**計測結果 (39手詰め, 10M ノード):**

| 方式 | NPS | 時間 | 結果 |
|------|-----|------|------|
| ベースライン | ~227K | 44.1s | UNKNOWN |
| sort_unstable_by (全体ソート) | ~205K | 48.7s | UNKNOWN |
| 部分選択 (最善1手のみ) | ~210K | 47.6s | UNKNOWN |

- **不採用理由:** OR ノード毎の O(n) スキャンが NPS を 7〜10% 低下させ，
  TT Best Move + Killer Move (§9.1, §9.2) の効果を上回らない．
  詰将棋では王手手数が少ない(典型 3〜15 手)ため，
  ヒストリテーブルの統計量が蓄積しにくい．

### 11.2 PNS 予算適応配分 — 不採用

PNS の予算上限(`PNS_BUDGET_CAP = 150K`)を問題の深さに応じて適応的に設定する．
depth >= 21 で 150K + (depth − 20) × 50K，上限 500K に拡大する案．

**計測結果:** 深い問題(39手詰め)で PNS フェーズが 150K → 500K ノードに拡大するが，
PNS アリーナの飽和による効率低下が顕著で，MID に回す予算が減少する．
NPS の改善は観測されず，TT エントリ数にも有意差なし．

- **不採用理由:** PNS のアリーナ飽和後の反復コストが大きく，
  150K の上限で十分．深い問題では MID 予算の確保が優先．

### 11.3 AND ノード応手順序改善 — 不採用

`edge_cost_and` を改善し，合駒の駒種に応じたコスト差(歩=0, 角飛=2)と
駒取りコストの引き上げ(2→3)を実装した．

**計測結果 (9手詰め):** 50 → 46 ノード(−8%)で改善．
**計測結果 (39手詰め):** TT パターンの変化により探索効率が低下．
**テスト影響:** `test_no_checkmate_gold_interposition` が 1M ノードで不詰を
証明できなくなる退行が発生．

- **不採用理由:** 不詰証明の退行が許容できない．
  駒種コストの差が探索パターンを大きく変え，特定の不詰証明で
  追加ノードが必要になる．既存の DFPN-E コスト(§5.2)で十分．

### 11.4 TT フラットハッシュテーブル化 — 採用

`FxHashMap<u64, Vec<DfPnEntry>>` を固定サイズのフラットハッシュテーブルに置換．

**実装 (v0.20.34):** 1M クラスタ × 6 エントリ/クラスタのフラット配列．
`pos_key % num_clusters` でクラスタを特定し，クラスタ内で線形スキャン．
パレートフロンティア管理(§6.3)，前方チェーン比較(§6.2)，
経路依存フラグ(§7.1)のセマンティクスは完全に維持．
クラスタ満杯時は異なる pos\_key のエントリを優先的に置換する．

**計測結果 (39手詰め, 10M ノード):**

| 方式 | NPS | TT エントリ数 | 時間 | 結果 |
|------|-----|-------------|------|------|
| HashMap ベースライン | ~227K | 14,290,056 | 44.1s | UNKNOWN |
| フラットハッシュテーブル | ~868K | 1,304,639 | 11.5s | UNKNOWN |

- **NPS: 3.83× 高速化** — Vec ヒープアロケーション完全排除 + キャッシュ局所性向上
- **TT メモリ: 91% 削減** — クラスタサイズ 6 の制限により，
  低価値の中間エントリが自動的に置換される．これは意図的な設計:
  証明/反証の保護ポリシーにより重要なエントリは維持され，
  HashMap 時代の 16 エントリ上限は過剰だった．
- **9手詰め:** 50 ノード(同等)，初期化 0.094s(テーブル確保)
- **採用理由:** NPS の劇的な改善．HashMap のポインタチェーシングと
  Vec のヒープアロケーションが主要なボトルネックだった．

### 11.5 Lazy Move Generation — 不採用

OR ノードで手を段階的に生成する:
TT best move → killer → 捕獲王手 → 成王手 → その他

**計測結果 (39手詰め, 10M ノード, フラット TT ベース):**

| 方式 | NPS | TT エントリ数 | 時間 |
|------|-----|-------------|------|
| ベースライン (フラット TT) | ~868K | 1,304,639 | 11.5s |
| Lazy (3手詰め判定スキップ) | ~605K | 2,145,723 | 16.5s |
| Lazy (即詰み判定のみ残し) | ~582K | 2,102,830 | 17.2s |

3種類の Lazy 方式を実装・計測したが，いずれも NPS が 30〜33% 低下:

1. **3手詰め判定全スキップ**: エッジコストのみで初期化 → 16.5s
2. **即詰み判定(応手なし)のみ残し**: `generate_defense_moves_inner(early_exit=true)` +
   応手なしなら即詰み → 17.2s
3. **全子フルインライン(Lazy 無効)**: 元と同等 → 12.6s

- **不採用理由:** インライン詰み検出(§5.4)は MID 関数の性能の鍵であり，
  スキップすると証明を安価に達成する機会を喪失し，mid() 再帰の増加で
  総ノード数と TT エントリ数が 60%+ 増大する．

  現在の MID 構造は TT Best Move を init ループの先頭で処理し，
  TT Best Move が証明を達成すれば init ループの early return により
  残りの全子の init をスキップする．これは実質的に最も効果的な
  「lazy」動作であり，追加の lazy 化は他の最適化(§5.4 インライン
  詰み検出，§9.1 TT Best Move)との相乗効果を損なう．

### 11.6 並列 df-pn (Kaneko 2010) — 対象外

**出典:** Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010);
Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014)

複数スレッドで異なるサブツリーを同時に探索する．
共有 TT + 仮想証明数による協調探索．
8スレッドで 3.58× 高速化 (並列効率 ~0.5)．

- **期待効果:** 大．マルチコアの活用
- **実装コスト:** 高．TT スレッドセーフ化 + 負荷分散

### 11.7 Frontier Variant (PNS→局所MID) — 採用

PNS でフロンティアノード(未展開の最有望ノード)を特定し，
各フロンティアに対して局所的に MID を起動する方式．
PNS が閾値なしでグローバル最適なノード選択を行い，
MID は浅い部分木のみを担当するため閾値飢餓が発生しにくい．

**実装 (v0.20.33→v0.20.34):** IDS-dfpn のフルデプスステップに統合(§2.6)．
フルデプスで MID(予算の 1/2)を先行実行し，未解決なら残り予算で
PNS→MID サイクルにフォールバックする．
各サイクルで PNS(残り予算の 1/20，上限 50K)→ MID(残り予算の 1/4)を交互実行し，
サイクル間で `retain_proofs()` による TT 清掃を行う．

**計測結果 (39手詰め, 10M ノード, フラット TT):**

| 方式 | NPS | TT エントリ数 | 時間 | 結果 |
|------|-----|-------------|------|------|
| MID only (Frontier なし) | ~868K | 1,304,639 | 11.5s | UNKNOWN |
| MID(1/2)+Frontier(1/2) | ~630K | 1,107,108 | 15.9s | UNKNOWN |

**計測結果 (39手詰め, 50M/100M ノード, 4M クラスタ TT):**

| 方式 | ノード | NPS | TT エントリ | 結果 |
|------|-------|-----|-----------|------|
| MID only | 100M | ~1,327K | **3,339,549** (50M→100M で成長ゼロ) | UNKNOWN |
| MID+Frontier | 100M | ~1,105K | **4,427,936** (50M→100M で +360K 成長) | UNKNOWN |

- **NPS 低下:** ~27%．PNS アリーナ確保・TT 清掃のオーバーヘッド．
- **閾値飢餓の回避効果:** MID only は 50M→100M で TT エントリが完全に停滞(同一値)するが，
  Frontier は TT が成長し続け，新しい局面の探索を継続できる．
- **採用理由:** NPS は低下するが，MID の閾値飢餓(§10.2)を構造的に回避する唯一の手法．
  大規模ノード予算で MID が完全停滞する問題の解決策として不可欠．

---

## 12. 参考文献

### 論文

- Nagai & Imai, "df-pn Algorithm Application to Tsume-Shogi" (IPSJ Journal 43(6), 2002)
- Nagai, "Df-pn Algorithm for Searching AND/OR Trees and Its Applications" (Ph.D. Dissertation, UTokyo, 2002)
- Seo, Iida & Uiterwijk, "The PN*-search algorithm: Application to tsume-shogi" (AI 129, 2001)
- Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)
- Kishimoto & Müller, "About the Completeness of Depth-First Proof-Number Search" (2008)
- Kishimoto, "Dealing with Infinite Loops, Underestimation, and Overestimation of Depth-First Proof-Number Search" (AAAI 2010)
- Pawlewicz & Lew, "Improving Depth-First PN-Search: 1+ε Trick" (CG 2007)
- Song Zhang et al., "Deep df-pn and Its Efficient Implementations" (CG 2017)
- "Depth-First Proof-Number Search with Heuristic Edge Cost" (NeurIPS 2019)
- Ueda, Hashimoto, Hashimoto & Iida, "Weak Proof-Number Search" (CG 2008)
- Saito et al., "Virtual Proof Number" (2006)
- Kishimoto, Winands, Müller & Saito, "Game-tree search using proof numbers: The first twenty years" (ICGA Journal 35, 2012)
- Kaneko, "Parallel Depth First Proof Number Search" (AAAI 2010)
- Pawlewicz & Hayward, "Scalable Parallel DFPN Search" (CG 2014)
- Hoki, Kaneko, Kishimoto & Ito, "Parallel Dovetailing and its Application to Depth-First Proof-Number Search" (ICGA Journal 36, 2013)
- Kaneko lab (UTokyo), "Initial pn/dn after expansion in df-pn for tsume-shogi" (GPW 2004)
- Breuker, Uiterwijk & van den Herik, "Replacement Schemes for Transposition Tables" (1994)
- Schaeffer, "The History Heuristic and Alpha-Beta Search Enhancements in Practice" (TCDE 14(4), 1989)
- Allis, "Searching for Solutions in Games and Artificial Intelligence" (1994)

### 既存ソルバー

| ソルバー | 特徴 |
|---------|------|
| KomoringHeights | df-pn+, SNDA, 証明駒/反証駒, GHI 対策, 合駒遅延展開 |
| shtsume | ミクロコスモス ~1分で解く最速ソルバー |
| やねうら王 | TT GC 未実装(~3.5TB 必要で解けない) |

### 日本語リソース

- [やねうら王 - 詰将棋アルゴリズムdf-pnのすべて](https://yaneuraou.yaneu.com/2024/05/08/all-about-df-pn/)
- [やねうら王 - ミクロコスモスは解けますか？](https://yaneuraou.yaneu.com/2020/12/30/yaneuraou-matesolver-microcosmos/)
- [コウモリのちょーおんぱ - df-pnアルゴリズムの解説](https://komorinfo.com/blog/df-pn-basics/)
- [コウモリのちょーおんぱ - KomoringHeightsを作った](https://komorinfo.com/blog/komoring-heights/)
- [コウモリのちょーおんぱ - GHI問題対策](https://komorinfo.com/blog/and-or-tree-ghi-problem/)
- [コウモリのちょーおんぱ - 証明駒／反証駒の活用方法](https://komorinfo.com/blog/proof-piece-and-disproof-piece/)
- [コウモリのちょーおんぱ - KomoringHeights v0.4.0](https://komorinfo.com/blog/komoring-heights-v040/)
- [コウモリのちょーおんぱ - KomoringHeights v0.5.0](https://komorinfo.com/blog/komoring-heights-v050/)
- [すぎゃーんメモ - Rustでつくる詰将棋Solver](https://memo.sugyan.com/entry/2021/11/11/005132)
- [Qhapaq - 高速な詰将棋アルゴリズムを完全に理解したい](https://qhapaq.hatenablog.com/entry/2020/07/19/233054)
- [人工知能学会誌 - 詰将棋探索技術(PDF)](https://www.jstage.jst.go.jp/article/jjsai/26/4/26_392/_pdf)

---

## 変更履歴

| 日付 | 版 | 内容 |
|------|-----|------|
| 2026-03-30 | v0.22.0 | §6.6 amount フィールド導入 + §6.6.1 クラスタ飽和問題の文書化．amount ベース replace\_weakest + GC．min\_depth は有用性なしで削除 |
| 2026-03-30 | v0.21.1 | §6.6 TT 2M クラスタ化 + クラスタ飽和対策 + 停滞検出改善．29手詰め 59M nodes / 201s で解決．§3.1 epsilon 1/3，§5.1 heuristic S-8S，§3.3 pn\_floor 2/3 |
| 2026-03-29 | v0.21.0 | §10.2 方針A+B 実装．PN\_UNIT=16 + 自然精度 epsilon(§3.1, §3.5) + heuristic 高解像度化(§5.1) + IDS 動的予算配分(§2.6)．N\*7g NoMate: 200K→82K，TT 成長 344K→628K |
| 2026-03-29 | v0.20.36 | §3.5 PN\_UNIT 統一スケーリング + divide-at-unit-scale パターン．スケーリング漏れ4件特定・修正，PN\_UNIT=64 で完全等価性確認 |
| 2026-03-29 | v0.20.35 | §3.3 チェーン AND pn\_floor boost．ply 24 サブ問題 1M Unknown → 397K Mate に改善 |
| 2026-03-29 | v0.20.34 | §6.6 フラットハッシュテーブル化，§11.4 採用．NPS 3.83× |
| 2026-03-29 | v0.20.33 | §11 最適化案の評価(§11.1-11.7)，§10.2 ベンチマーク追加，§11.7 Frontier Variant 採用 |
| — | v0.20.0 | §4.2 CD-WPN，§6.2 前方チェーン補填，§8.7-8.8 チェーン順序最適化 |
| — | v0.18.0 | §2.2 Best-First PNS，§8.5 同一マス証明転用，§9.2 Killer Move |
| — | v0.16.0 | §2.3 IDS-dfpn，§6.5 TT Best Move，§7.2 NM Remaining 伝播 |
| — | v0.1.0 | 初版: §2.1 Df-Pn，§6.1 持ち駒優越 |
