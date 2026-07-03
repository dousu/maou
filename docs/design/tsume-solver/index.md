# 詰将棋ソルバー設計ドキュメント

## 目次

| ドキュメント | 内容 |
|------------|------|
| [探索アーキテクチャ](search-architecture.md) | df-pn，φ/δ 統一探索，統一 `mid` 再帰，反復深化 (閾値成長)，全体フロー |
| [閾値制御](threshold-control.md) | 1+ε 子閾値，TCA (inc_flag)，PN_UNIT スケーリング，閾値成長 |
| [証明数・反証数の集約](proof-disproof-numbers.md) | φ/δ 集約，δ-sum と二重計数除去 (sum_mask)，DAG，cross-hand |
| [初期値ヒューリスティック](initial-heuristics.md) | df-pn+ 風初期化，エッジコスト，decouple，インライン 1 手詰 |
| [転置表管理](transposition-table.md) | len-aware 単一 TT，持ち駒優越，cross-hand look_up_parent，amount GC |
| [ループ・GHI 対策](loop-ghi.md) | 千日手検出，経路依存反証の scope 化，dominance 枝刈り |
| [手順改善・PV 復元](move-ordering-and-pv.md) | move ordering，PV 復元 (verify_proof + build_pv) |
| [合駒最適化](aigoma-optimization.md) | 中合いの再帰構造，遅延展開 (DelayedMoveList)，chain 対称性 |
| [最適化案の評価](optimization-proposals.md) | 単一スレッド方針 (binding)，不採用方針 |
| [39手詰 Oracle リファレンス](39te-oracle.md) | 39te ベンチ問題の user 確認済 oracle (正解手数・PV・サブ局面)，検証規則 |
| [参考文献](references.md) | 論文，既存ソルバー，日本語リソース |

---

## 1. 概要

maou_shogi の詰将棋ソルバーは **Df-Pn (Depth-First Proof-Number Search, Nagai 2002)** を
基盤とする．**単一の統一探索本体 `mid`** が，証明数 (pn)・反証数 (dn) を φ/δ に統一した
深さ優先探索を，root で閾値を段階的に拡大しながら (反復深化) 実行する．

### 設計目標

1. **cshogi が解けない問題をカバーする**: 片玉局面，合駒 (中合い) の正確な探索，最短手順保証．
2. **どんな詰将棋問題も解ける**: リソースパラメータ (`depth`, `nodes`, `timeout`) の増加で対応．
3. **健全性 (soundness) 最優先**: 偽の証明・偽の不詰を返さない．探索効率はその次．

### 設計方針: 単一スレッド維持

**性能改善は単一スレッドでのアルゴリズム改良で達成する方針とし，並列探索は採用しない**．
wheel の可搬性を binding 制約とするため，target-cpu=native ビルドも採用しない
(HW 命令は runtime gate のみ)．理由と詳細は
[optimization-proposals.md](optimization-proposals.md) 参照．

### 実装ファイル

`rust/maou_shogi/src/dfpn/` モジュールに全探索ロジックを実装する．主要構成:

| パス | 内容 |
|------|------|
| `mod.rs` | 薄い root．定数 (`PN_UNIT` 等)，再エクスポート |
| `api.rs` | 公開 facade (`solve_tsume*`) |
| `solver.rs` | `DfPnSolver` core + `TsumeResult` |
| `search/mod.rs` | 探索コア (`solve_impl` / `search_impl` / `emplace` / `step_best_child`) + 反復深化 root loop + TCA + DAG + 常設診断 (env-gated) |
| `search/expansion.rs` | `LocalExpansion` (φ/δ 集約・子閾値・sum_mask・sort) |
| `search/pv.rs` | PV 復元 (`verify_proof` / `build_pv`) |
| `movegen/mod.rs` | ノード手生成 (王手 / 応手) |
| `movegen/mate1ply.rs` | 1 手詰 detection (constructive) |
| `movegen/delayed_move_list.rs` | 中合い遅延展開 (DelayedMoveList) |
| `movegen/check_cache.rs` | 王手生成キャッシュ (CheckCache) |
| `tt/mod.rs` | 転置表 (`look_up_parent` / GC) |
| `tt/entry.rs` | TT エントリ (64 byte, len-aware) |
| `heuristics.rs` | 初期値 (`init_pn_dn_*`) / move ordering (`move_brief_eval`) |
| `proof_hand.rs` | 証明駒・反証駒 (`hand_gte`, ProofHandSet/DisproofHandSet) |
| `mate_len.rs` `search_result.rs` `path_stack.rs` `path_key.rs` | 値型・補助 |

### 実装済み手法一覧

出典のある手法は論文を併記する．maou 内部の版数は記さない．

| # | 手法 | 出典 | 節 |
|---|------|------|-----|
| 1 | Df-Pn | Nagai 2002 | [search §2.1](search-architecture.md) |
| 2 | φ/δ 統一探索 | Nagai 2002; Kishimoto et al. 2012 | [search §2.2](search-architecture.md) |
| 3 | 反復深化 (root 閾値成長) | Nagai 2002 (df-pn IDS) | [search §2.3](search-architecture.md) |
| 4 | 1+ε 子閾値トリック | Pawlewicz & Lew 2007 | [threshold §3.1](threshold-control.md) |
| 5 | TCA (inc_flag) | Kishimoto & Müller 2008; Kishimoto 2010 | [threshold §3.2](threshold-control.md) |
| 6 | PN_UNIT スケーリング | maou 独自 | [threshold §3.3](threshold-control.md) |
| 7 | δ-sum 集約 + 二重計数除去 (sum_mask) | KomoringHeights | [proof-numbers §4.1](proof-disproof-numbers.md) |
| 8 | EliminateDoubleCount (DAG) | KomoringHeights | [proof-numbers §4.2](proof-disproof-numbers.md) |
| 9 | cross-hand 親参照 (look_up_parent) | KomoringHeights | [proof-numbers §4.3](proof-disproof-numbers.md) |
| 10 | df-pn+ ヒューリスティック初期化 | GPW 2004; KomoringHeights | [heuristics §5.1](initial-heuristics.md) |
| 11 | エッジコスト (DFPN-E 系) | NeurIPS 2019 | [heuristics §5.2](initial-heuristics.md) |
| 12 | インライン 1 手詰 (constructive) | — | [heuristics §5.3](initial-heuristics.md) |
| 13 | 持ち駒優越 (hand dominance) | Nagai 2002 | [TT §6.1](transposition-table.md) |
| 14 | forward-chain 持ち駒代替 | maou 独自 | [TT §6.2](transposition-table.md) |
| 15 | len-aware 単一 TT (mate length 保持) | KomoringHeights | [TT §6.3](transposition-table.md) |
| 16 | 証明駒・反証駒の活用 | KomoringHeights | [TT §6.4](transposition-table.md) |
| 17 | amount ベース サンプリング GC | KomoringHeights 参考 | [TT §6.5](transposition-table.md) |
| 18 | GHI / 千日手対策 | Kishimoto & Müller 2004/2005 | [loop-ghi §7.1](loop-ghi.md) |
| 19 | 経路依存反証の scope 化 | maou 独自 | [loop-ghi §7.2](loop-ghi.md) |
| 20 | visit-history dominance | maou 独自 | [loop-ghi §7.3](loop-ghi.md) |
| 21 | 中合い遅延展開 (DelayedMoveList) | KomoringHeights v0.5.0 | [aigoma §8.1](aigoma-optimization.md) |
| 22 | move ordering (move_brief_eval) | maou 独自 | [move-ordering §9.1](move-ordering-and-pv.md) |
| 23 | PV 復元 (verify_proof + build_pv) | maou 独自 | [move-ordering §9.2](move-ordering-and-pv.md) |
| 24 | 単一スレッド方針 (並列不採用) | maou 方針 | [opt-proposals](optimization-proposals.md) |

---

## 2. 公開 API

`api.rs` が `solve_tsume*` を公開する (`DfPnSolver` を内部で構築)．

- **`solve_tsume(sfen, depth, nodes)`**: デフォルトタイムアウト (300 秒) で解く．
  `depth` 既定 31 (上限 47)，`nodes` 既定 1,048,576．
- **`solve_tsume_with_timeout(..., timeout_secs, find_shortest, pv_nodes_per_child, tt_gc_threshold)`**:
  全パラメータ指定版．`find_shortest` 既定 true (最短手順を保証する再探索を行う)．
- 返り値 `TsumeResult`:
  - `Checkmate { moves, nodes_searched }`: 詰み + PV．
  - `CheckmateNoPv { nodes_searched }`: 詰みは証明したが PV 復元の予算不足．
    `pv_nodes_per_child` を増やすと改善する．
  - `NoCheckmate { nodes_searched }`: 不詰 (健全に証明)．
  - `Unknown { nodes_searched }`: 予算・時間内に未解決．

詰み手順 (mate length) は engine 依存であり厳密最短とは限らない (proof-tree の手数)．
PV は `verify_proof` による replay で健全性 (PV 上の全応手が詰む) を検証する．
