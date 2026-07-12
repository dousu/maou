# HCPE 変換 golden fixtures

cshogi 遺構リファクタリング (HCPE 変換の Rust 一括化) の parity gate 用．
リファクタ**前**の実装 (feat/kifu-parser, maou 0.34.0 時点) の出力を固定したもの．

## 再生成

生成器は `scratchpad/gen_refactor_golden.py` (gitignored)．

```bash
uv run python scratchpad/gen_refactor_golden.py
```

リファクタ後に再生成すると「旧実装の挙動固定」という意味が失われる．
意図的な挙動変更で期待値を更新する場合のみ，対応するコミットで明示的に行うこと．

## ファイル一覧

| golden | 入力 | 内容 |
|---|---|---|
| `csa_test_data_{1,2,3}.feather` | `test_dir/input/test_data_{1,2,3}.csa` | 先手勝ち(%TORYO)/後手勝ち(%TORYO)/引き分け(%JISHOGI) |
| `csa_floodgate2025_sennichite.feather` | `test_dir/input/floodgate2025_sennichite.csa` | %SENNICHITE，片側レーティング無し |
| `csa_floodgate2025_timeup.feather` | `test_dir/input/floodgate2025_timeup.csa` | %TIME_UP，両側レーティング無し |
| `csa_floodgate2025_toryo.feather` | `test_dir/input/floodgate2025_toryo.csa` | floodgate 2025 実棋譜の %TORYO |
| `csa_multi_game.current_first_game.feather` | `test_dir/input/csa_multi_game.csa` | 現行挙動 (先頭 1 局のみ変換) |
| `csa_multi_game.expected_all_games.feather` | 同上 | 新挙動期待 (全局変換，game>=1 は id `{stem}.hcpe_g{g}_{idx}`) |
| `kif_test_data_1.feather` | `test_dir/input/test_data_1.kifu` | KIF 163 手 (開始日時あり) |
| `kif_test_data_no_date.feather` | `test_dir/input/test_data_no_date.kifu` | KIF 開始日時なし (partitioningKey null) |
| `kif_test_data_sjis.expected.feather` | `test_dir/input/test_data_sjis.kif` | 新挙動期待 (cp932 fallback，UTF-8 版と同一内容で id の stem のみ差) |
| `hcpe_schema.txt` | - | 出力 .feather の Arrow schema ダンプ |

## floodgate fixture の出典

floodgate 2025-01-05 (https://wdoor.c.u-tokyo.ac.jp/shogi/x/2025/01/05/) より:

- `floodgate2025_sennichite.csa` = `wdoor+floodgate-300-10F+IVI_250105_C-5+AobaFuribisha_w1903_avg_1600p+20250105100003.csa`
- `floodgate2025_timeup.csa` = `wdoor+floodgate-300-10F+IVI_250105_C-5+AkeOme+20250105070002.csa`
- `floodgate2025_toryo.csa` = `wdoor+floodgate-300-10F+AobaFuribisha_w1909_avg_1600p+elmo_WCSC27_479_1000k+20250105233006.csa`

## 関連 golden

- 前処理: `tests/maou/app/pre_process/resources/golden/preprocess_golden.npz`
  (入力は `csa_test_data_{1,2,3}.feather`，`_process_single_array` の出力を hash 昇順で固定)
- stage2: `tests/maou/app/utility/resources/golden/stage2_golden.npz`
  (同入力の unique 局面ごとの特徴量/合法手ラベル)

前処理 golden の win 系列 (win_counts/move_win_values) は gameResult 規約バグ
(旧 shogi.Result 定義による誤読) の修正に伴い修正後の値で再生成済み．
hashes/labels/特徴量はリファクタ前実装と bit-exact のまま．
stage2 golden は二歩マスク修正 (occupied_files) 後の movegen で再生成済み．
