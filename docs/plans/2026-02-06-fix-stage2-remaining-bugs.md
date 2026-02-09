# Stage2可視化 残存バグ修正 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stage2/Preprocessingデータの可視化で (1) 後手駒が「?」ではなく漢字で表示される (2) 合法手が`<invalid:N>`ではなくUSI形式で表示されるようにする．

**Architecture:** 2つの独立したバグを修正する．バグA: `SVGBoardRenderer._draw_pieces()`がcshogi形式の定数で白駒判定しているが，Stage2データはdomain形式 → domain定数に切替．バグB: `stage2_data_generation.py`が後手番の合法手ラベルを元の手番(WHITE)で生成しているが，盤面は先手視点に正規化済み → 正規化後の盤面で合法手ラベルを再生成する．バグBはStage2データの再生成が必要な破壊的変更．

**Tech Stack:** Python, cshogi, Polars, SVG

---

## 背景

### バグA: 後手駒が「?」で表示される

`SVGBoardRenderer._draw_pieces()` (board_renderer.py:449-458) が白駒判定に**cshogi形式の定数**を使用:

```python
# 現状 (board_renderer.py:451-458)
is_white = piece_id >= CSHOGI_WHITE_MIN      # >= 17
actual_piece_id = (
    piece_id - CSHOGI_WHITE_OFFSET            # - 16
    if is_white
    else piece_id
)
symbol = self.PIECE_SYMBOLS.get(actual_piece_id, "?")
```

Stage2の`boardIdPositions`はdomain形式(白駒=15-28):
- 白歩(15): `15 >= 17` → False → `PIECE_SYMBOLS.get(15, "?")` → "?"
- 白香(16): `16 >= 17` → False → `PIECE_SYMBOLS.get(16, "?")` → "?"
- 白桂(17): `17 >= 17` → True → `17 - 16 = 1` → "歩" (間違い！桂ではなく歩になる)

**修正:** `DOMAIN_WHITE_MIN` / `DOMAIN_WHITE_OFFSET` に切り替える．

### バグB: 合法手が`<invalid:N>`で表示される

`stage2_data_generation.py:260-263`でラベル生成時に:

```python
for move in board.get_legal_moves():
    label = make_move_label(board.get_turn(), move)  # 元のturn (WHITEの場合あり)
    legal_labels[label] = 1
```

一方，`make_board_id_positions(board)` (feature.py:168-171) は後手番なら盤面を**180度回転**して先手視点に正規化する．

可視化時，`_create_board_from_record()` は正規化済み盤面からBoardを再構築(turn=BLACK)し，`make_usi_move_from_label(board, label)` でラベルをデコードする．

**問題:** `make_move_label(Turn.WHITE, move)` と `make_move_label(Turn.BLACK, move)` は**異なるラベル空間**を使う(`Turn.WHITE`はsquareを`80 - sq`に変換する)．正規化済み(BLACK視点)の盤面でWHITE用ラベルをデコードしようとしても対応する合法手が見つからない → `<invalid>` になる．

**検証結果:**
```
Turn.WHITE labels: indices [5, 7, 13, 21, 23, ...]
Turn.BLACK labels: indices [264, 291, 300, 309, 318, ...]
→ 完全に異なるインデックス空間
```

**修正:** データ生成時に，正規化後の盤面を再構築し，そのboardの合法手からラベルを生成する．これにより，盤面(BLACK視点)とラベル(BLACK turn)が一致する．**既存Stage2データの再生成が必要．**

### 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/maou/domain/visualization/board_renderer.py` | `_draw_pieces()`の白駒定数をdomain形式に変更 |
| `src/maou/app/utility/stage2_data_generation.py` | 合法手ラベル生成を正規化済み盤面ベースに変更 |
| `tests/maou/app/visualization/test_board_renderer.py` | 白駒描画テスト追加 |
| `tests/maou/app/utility/test_stage2_data_generation.py` | 後手番ラベル正規化テスト追加 |
| `tests/maou/app/visualization/test_record_renderer.py` | E2Eテスト追加 |

---

## Task 1: 白駒描画のテストを書く (バグA)

**Files:**
- Modify: `tests/maou/app/visualization/test_board_renderer.py`

**Step 1: テストファイルの確認**

テストファイルが存在するか確認する．なければ作成する．

```bash
ls tests/maou/app/visualization/test_board_renderer.py 2>/dev/null || echo "NOT FOUND"
# 存在しない場合，tests/maou/domain/visualization/test_board_renderer.py も確認
ls tests/maou/domain/visualization/test_board_renderer.py 2>/dev/null || echo "NOT FOUND"
```

board_renderer.pyは `src/maou/domain/visualization/board_renderer.py` にあるため，テストは `tests/maou/domain/visualization/test_board_renderer.py` に作成する．

**Step 2: 失敗するテストを書く**

domain形式の白駒PieceId(15-28)がSVGに正しく描画されることを検証する．

```python
"""SVGBoardRendererのテスト．"""

import pytest

from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
    BoardPosition,
)
from maou.domain.board.shogi import Turn


class TestDrawPiecesWhitePieces:
    """_draw_pieces()のdomain形式白駒描画テスト．"""

    @pytest.fixture
    def renderer(self) -> SVGBoardRenderer:
        """テスト用SVGBoardRendererを作成．"""
        return SVGBoardRenderer()

    def test_white_pawn_renders_as_kanji(
        self, renderer: SVGBoardRenderer
    ) -> None:
        """domain形式の白歩(15)が「歩」として描画されることを検証する．"""
        # 9x9の空盤に白歩(domain PieceId=15)を配置
        board = [[0] * 9 for _ in range(9)]
        board[6][4] = 15  # row=6, col=4に白歩を配置

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=[0] * 14,
        )

        svg = renderer.render(position, turn=Turn.BLACK)

        # 「?」が含まれないことを検証
        assert "?" not in svg, (
            "White pawn (domain PieceId=15) rendered as '?' instead of kanji"
        )
        # 「歩」が含まれることを検証
        assert "歩" in svg

    def test_all_white_pieces_render_correctly(
        self, renderer: SVGBoardRenderer
    ) -> None:
        """domain形式の全白駒(15-28)が「?」ではなく漢字で描画されることを検証する．"""
        # 各白駒を1つずつ配置
        board = [[0] * 9 for _ in range(9)]
        white_pieces = list(range(15, 29))  # domain白駒: 15-28
        for i, piece_id in enumerate(white_pieces):
            row = i // 9
            col = i % 9
            board[row][col] = piece_id

        position = BoardPosition(
            board_id_positions=board,
            pieces_in_hand=[0] * 14,
        )

        svg = renderer.render(position, turn=Turn.BLACK)

        assert "?" not in svg, (
            "Some white pieces rendered as '?' - "
            "board_renderer may be using cshogi constants "
            "instead of domain constants"
        )
```

**Step 3: テストが失敗することを確認**

```bash
uv run pytest tests/maou/domain/visualization/test_board_renderer.py::TestDrawPiecesWhitePieces -v
```

期待: FAIL (`?` が出力される)

---

## Task 2: `_draw_pieces()` の白駒定数を修正する (バグA)

**Files:**
- Modify: `src/maou/domain/visualization/board_renderer.py` (L13-14, L449-458)

**Step 4: import文を修正**

```python
# 修正前 (board_renderer.py:13-14)
from maou.domain.board.shogi import (
    CSHOGI_WHITE_MIN,
    CSHOGI_WHITE_OFFSET,
    PieceId,
    Turn,
)

# 修正後
from maou.domain.board.shogi import (
    DOMAIN_WHITE_MIN,
    DOMAIN_WHITE_OFFSET,
    PieceId,
    Turn,
)
```

**Step 5: `_draw_pieces()`の白駒判定を修正**

```python
# 修正前 (board_renderer.py:449-458)
                # cshogiの駒ID: 先手=1-14, 後手=17-30（先手+16）
                # 定数は shogi.py の CSHOGI_* を使用
                is_white = piece_id >= CSHOGI_WHITE_MIN
                actual_piece_id = (
                    piece_id - CSHOGI_WHITE_OFFSET
                    if is_white
                    else piece_id
                )

# 修正後
                # domain形式の駒ID: 先手=1-14, 後手=15-28（先手+14）
                # boardIdPositionsはdomain PieceId形式
                is_white = piece_id >= DOMAIN_WHITE_MIN
                actual_piece_id = (
                    piece_id - DOMAIN_WHITE_OFFSET
                    if is_white
                    else piece_id
                )
```

**Step 6: テストがパスすることを確認**

```bash
uv run pytest tests/maou/domain/visualization/test_board_renderer.py::TestDrawPiecesWhitePieces -v
```

期待: PASS

**Step 7: 既存テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py -v
```

期待: ALL PASS

---

## Task 3: 後手番ラベル正規化の失敗テストを書く (バグB)

**Files:**
- Modify: `tests/maou/app/utility/test_stage2_data_generation.py`

**Step 8: 後手番データの合法手ラベルが可視化で正しくデコードされるテストを追加**

```python
    def test_white_turn_labels_decodable(
        self, tmp_path: Path
    ) -> None:
        """後手番局面の合法手ラベルが可視化時にUSIデコード可能であることを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 後手番の局面を作成 (1手進める)
        board = shogi.Board()
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])
        assert board.get_turn() == shogi.Turn.WHITE

        hcp_bytes = _board_to_hcp_bytes(board)
        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        row = df.row(0, named=True)

        # Stage2RecordRendererで合法手をデコード
        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.domain.visualization.move_label_converter import (
            MoveLabelConverter,
        )

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )
        fields = renderer.extract_display_fields(row)

        # <invalid:N> が含まれないことを検証
        legal_moves_str = fields["legal_moves"]
        assert "<invalid" not in legal_moves_str, (
            f"White turn labels not decodable: {legal_moves_str}"
        )
        # 合法手数が妥当であることを検証 (後手の初期合法手は30手)
        assert fields["legal_moves_count"] == 30
```

**Step 9: テストが失敗することを確認**

```bash
uv run pytest tests/maou/app/utility/test_stage2_data_generation.py::TestStage2DataGenerationUseCase::test_white_turn_labels_decodable -v
```

期待: FAIL (`<invalid:N>` が含まれる)

---

## Task 4: `stage2_data_generation.py` のラベル生成を修正する (バグB)

**Files:**
- Modify: `src/maou/app/utility/stage2_data_generation.py` (L249-264)

**Step 10: 正規化後の盤面で合法手ラベルを生成するように修正**

修正の方針: `make_board_id_positions(board)` で盤面を正規化した後，その正規化済み盤面をSFEN経由で再構築し，再構築した盤面(turn=BLACK)の合法手からラベルを生成する．

```python
# 修正前 (stage2_data_generation.py:246-264)
            for hash_id, hcp_bytes in rows:
                hcp_array = np.frombuffer(
                    hcp_bytes, dtype=np.uint8
                )
                board.set_hcp(hcp_array)

                # Generate features
                board_positions = make_board_id_positions(board)
                pieces_in_hand = make_pieces_in_hand(board)

                # Generate legal move labels
                legal_labels = np.zeros(
                    MOVE_LABELS_NUM, dtype=np.uint8
                )
                for move in board.get_legal_moves():
                    label = make_move_label(
                        board.get_turn(), move
                    )
                    legal_labels[label] = 1

# 修正後
            for hash_id, hcp_bytes in rows:
                hcp_array = np.frombuffer(
                    hcp_bytes, dtype=np.uint8
                )
                board.set_hcp(hcp_array)

                # Generate features (正規化: 後手番なら180度回転)
                board_positions = make_board_id_positions(board)
                pieces_in_hand = make_pieces_in_hand(board)

                # Generate legal move labels
                # 盤面は先手視点に正規化済みなので，
                # 正規化後の盤面の合法手からラベルを生成する
                legal_labels = np.zeros(
                    MOVE_LABELS_NUM, dtype=np.uint8
                )
                if board.get_turn() == shogi.Turn.BLACK:
                    # 先手番: 正規化なし，元のboardの合法手をそのまま使用
                    for move in board.get_legal_moves():
                        label = make_move_label(
                            shogi.Turn.BLACK, move
                        )
                        legal_labels[label] = 1
                else:
                    # 後手番: 盤面が180度回転されているため，
                    # 正規化後の盤面を再構築して合法手を取得
                    normalized_board = (
                        self._reconstruct_normalized_board(
                            board_positions, pieces_in_hand
                        )
                    )
                    for move in normalized_board.get_legal_moves():
                        label = make_move_label(
                            shogi.Turn.BLACK, move
                        )
                        legal_labels[label] = 1
```

**Step 11: `_reconstruct_normalized_board` ヘルパーメソッドを追加**

`Stage2DataGenerationUseCase` クラスに以下のメソッドを追加する:

```python
    @staticmethod
    def _reconstruct_normalized_board(
        board_positions: np.ndarray,
        pieces_in_hand: np.ndarray,
    ) -> shogi.Board:
        """正規化済み盤面からBoardを再構築する．

        make_board_id_positions()で先手視点に正規化された盤面と
        make_pieces_in_hand()の持ち駒から，cshogi.Boardを再構築する．
        再構築されたBoardはturn=BLACKとなる．

        Args:
            board_positions: 正規化済み9x9盤面配列(domain PieceId)
            pieces_in_hand: 正規化済み持ち駒配列(14要素)

        Returns:
            再構築されたBoardインスタンス(turn=BLACK)
        """
        from maou.app.visualization.record_renderer import (
            RecordRenderer,
        )

        # RecordRendererの_convert_to_sfenを利用してSFEN形式に変換
        # ここではRecordRendererのインスタンスメソッドを直接使えないため，
        # SFEN変換ロジックを呼び出す
        sfen = RecordRenderer._convert_to_sfen_static(
            board_id_positions=board_positions.tolist(),
            pieces_in_hand=pieces_in_hand.tolist(),
        )
        board = shogi.Board()
        board.set_sfen(sfen)
        return board
```

**注意:** `_convert_to_sfen` は現在インスタンスメソッドだが，`self` を使っていない．staticmethodに変換するか，あるいはSFEN変換ロジックをドメイン層に切り出す．

**代替案 (よりシンプル):** `_convert_to_sfen` を使わず，domain PieceIdから直接cshogiのBoardを構築する．

実装時は以下の判断をする:
1. `record_renderer.py` の `_convert_to_sfen` をstaticmethodにリファクタリングして共用する
2. もしくは，SFEN変換ロジックを `domain/board/` に切り出す
3. もしくは，`stage2_data_generation.py` 内にローカルな変換関数を書く

**推奨: 方法3** — 最も影響範囲が小さい．`_convert_to_sfen` のロジックは `record_renderer.py` から移植するのではなく，cshogiの `Board.set_position()` を使って直接構築する:

```python
    @staticmethod
    def _reconstruct_normalized_board(
        board_positions: np.ndarray,
        pieces_in_hand: np.ndarray,
    ) -> shogi.Board:
        """正規化済み盤面からBoardを再構築する．"""
        # domain PieceId → cshogi piece mapping
        # domain: 先手=1-14, 後手=15-28(先手+14)
        # cshogi: 先手=1-14, 後手=17-30(先手+16)
        DOMAIN_TO_CSHOGI_WHITE_ADJUST = 2  # 16 - 14

        # 盤面をcshogi形式に変換 (1次元81要素)
        # boardIdPositionsはcol=0が1筋の形式
        cshogi_board = []
        for row in board_positions:
            for piece_id in reversed(row):
                # 列を反転(SFENと同じ9筋→1筋順)
                if piece_id >= 15:  # domain白駒
                    cshogi_board.append(
                        int(piece_id) + DOMAIN_TO_CSHOGI_WHITE_ADJUST
                    )
                else:
                    cshogi_board.append(int(piece_id))

        # 持ち駒を cshogi 形式に変換
        # pieces_in_hand: [先手歩,香,桂,銀,金,角,飛, 後手歩,香,桂,銀,金,角,飛]
        hand = pieces_in_hand.tolist()
        black_hand = hand[:7]
        white_hand = hand[7:14]

        board = shogi.Board()
        board.set_position(
            cshogi_board,
            [tuple(black_hand), tuple(white_hand)],
        )
        return board
```

**実装時の注意:** `board.set_position()` のAPIを確認すること．cshogiの `Board.set_position()` は `(pieces, pieces_in_hand)` を受け取る形式かもしれないし，異なるかもしれない．実装時に `Board` クラスの `set_position` メソッドのシグネチャを確認し，適切に呼び出すこと．

もし `set_position` が使えない場合は，`record_renderer.py` の `_convert_to_sfen` と同等のSFEN変換ロジックを書いて `board.set_sfen()` を使う．

**Step 12: import文の追加**

`stage2_data_generation.py` の先頭に必要なimportを追加:

```python
from maou.domain.board import shogi  # 既にimportされているか確認
```

**Step 13: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/utility/test_stage2_data_generation.py::TestStage2DataGenerationUseCase::test_white_turn_labels_decodable -v
```

期待: PASS

---

## Task 5: 既存テストの後手番対応を確認する

**Files:**
- Modify: `tests/maou/app/utility/test_stage2_data_generation.py`

**Step 14: 既存の `test_legal_moves_labels_correctness` を更新**

現在のテストは初期局面(先手番)のみ検証している．後手番のケースを追加する:

```python
    def test_legal_moves_labels_correctness_white_turn(
        self, tmp_path: Path
    ) -> None:
        """後手番の合法手ラベルが正規化後の盤面と整合することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 後手番の局面を使用
        board = shogi.Board()
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])
        assert board.get_turn() == shogi.Turn.WHITE

        hcp_bytes = _board_to_hcp_bytes(board)
        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        assert len(df) == 1

        # 出力された合法手ラベルを取得
        legal_labels = df["legalMovesLabel"][0].to_list()
        active_labels = [
            i for i, val in enumerate(legal_labels) if val == 1
        ]

        # 正規化後の盤面を再構築してラベルをデコード
        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.domain.visualization.move_label_converter import (
            MoveLabelConverter,
        )
        from maou.domain.move.label import make_usi_move_from_label

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )
        row = df.row(0, named=True)
        reconstructed_board = renderer._create_board_from_record(row)

        # 全ラベルがUSIデコード可能であることを検証
        for label in active_labels:
            usi = make_usi_move_from_label(
                reconstructed_board, label
            )
            # デコードに失敗しなければOK (例外が出なければ通過)
```

**Step 15: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/utility/test_stage2_data_generation.py -v
```

期待: ALL PASS

---

## Task 6: E2E可視化テスト

**Files:**
- Modify: `tests/maou/app/visualization/test_record_renderer.py`

**Step 16: 実データ経由の`extract_display_fields`テストを追加**

TestConvertToSfenクラスに以下を追加:

```python
    def test_extract_display_fields_from_generated_data(
        self, renderer: Stage2RecordRenderer, tmp_path: Path
    ) -> None:
        """生成済みStage2データからextract_display_fieldsが正しく動作することを検証する．"""
        from pathlib import Path as P

        import numpy as np
        import polars as pl

        from maou.app.utility.stage2_data_generation import (
            Stage2DataGenerationConfig,
            Stage2DataGenerationUseCase,
        )
        from maou.domain.board import shogi
        from maou.domain.data.rust_io import load_stage2_df

        def board_to_hcp_bytes(board: shogi.Board) -> bytes:
            hcp = np.empty(32, dtype=np.uint8)
            board.to_hcp(hcp)
            return hcp.tobytes()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 先手番と後手番の両方のデータを生成
        board1 = shogi.Board()  # 先手番
        hcp1 = board_to_hcp_bytes(board1)

        board2 = shogi.Board()
        moves = list(board2.get_legal_moves())
        board2.push_move(moves[0])  # 後手番
        hcp2 = board_to_hcp_bytes(board2)

        n = 2
        df = pl.DataFrame({
            "hcp": pl.Series("hcp", [hcp1, hcp2], dtype=pl.Binary),
            "eval": pl.Series("eval", [100] * n, dtype=pl.Int16),
            "bestMove16": pl.Series("bestMove16", [0] * n, dtype=pl.Int16),
            "gameResult": pl.Series("gameResult", [1] * n, dtype=pl.Int8),
        })
        df.write_ipc(str(input_dir / "data.feather"))

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = P(result["output_files"][0])
        stage2_df = load_stage2_df(output_file)

        # 全レコードでextract_display_fieldsが<invalid>を含まないことを検証
        for i in range(len(stage2_df)):
            row = stage2_df.row(i, named=True)
            fields = renderer.extract_display_fields(row)
            legal_moves_str = fields["legal_moves"]
            assert "<invalid" not in legal_moves_str, (
                f"Record {i}: Invalid moves found: {legal_moves_str}"
            )
```

**Step 17: テストがパスすることを確認**

```bash
uv run pytest tests/maou/app/visualization/test_record_renderer.py::TestConvertToSfen::test_extract_display_fields_from_generated_data -v
```

期待: PASS

---

## Task 7: QAパイプラインとコミット

**Step 18: QAパイプライン**

```bash
uv run ruff format \
    src/maou/domain/visualization/board_renderer.py \
    src/maou/app/utility/stage2_data_generation.py \
    tests/maou/domain/visualization/test_board_renderer.py \
    tests/maou/app/utility/test_stage2_data_generation.py \
    tests/maou/app/visualization/test_record_renderer.py

uv run ruff check \
    src/maou/domain/visualization/board_renderer.py \
    src/maou/app/utility/stage2_data_generation.py \
    tests/maou/domain/visualization/test_board_renderer.py \
    tests/maou/app/utility/test_stage2_data_generation.py \
    tests/maou/app/visualization/test_record_renderer.py \
    --fix

uv run isort \
    src/maou/domain/visualization/board_renderer.py \
    src/maou/app/utility/stage2_data_generation.py \
    tests/maou/domain/visualization/test_board_renderer.py \
    tests/maou/app/utility/test_stage2_data_generation.py \
    tests/maou/app/visualization/test_record_renderer.py

uv run mypy \
    src/maou/domain/visualization/board_renderer.py \
    src/maou/app/utility/stage2_data_generation.py
```

**Step 19: 全テスト実行**

```bash
uv run pytest tests/maou/domain/visualization/test_board_renderer.py -v
uv run pytest tests/maou/app/visualization/test_record_renderer.py -v
uv run pytest tests/maou/app/utility/test_stage2_data_generation.py -v
```

期待: ALL PASS

**Step 20: コミット**

2つの独立したバグなので，可能であればコミットを分ける:

```bash
# コミット1: バグA (白駒描画)
git add src/maou/domain/visualization/board_renderer.py tests/maou/domain/visualization/test_board_renderer.py
git commit -m "fix(visualization): use domain PieceId constants for white piece rendering in SVGBoardRenderer"

# コミット2: バグB (合法手ラベル正規化)
git add src/maou/app/utility/stage2_data_generation.py tests/maou/app/utility/test_stage2_data_generation.py tests/maou/app/visualization/test_record_renderer.py
git commit -m "fix(utility): normalize legal move labels to BLACK perspective in Stage2 data generation"
```

---

## Task 8: 可視化動作確認とスクリーンショット

**Step 21: Stage2データを再生成**

バグB修正後のコードでStage2データを再生成する:

```bash
# テストデータ生成
uv run python -c "
import numpy as np
import polars as pl
import cshogi

def board_to_hcp_bytes(board):
    hcp = np.empty(32, dtype=np.uint8)
    board.to_hcp(hcp)
    return hcp.tobytes()

boards = []
board = cshogi.Board()
boards.append(board_to_hcp_bytes(board))

# 後手番のデータも含める
for i in range(5):
    b = cshogi.Board()
    legal = list(b.legal_moves)
    for j in range(i + 1):
        if legal:
            b.push(legal[0])
            legal = list(b.legal_moves)
    boards.append(board_to_hcp_bytes(b))

n = len(boards)
df = pl.DataFrame({
    'hcp': pl.Series('hcp', boards, dtype=pl.Binary),
    'eval': pl.Series('eval', [100] * n, dtype=pl.Int16),
    'bestMove16': pl.Series('bestMove16', [0] * n, dtype=pl.Int16),
    'gameResult': pl.Series('gameResult', [1] * n, dtype=pl.Int8),
})
import os
os.makedirs('/tmp/test_hcpe', exist_ok=True)
df.write_ipc('/tmp/test_hcpe/test.feather')
print(f'Created HCPE feather with {n} records')
"

# Stage2データ生成
rm -rf /tmp/test_stage2
uv run maou utility generate-stage2-data --input-path /tmp/test_hcpe/ --output-dir /tmp/test_stage2/
```

**Step 22: 可視化サーバー起動とスクリーンショット**

```bash
# サーバー起動
uv run maou visualize --input-path /tmp/test_stage2/stage2.feather --array-type stage2 --server-name 0.0.0.0 --port 7860 &
sleep 10

# スクリーンショット取得
uv run maou screenshot --url http://localhost:7860 --output /tmp/stage2-final.png --settle-time 5000 --width 1920 --height 1080 --no-full-page

# サーバー停止
lsof -ti :7860 | xargs kill -9 2>/dev/null
```

確認ポイント:
- [ ] 白駒(7-9段目)が漢字で表示される (「?」ではない)
- [ ] レコード詳細の合法手に `<invalid:N>` が含まれない
- [ ] 先手駒・後手駒が正しい位置にある
- [ ] 複数レコードをナビゲーションして異常がない

---

## 注意事項

- **バグBは破壊的変更**: Stage2データの再生成が必要．既存のStage2 featherファイルは修正後のコードと互換性がない
- **バグAとバグBは独立**: 別々にコミット可能．バグAのみ先に修正してもよい
- **`_reconstruct_normalized_board` の実装**: cshogiの`Board`クラスのAPIに依存する．`set_position()`が使えない場合はSFEN経由で構築する．実装時に`Board`クラスのメソッドを確認すること
- **テスト `test_legal_moves_labels_correctness` への影響**: 既存テストは先手番のみなのでバグB修正後もそのままパスするはず．ただし検証すること
