# Shogi Visualization Conventions

When implementing visualization features for `maou visualize`, Claude Code must understand Shogi-specific conventions to avoid incorrect coordinate mappings, piece placements, and display ordering. This document contains the three critical rules that cause the most implementation errors.

## Critical Conventions

### 1. Board Coordinate System (CRITICAL)

**Array Structure (Fortran Order):**

The `Board.get_board_id_positions_df()` method returns board positions in **Fortran order (column-major)**:

```python
positions = self.board.pieces.reshape((9, 9), order="F")
# CRITICAL: Fortran order means positions[col][row], NOT positions[row][col]
```

**Array Index Mapping:**

```python
# Array indices to 筋 (file) numbers:
# col: 0 → 筋1 (画面右端, screen right edge)
# col: 1 → 筋2
# col: 2 → 筋3
# ...
# col: 7 → 筋8
# col: 8 → 筋9 (画面左端, screen left edge)
```

**Screen Display Convention:**

When displaying from Black's perspective (先手視点):
- **Screen left edge (画面左端)**: 筋9
- **Screen right edge (画面右端)**: 筋1

**Coordinate Transformation for Rendering:**

```python
# Convert array column index to visual column position
visual_col = 8 - col

# Examples:
# col=0 (筋1) → visual_col=8 → rightmost screen position
# col=8 (筋9) → visual_col=0 → leftmost screen position

# Convert visual column to 筋 number
col_number = col + 1  # or equivalently: 9 - visual_col
```

**IMPORTANT - Incorrect Comment:**

The comment in `board_renderer.py:246` states `"col: 0=右端(筋9), 8=左端(筋1)"` which is **INCORRECT**. The actual mapping is:
- `col: 0 → 筋1 (画面右端)`
- `col: 8 → 筋9 (画面左端)`

The transformation logic in `board_renderer.py:264-265` is correct despite the incorrect comment.

**Common Mistake:**

Assuming `positions[row][col]` (row-major order) leads to transposed or mirrored board displays. Always remember: **Fortran order = `positions[col][row]`**.

**References:**
- `src/maou/domain/board/shogi.py:396-401` - Fortran order reshape
- `src/maou/domain/visualization/board_renderer.py:264-265` - Correct transformation
- `src/maou/domain/visualization/board_renderer.py:436-476` - Coordinate label rendering

### 2. Initial Position Reference

**Standard SFEN Initial Position:**

```
lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
```

**SFEN Notation:**
- **Uppercase letters** = Black pieces (先手, e.g., `B` = Black Bishop)
- **Lowercase letters** = White pieces (後手, e.g., `b` = White Bishop)
- **Direction**: Left-to-right in SFEN string corresponds to 筋9→筋1 (left-to-right on screen)

**Black's Second Rank (`1B5R1`):**

Reading left-to-right in SFEN:
- Position 0 (筋9): empty
- Position 1 (筋8): **角 (Bishop, `B`)**
- Positions 2-6 (筋7-筋3): empty (5 squares)
- Position 7 (筋2): **飛車 (Rook, `R`)**
- Position 8 (筋1): empty

**Array Representation:**
- **筋8 (col=7)**: 角 (Bishop) → `visual_col=1` → screen left-ish area
- **筋2 (col=1)**: 飛車 (Rook) → `visual_col=7` → screen right-ish area

**Visual Layout (先手視点):**
- **Screen left side**: 角 (Bishop)
- **Screen right side**: 飛車 (Rook)

**cshogi Integration:**

The cshogi library uses square numbering: `square = col * 9 + row` where `col=0→筋1, col=8→筋9`. The `Board` class in `shogi.py` encapsulates cshogi and handles piece ID conversions.

**References:**
- `src/maou/domain/board/shogi.py:425` - SFEN example in docstring
- `src/maou/domain/board/shogi.py` - cshogi encapsulation and piece ID mapping

### 3. Captured Pieces (持ち駒) Display Order

**Array Structure:**

```python
pieces_in_hand[0:7]   # Black's hand (先手の持ち駒)
pieces_in_hand[7:14]  # White's hand (後手の持ち駒)
```

**Ideal Display Order (価値順 - Value Descending):**

```python
# Display from top to bottom (or left to right):
["飛", "角", "金", "銀", "桂", "香", "歩"]

# Corresponding array indices:
# pieces_in_hand[6]: 飛 (Rook)
# pieces_in_hand[5]: 角 (Bishop)
# pieces_in_hand[4]: 金 (Gold)
# pieces_in_hand[3]: 銀 (Silver)
# pieces_in_hand[2]: 桂 (Knight)
# pieces_in_hand[1]: 香 (Lance)
# pieces_in_hand[0]: 歩 (Pawn)
```

**Current Implementation:**

`board_renderer.py:77-86` defines `HAND_PIECE_NAMES` in **reverse order** (歩→飛), which displays pieces from lowest to highest value. This is acceptable but not the ideal ordering.

**Display Format:**
- Single piece: Display name only (e.g., `"飛"`)
- Multiple pieces: Display name with count (e.g., `"歩×5"`)
- Only show pieces where `count > 0`

**References:**
- `src/maou/domain/visualization/board_renderer.py:77-86` - HAND_PIECE_NAMES definition
- `src/maou/domain/visualization/board_renderer.py:326-434` - Hand piece rendering logic

## Common Implementation Pitfalls

1. **Forgetting Fortran Order**: Accessing `positions[row][col]` instead of `positions[col][row]` causes transposed board display.

2. **Wrong Visual Transformation**: Forgetting `visual_col = 8 - col` or implementing it incorrectly leads to mirrored boards.

3. **Incorrect Initial Positions**: Placing 角 on the right and 飛車 on the left (opposite of correct placement).

4. **Wrong Hand Piece Ordering**: Displaying captured pieces in storage order instead of value-descending order.

5. **Trusting Incorrect Comments**: The comment at `board_renderer.py:246` contradicts the actual implementation. Always verify with the transformation code.

## Detailed Implementation References

**Core Implementation Files:**
- `src/maou/domain/visualization/board_renderer.py` - Complete SVG rendering logic with coordinate transformations
- `src/maou/domain/board/shogi.py` - Board abstraction, cshogi encapsulation, piece ID conversions
- `src/maou/domain/visualization/piece_mapping.py` - Piece rendering and rotation logic

**Design Documentation:**
- `docs/visualization/design.md` - Comprehensive visualization design specification
- `docs/visualization/UI_UX_REDESIGN.md` - Modern UI/UX guidelines and color schemes
