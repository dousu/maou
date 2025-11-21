---
name: japanese-doc-validator
description: Validate Japanese documentation punctuation rules including full-width comma and period usage (，．), half-width parenthesis validation, and format compliance checking. Use when writing Japanese docstrings, validating Japanese comments, or ensuring Japanese text formatting standards.
allowed-tools: Read, Grep, Glob
---

# Japanese Documentation Validator

Validates Japanese text formatting rules for the Maou project.

## Japanese Punctuation Rules

The project enforces specific Japanese punctuation standards:

### 句読点 (Punctuation Marks)

**句点 (Period)**: `．` (全角ピリオド / Full-width period)
- Unicode: U+FF0E
- NOT: `。` (句点)
- NOT: `.` (Half-width period)

**読点 (Comma)**: `，` (全角コンマ / Full-width comma)
- Unicode: U+FF0C
- NOT: `、` (読点)
- NOT: `,` (Half-width comma)

### 括弧 (Parentheses)

**Brackets**: `()` (半角括弧 / Half-width parentheses)
- Use half-width: `(example)`
- NOT full-width: `（example）`

## Correct Format Examples

### Correct Docstring

```python
def process_shogi_game(game_data: str) -> ProcessingResult:
    """
    将棋の棋譜データを処理し，HCPE形式に変換する．

    Args:
        game_data: CSA形式またはKIF形式の棋譜データ

    Returns:
        変換結果を含むProcessingResultオブジェクト

    Raises:
        ValueError: 入力形式が不正な場合
    """
```

**Key points**:
- `処理し，HCPE形式に` - Full-width comma (，)
- `変換する．` - Full-width period (．)
- `(Japanese chess)` - Half-width parentheses

### Correct Comment

```python
# 学習率を調整し，収束を安定化させる．
learning_rate = 0.001

# データを前処理し，正規化を適用する(0-1スケール)．
normalized_data = preprocess(data)
```

## Validation Methods

### Search for Incorrect Punctuation

#### Find Half-width Periods in Japanese Text

```bash
# Search for lines with Japanese and half-width periods
grep -n "[ぁ-ん][^．]*\." src/maou/ | grep "\.py:" | head -20
```

Should return: (empty - no violations)

#### Find Half-width Commas in Japanese Text

```bash
# Search for lines with Japanese and half-width commas
grep -n "[ぁ-ん][^，]*," src/maou/ | grep "\.py:" | head -20
```

Should return: (empty - no violations)

#### Find Japanese Periods (。)

```bash
# Search for incorrect Japanese period
grep -n "。" src/maou/*.py src/maou/**/*.py
```

Should return: (empty - should use ．instead)

#### Find Japanese Commas (、)

```bash
# Search for incorrect Japanese comma
grep -n "、" src/maou/*.py src/maou/**/*.py
```

Should return: (empty - should use ，instead)

### Search for Full-width Parentheses

```bash
# Find full-width parentheses (should be half-width)
grep -n "（\|）" src/maou/*.py src/maou/**/*.py
```

Should return: (empty - no violations)

## Common Violations

### ❌ WRONG: Using 。and 、

```python
def convert_hcpe(data: str) -> int:
    """
    将棋の棋譜データを処理し、HCPE形式に変換する。
    """
```

**Issues**:
- Uses `、` (Japanese comma) instead of `，` (full-width comma)
- Uses `。` (Japanese period) instead of `．` (full-width period)

### ❌ WRONG: Using Half-width Punctuation

```python
def train_model(config: TrainingConfig) -> None:
    """
    モデルを学習し, 結果を保存する.
    """
```

**Issues**:
- Uses `,` (half-width comma) instead of `，`
- Uses `.` (half-width period) instead of `．`

### ❌ WRONG: Using Full-width Parentheses

```python
def load_data(path: Path) -> Data:
    """
    データをロードする（ファイルシステムから）．
    """
```

**Issue**:
- Uses `（）` (full-width) instead of `()` (half-width)

### ✓ CORRECT: Proper Format

```python
def convert_and_save(input_path: Path, output_path: Path) -> int:
    """
    棋譜を変換し，HCPE形式でファイルに保存する．

    Args:
        input_path: 入力ファイルのパス(CSA形式またはKIF形式)
        output_path: 出力ファイルのパス

    Returns:
        変換されたレコードの数

    Note:
        ファイルが存在しない場合，新規作成される．
    """
```

**Correct features**:
- `変換し，HCPE形式` - Full-width comma (，)
- `保存する．` - Full-width period (．)
- `パス(CSA形式` - Half-width parentheses ()

## File-by-File Validation

### Check Specific File

```bash
# Check domain layer file
grep -n "[。、]" src/maou/domain/data/schema.py

# Check app layer file
grep -n "[。、]" src/maou/app/learning/training_loop.py

# Check interface layer
grep -n "[。、]" src/maou/interface/converter.py
```

### Check All Python Files

```bash
# Find all files with violations
find src/maou -name "*.py" -exec grep -l "[。、（）]" {} \;
```

### Generate Violation Report

```bash
# Create detailed report
echo "=== Japanese Punctuation Violations ==="
echo ""
echo "Files with incorrect periods (。):"
grep -r "。" src/maou/*.py src/maou/**/*.py | cut -d: -f1 | sort -u
echo ""
echo "Files with incorrect commas (、):"
grep -r "、" src/maou/*.py src/maou/**/*.py | cut -d: -f1 | sort -u
echo ""
echo "Files with full-width parentheses (（）):"
grep -r "（\|）" src/maou/*.py src/maou/**/*.py | cut -d: -f1 | sort -u
```

## Automatic Correction

### Manual Replacement Patterns

If violations found, use these replacements:

```bash
# Replace Japanese period with full-width period
sed -i 's/。/．/g' file.py

# Replace Japanese comma with full-width comma
sed -i 's/、/，/g' file.py

# Replace full-width parentheses with half-width
sed -i 's/（/(/g' file.py
sed -i 's/）/)/g' file.py
```

**Warning**: Always review changes before committing!

## Unicode Reference

### Correct Characters

| Character | Name | Unicode | Usage |
|-----------|------|---------|-------|
| ，| Full-width comma | U+FF0C | Sentence separation |
| ．| Full-width period | U+FF0E | Sentence end |
| ( | Half-width open paren | U+0028 | Parenthetical |
| ) | Half-width close paren | U+0029 | Parenthetical |

### Incorrect Characters (Do Not Use)

| Character | Name | Unicode | Why Wrong |
|-----------|------|---------|-----------|
| 、| Japanese comma | U+3001 | Wrong style |
| 。| Japanese period | U+3002 | Wrong style |
| （| Full-width open paren | U+FF08 | Too wide |
| ）| Full-width close paren | U+FF09 | Too wide |

## Integration with Code Review

### Pre-commit Check

Add to pre-commit validation:

```bash
# Check for violations before commit
VIOLATIONS=$(grep -r "[。、（）]" src/maou/*.py src/maou/**/*.py || true)

if [ -n "$VIOLATIONS" ]; then
    echo "❌ Japanese punctuation violations found:"
    echo "$VIOLATIONS"
    exit 1
else
    echo "✓ Japanese punctuation is correct"
fi
```

## Editor Configuration

### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.py": "python"
  },
  "editor.unicodeHighlight.ambiguousCharacters": false,
  "editor.unicodeHighlight.invisibleCharacters": false
}
```

### Input Method Configuration

When typing Japanese:
1. Use full-width mode (全角)
2. Type punctuation in full-width
3. Switch to half-width (半角) for parentheses
4. Or paste from this reference: `，．()`

## Validation Report Format

```
Japanese Documentation Validation Report
=======================================

Files checked: 127

Violations found:
- Incorrect periods (。): 0
- Incorrect commas (、): 0
- Full-width parentheses (（）): 0

Status: ✓ ALL JAPANESE TEXT COMPLIANT
```

## When to Use

- Before committing Japanese documentation
- During code review of Japanese text
- After writing Japanese docstrings
- When validating imported Japanese code
- Before release preparation
- During documentation updates

## Common Questions

**Q: Why not use 。and 、?**
A: Project standard prefers full-width comma and period (，．) for consistency with technical writing.

**Q: What about English text in docstrings?**
A: English text uses normal punctuation (`,` and `.`).

**Q: Mixed Japanese/English sentences?**
A: Use Japanese rules for Japanese portions, English rules for English portions.

**Q: What about markdown files?**
A: Same rules apply to `.md` files with Japanese content.

## References

- **CLAUDE.md**: Japanese punctuation rules (lines 370-393)
- **AGENTS.md**: Japanese documentation standards (lines 259-283)
- Unicode standard: https://www.unicode.org/charts/
- Japanese typography: JIS Z 8301
