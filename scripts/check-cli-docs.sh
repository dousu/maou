#!/bin/bash
# Pre-commit hook: CLI ファイル変更時にドキュメント更新を強制する．
#
# src/maou/infra/console/ 配下のコマンドファイルがステージングされた場合，
# 対応する docs/commands/*.md もステージングされているか検証する．
set -euo pipefail

CLI_DIR="src/maou/infra/console/"
DOC_DIR="docs/commands/"

# マッピングテーブル: CLIファイル名 → ドキュメントファイル名
declare -A CLI_DOC_MAP=(
  ["learn_model.py"]="learn_model.md"
  ["pretrain_cli.py"]="pretrain.md"
  ["evaluate_board.py"]="evaluate.md"
  ["hcpe_convert.py"]="hcpe_convert.md"
  ["pre_process.py"]="pre_process.md"
  ["visualize.py"]="visualize.md"
  ["screenshot.py"]="screenshot.md"
)

# utility.py は特殊: 関連ドキュメントのうち少なくとも1つがstagedであればパス
UTILITY_DOCS=(
  "utility_benchmark_dataloader.md"
  "utility_benchmark_training.md"
  "generate-stage1-data.md"
  "generate-stage2-data.md"
)

# 除外ファイル（マッピング不要）
EXCLUDED=("app.py" "common.py" "__init__.py")

# ステージングされたファイルを1パスで分類（空白を含むパスにも対応）
CLI_FILES=()
STAGED_DOCS=()
while IFS= read -r file; do
  case "$file" in
    "${CLI_DIR}"*.py)
      CLI_FILES+=("$(basename "$file")")
      ;;
    "${DOC_DIR}"*)
      STAGED_DOCS+=("$(basename "$file")")
      ;;
  esac
done < <(git diff --cached --name-only)

# CLI ファイルがなければ即パス
if [ ${#CLI_FILES[@]} -eq 0 ]; then
  exit 0
fi

has_error=0

for cli_file in "${CLI_FILES[@]}"; do
  # 除外ファイルはスキップ
  is_excluded=0
  for excluded in "${EXCLUDED[@]}"; do
    if [ "$cli_file" = "$excluded" ]; then
      is_excluded=1
      break
    fi
  done
  if [ "$is_excluded" -eq 1 ]; then
    continue
  fi

  # utility.py の特殊処理
  if [ "$cli_file" = "utility.py" ]; then
    found=0
    for doc in "${UTILITY_DOCS[@]}"; do
      for staged_doc in "${STAGED_DOCS[@]}"; do
        if [ "$doc" = "$staged_doc" ]; then
          found=1
          break 2
        fi
      done
    done
    if [ "$found" -eq 0 ]; then
      echo "ERROR: CLI file 'utility.py' was modified but none of the related docs are staged." >&2
      echo "HINT: Stage at least one of: ${UTILITY_DOCS[*]}" >&2
      echo "      (in ${DOC_DIR})" >&2
      has_error=1
    fi
    continue
  fi

  # マッピングテーブルに存在するか確認
  if [ -z "${CLI_DOC_MAP[$cli_file]+x}" ]; then
    echo "ERROR: CLI file '${cli_file}' has no mapping in check-cli-docs.sh." >&2
    echo "HINT: Add it to the CLI_DOC_MAP in scripts/check-cli-docs.sh." >&2
    has_error=1
    continue
  fi

  # 対応ドキュメントがstagedにあるか確認
  expected_doc="${CLI_DOC_MAP[$cli_file]}"
  found=0
  for staged_doc in "${STAGED_DOCS[@]}"; do
    if [ "$expected_doc" = "$staged_doc" ]; then
      found=1
      break
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "ERROR: CLI file '${cli_file}' was modified but '${DOC_DIR}${expected_doc}' is not staged." >&2
    echo "HINT: Update the documentation and stage it, or verify no CLI options changed." >&2
    has_error=1
  fi
done

exit "$has_error"
