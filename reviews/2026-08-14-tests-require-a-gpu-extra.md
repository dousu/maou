---
status: pending
date: 2026-08-14
target: [docs/testing-guide.md]
risk: low
reversibility: trivial
---

# テスト実行が GPU extra を前提とすることが doc に書かれていない

## Trigger

`/audit-backlog` (2026-08-14)．backlog の **N4** 行について，step 3d で
ユーザに設計判断を問い，**「CPU extra を必須化」**という回答を得た．
この回答を `tests/conftest.py` に実装した結果として生じる doc drift．

## 決定の出所 (P2 の恒久承認では**ない**)

この提案は CLAUDE.md § "Standing approval — drift corrections only" の
P2 恒久承認では**カバーされない**．訂正後の本文が現行コードから一意に
決まる drift correction ではなく，**この run でユーザが選んだ新しい方針**
だからである．

適用の根拠は 3d でユーザが明示的に選んだこと (judgment band の承認) で
あって，standing approval ではない．記録にもそう書く．

## 検証結果 (HEAD `2312f65`)

`tests/conftest.py` は collect 段の `ModuleNotFoundError` を
`_OPTIONAL_DEPS` に載っている依存についてのみ skip へ書き換える
(`pytest_make_collect_report` `:97-121`)．書き換えは **collector 粒度**
なので，モジュールが丸ごと消える．

`torch` がその集合に入っていたため，GPU extra を入れていない環境では
`tests/maou/app/learning/` をはじめ torch を import する全モジュールが
消え，**残ったものだけが緑として報告されていた**．この状態は 2026-08-12
から 2026-08-14 にかけて 8 run 連続で再確認されている．

2026-08-14 に `pytest_terminal_summary` (`:193-204`) を足して「黙って
消える」ことは解消したが，**緑に見えること自体は変わっていない**．

この run で `torch` を `_OPTIONAL_DEPS` から外したので，GPU extra 無しの
環境では collect が**失敗として**報告されるようになった．

`docs/testing-guide.md` はこの前提をどこにも書いていない．冒頭の

```bash
uv run pytest                           # Run all tests
```

は，extra 無しでも全テストが走るかのように読める．

## 提案

### Before (`docs/testing-guide.md:3-12`)

```markdown
## Testing Requirements

**Framework**: Use `uv run pytest`

​```bash
uv run pytest                           # Run all tests
uv run pytest --cov=src/maou            # Run with coverage
TEST_GCP=true uv run pytest             # Test GCP features
TEST_AWS=true uv run pytest             # Test AWS features
​```
```

### After

```markdown
## Testing Requirements

**Framework**: Use `uv run pytest`

### 前提: GPU extra が要る

テストスイートの実行には `torch` が要る．`uv sync` の base install だけ
では `tests/maou/app/learning/` をはじめ torch を import するモジュール
が collect 段で失敗する．**先に GPU extra を入れること．**

​```bash
uv sync --extra cpu                     # または --extra cuda
​```

`torch` は `tests/conftest.py` の `_OPTIONAL_DEPS` に**敢えて入れて
いない**．collect 段の skip はモジュールを丸ごと落とすため，torch を
そこに入れると「環境が整っていない」実行が緑として報告されてしまう．
`onnxruntime` / `onnx` / `gradio` / `matplotlib` は該当モジュールが
局所的なので，従来どおり skip に書き換えられる．

​```bash
uv run pytest                           # Run all tests
uv run pytest --cov=src/maou            # Run with coverage
TEST_GCP=true uv run pytest             # Test GCP features
TEST_AWS=true uv run pytest             # Test AWS features
​```
```

## 影響

- `docs/testing-guide.md` のみ．コードの変更は `tests/conftest.py` と
  `tests/test_conftest_optional_deps.py` (どちらも P1，version bump 不要)．
- 既存の CI / 開発フローで既に `--extra cpu` を入れている場合は無変化．
