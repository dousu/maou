---
status: applied
applied_in: cae7445
date: 2026-08-14
target: [docs/rust-backend.md]
risk: low
reversibility: trivial
---

# `docs/rust-backend.md` が `.npy` と `.feather` の「併存」を謳っている

## Trigger

`/audit-backlog` (2026-08-14)．N9 (`scripts/` に残る `.npy` 遺物) の
再検証で `.npy` の残存箇所を走査していて見つけた．backlog の行ではなく
**この run が新たに気づいた所見**である．

## 検証結果 (HEAD `001d16e`)

`docs/rust-backend.md` § "File Format Migration" (`:809`) が

> **Note**: Both formats are currently supported．Gradual migration recommended．

と書いているが，`.npy` は**もう 1 経路も受け付けない**．

| 経路 | 拒否箇所 |
|---|---|
| `FileDataSource` | `Only .feather files are supported. Got: {suffix}` |
| object storage | 同上 |
| BigQuery ローカルキャッシュ | 同上 |

`src/maou/` 全体で `.npy` を読み書きするコードは**ゼロ**である
(残っているのは `infra/utility/benchmark_polars_io.py:5,8` の
「以前は `.npy` と比較していた」という履歴コメントだけで，比較経路は
2026-08-13 の N-2 で削除済み)．

しかも**同じファイルの `:726`** が既に

> データソースが受け付けるのは `.feather` (Arrow IPC) だけである．`.npy` は
> ... `Only .feather files are supported` で拒否される．

と書いており，`:809` と**自己矛盾**している．読者はどちらが正しいのか
判断できない．

## 提案

### Before (`docs/rust-backend.md:809`)

```markdown
**Note**: Both formats are currently supported．Gradual migration recommended．
```

### After

```markdown
**Note**: 移行は完了しており，`.npy` は**もう受け付けない**．
`FileDataSource` / object storage / BigQuery キャッシュのいずれも
`Only .feather files are supported` で拒否する．`.npy` で保存した
データは `.feather` へ変換してから使うこと．
```

§ "Legacy Format" の見出しと `.npy` の説明そのものは残す — 移行の
経緯を説明する節なので，「かつて何だったか」の記述は正しい．誤りなのは
**現在も併存している**という 1 文だけである．

## P2 判定 (drift correction)

訂正後の本文はコードから**一意に決まる**: `.npy` は 3 経路すべてで
`ValueError` になり，`src/` に読み書きコードが無い．「まだサポート
されている」と書ける読み方は存在しない．したがって CLAUDE.md の
常設承認 (§ "Standing approval — drift corrections only") が適用され，
この run 内で適用する．
