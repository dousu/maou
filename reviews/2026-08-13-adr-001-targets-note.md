---
status: pending
date: 2026-08-13
target: [docs/adr-001-dataloader-multiprocessing-optimization.md]
risk: low
reversibility: trivial
---

# ADR-001 の 2026-08-09 付の注が言う「3 要素目は省略可」が preprocessing 経路では成り立たない

## Trigger

`/audit-backlog` (2026-08-13, `ejdnzm`)．backlog 行
[2026-08-12 backlog arrow-format-and-clippy](../audits/2026-08-12-backlog-arrow-format-and-clippy.md)
N-3 の再検証．

## 検証結果 (HEAD `05654ba`)

`docs/adr-001-dataloader-multiprocessing-optimization.md:141` の注:

> 現在 `legal_move_mask` はどのデータ経路も供給せず，targets は
> `(labels_policy, labels_value, move_win_rate)` (3 要素目は省略可)
> になっている．

`KifDataset.__getitem__` (`src/maou/app/learning/dataset.py:137-142`) は
`data.dtype.names` に `moveWinRate` があるときだけ 3 要素を返す:

```python
if self._has_move_win_rate is None:
    self._has_move_win_rate = (
        data.dtype.names is not None
        and "moveWinRate" in data.dtype.names
    )
```

`moveWinRate` は preprocessing の structured dtype に載っている
(`src/maou/domain/data/schema.py:164`，2026-08-12 の PR #487 で追加)．
したがって **preprocessing 経路では `_has_move_win_rate` が常に True** で，
2 要素側の分岐には到達しない．「省略可」は，コードの形としては真
(分岐は残っている) だが，実際に走る経路の説明としては誤読を招く．

## なぜ P2 (drift correction) ではないか

訂正後の本文が現行コードから**一意に決まらない**．日付入りの ADR 注を
どう扱うかに，少なくとも 3 通りの妥当な書き方がある:

1. **当時の記述として残し，追記する** — 注は 2026-08-09 時点の記述だと
   明示したうえで「2026-08-12 以降 preprocessing は常に 3 要素」を
   別行で足す．ADR の履歴性を最も尊重する．
2. **書き換える** — 注の本文自体を現状に合わせて直す．読者が最新の
   事実に一度で辿り着くが，注の日付と内容がずれる．
3. **注を消して本文の例を直す** — 上のコード例そのものを現在の
   targets に差し替え，注を不要にする．ADR のコード例が「当時の姿」で
   なくなる．

どれも現行コードと矛盾しないので，`/audit-backlog` の P2 が持つ
「一意に決まるか」の条件を満たさない．CLAUDE.md の standing approval は
及ばないため，本提案は `status: pending` のままで**適用していない**．

## 提案 (案 1 を推す)

ADR は決定の記録であり，本文は当時の姿を保つのが原則．注の追記が
最も情報を失わない．

### Before (`docs/adr-001-dataloader-multiprocessing-optimization.md:141-147`)

```markdown
> **注 (2026-08-09)**: 上のコード例は当時の targets タプルである．
> 現在 `legal_move_mask` はどのデータ経路も供給せず，targets は
> `(labels_policy, labels_value, move_win_rate)` (3 要素目は省略可)
> になっている．経緯は [loss-functions.md](loss-functions.md) の
> 「合法手マスキング」節を参照．本 ADR の主題である
> DataLoader/GPU 転送の決定自体は有効である．
```

### After

```markdown
> **注 (2026-08-09)**: 上のコード例は当時の targets タプルである．
> 現在 `legal_move_mask` はどのデータ経路も供給せず，targets は
> `(labels_policy, labels_value, move_win_rate)` になっている．
> 経緯は [loss-functions.md](loss-functions.md) の
> 「合法手マスキング」節を参照．本 ADR の主題である
> DataLoader/GPU 転送の決定自体は有効である．
>
> **追記 (2026-08-13)**: `moveWinRate` が preprocessing の
> structured dtype に入った (2026-08-12) ため，preprocessing 経路の
> `KifDataset` は**常に 3 要素**を返す
> (`app/learning/dataset.py:137-142`)．2 要素を返す分岐は
> `moveWinRate` を持たない dtype 向けに防御的に残っているだけで，
> 現在のどのデータ経路からも到達しない．
```

## 判断が要る点

案 1 / 2 / 3 のどれを採るか．承認をもらえた案をこの run か次の run で
適用し，`status: applied` + `applied_in: <sha>` を埋める．
