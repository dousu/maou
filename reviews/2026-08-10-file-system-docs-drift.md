---
status: applied
applied_in: 1c6a442
date: 2026-08-10
target:
  - docs/commands/utility_benchmark_dataloader.md
  - docs/commands/utility_benchmark_training.md
  - docs/commands/pre_process.md
  - docs/rust-backend.md
risk: low
reversibility: trivial (4ファイルの局所置換)
---

# infra/file_system を説明する docs のドリフト (.npy / bit-pack / cache_mode)

## Trigger

`/audit-and-fix src/maou/infra/file_system high` の step 4．
`<path>` を説明する doc を洗い出して各主張を code と照合した結果，
4ファイルに **stale** 3件・**wrong** 3件を確認した (HEAD `64afa41`)．

## Motivation

いずれも「真偽値のある主張」であり，コードと突き合わせれば判定できる．
3つのテーマに分かれる．

1. **Arrow IPC 移行の取りこぼし**．`.npy` を読むと書いてあるが，
   実際は Arrow IPC (`.feather`)．`docs/adr-004-arrow-ipc-migration.md`
   の移行から漏れた記述．
2. **`--input-file-packed` / bit-pack**．オプション自身の help が
   `"[Deprecated] ... This option has no effect."` と言っている
   (`pre_process.py:33`)．doc だけが機能すると書いている．
3. **`cache_mode` の意味**．「`file` は通常のファイル I/O，`memory` は
   RAM へコピー」と読めるが，実際は **どちらのモードでも `__init__` で
   全ファイルを RAM へ読み込む**．差はファイルごとの配列を保つか単一
   配列へ結合するかだけで，常駐量の軸ではない
   (`file_data_source.py:321-422`, `:428-436`)．メモリを気にして
   `file` を選ぶ利用者を誤導する．

`docs/rust-backend.md` の例は単に古いのではなく **動かない**:
`cache_mode="mmap"` は `FileDataSource.__init__` の検証
(`file_data_source.py:254-261`，許可値は `"file"` / `"memory"`) で
`ValueError` になる．CLI 側は `mmap` を `file` へ正規化してから渡すので
CLI 経由では起きないが，doc がそのまま示している Python API 直呼びでは
落ちる．

なお **正確だったもの** も記録しておく (再確認を防ぐため):
`docs/architecture.md:158-160` の「`array_type` Literal の正準定義は
`file_data_source.py`」という主張は正しい．step 2.5 の sweep で
Python 15箇所 + Rust enum を照合し，メンバ集合はすべて一致していた．

## Proposed change

### 1. `docs/commands/utility_benchmark_dataloader.md:19`

before:
```markdown
| `--stage3-data-path PATH` | one of the sources | Reads local `.npy` tensors for Stage 3 (policy+value) benchmarking via `FileDataSource.FileDataSourceSpliter` with optional bit-packed decoding (`--input-file-packed`). `--sample-ratio` is ignored because the files already reside on disk.【F:src/maou/infra/console/utility.py†L217-L274】 |
```
after:
```markdown
| `--stage3-data-path PATH` | one of the sources | Reads local Arrow IPC (`.feather`) files for Stage 3 (policy+value) benchmarking via `FileDataSource.FileDataSourceSpliter`. `--sample-ratio` is ignored because the files already reside on disk.【F:src/maou/infra/console/utility.py†L217-L274】 |
```

### 2. `docs/commands/utility_benchmark_dataloader.md:23`

before:
```markdown
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/utility.py†L72-L78】 |
```
after:
```markdown
| `--input-cache-mode {file,memory,mmap}` | default `file` | In-memory layout for local inputs. **Both modes load every file into RAM**; the difference is that `file` keeps one array per input file while `memory` concatenates them into a single array (removing one `searchsorted` per sample, at the cost of a transient 2x memory spike during concatenation). This is not a residency knob — to avoid holding the dataset, use the streaming path instead. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/file_system/file_data_source.py†L321-L436】【F:src/maou/infra/console/utility.py†L72-L78】 |
```

### 3. `docs/commands/utility_benchmark_training.md:30`

before:
```markdown
| `--stage3-data-path PATH` + optional `--input-file-packed` | one of the sources | Streams local `.npy` shards for Stage 3 (policy+value) benchmarking and can unpack bit-packed HCPE tensors. Supplying `--sample-ratio` here logs a warning because every file is already on disk.【F:src/maou/infra/console/utility.py†L520-L821】 |
```
after:
```markdown
| `--stage3-data-path PATH` | one of the sources | Streams local Arrow IPC (`.feather`) shards for Stage 3 (policy+value) benchmarking. Supplying `--sample-ratio` here logs a warning because every file is already on disk.【F:src/maou/infra/console/utility.py†L520-L821】 |
```

### 4. `docs/commands/utility_benchmark_training.md:36`

before:
```markdown
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/utility.py†L469-L475】 |
```
after: 2 と同一文面 (引用は `†L469-L475` のまま)．

### 5. `docs/commands/pre_process.md:20`

before:
```markdown
| Local filesystem | `--input-path PATH` (file or directory), optional `--input-file-packed` | Walks recursively via `FileSystem.collect_files` and decodes bit-packed numpy payloads when requested.【F:src/maou/infra/console/pre_process.py†L16-L66】 |
```
after:
```markdown
| Local filesystem | `--input-path PATH` (file or directory) | Walks recursively via `FileSystem.collect_files`, which returns a **sorted** list and skips cloud-download temp artifacts (`.gstmp`/`.tmp`/`.partial`/`.crc`). `--input-file-packed` is accepted but deprecated and has no effect.【F:src/maou/infra/console/pre_process.py†L16-L66】【F:src/maou/infra/file_system/path_utils.py†L31-L90】 |
```

### 6. `docs/rust-backend.md:677`

before:
```python
    cache_mode="mmap",
```
after:
```python
    cache_mode="file",
```

## Alternatives considered

- **`--input-file-packed` の記述を「deprecated」と注記して残す.**
  5 では残した (オプションは実在し `--help` に出るため，利用者が
  doc で引けないと困る)．1 と 3 では削った — そこは「この入力ソースは
  何を読むか」の説明であり，無効オプションを併記すると読者が
  bit-pack された入力を渡せると誤解するため．
- **cache_mode の説明を短く「両モードとも全ロード」だけにする.**
  短いが，ではなぜ2つあるのかが分からず，次の読者が同じ調査をする．
  差 (配列の持ち方と 2x スパイク) まで書いて初めて選択の判断ができる．
- **`cache_mode` そのものを廃止する提案にする.**
  step 2.5 の altitude 指摘 (両モードとも全ロードで，本物のメモリ軸は
  streaming 側にある) は妥当だが，`interface/learn.py`，
  `console/utility.py`，`app/learning/dl.py` と CLI に跨る変更であり，
  doc 提案に混ぜる話ではない．backlog へ回す．

## What this enables

- メモリ制約下の利用者が `--input-cache-mode file` を「常駐を抑える
  設定」と誤解しなくなる．実際には抑えられない．
- `docs/rust-backend.md` のサンプルがそのまま実行できるようになる．
- Arrow IPC 移行後に残った `.npy` の記述が 2 コマンド分減る．

## What this constrains

- 2 と 4 は同一文面を2ファイルに複製する．`--input-cache-mode` は
  2コマンドが共有するオプションであり，`docs/commands/` の構成上
  コマンドごとにオプション表を持つ規約なので複製は避けられない．
  片方だけ直す将来のドリフト余地は残る．

## Rollback plan

各ファイルの該当行を before に戻すだけ．相互依存なし．コードに影響なし．
