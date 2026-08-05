# `maou floodgate`

## Overview

- Plays games on a **CSA server** over the CSA TCP/IP protocol
  (ver 1.2.1). The default target is
  [floodgate](http://wdoor.c.u-tokyo.ac.jp/shogi/), the public automatic
  tournament server run at wdoor, which is how maou's playing strength gets
  measured against other engines.
- The protocol (login, game conditions, move exchange, game end), the clock
  bookkeeping, and the search all run in Rust (`maou_usi::csa`) with the GIL
  released; the Python layer only passes configuration
  (`maou._rust.maou_usi.run_csa`). The game-playing agent is the **same
  `Agent` the USI engine uses** — CSA is a second transport next to
  [`maou usi`](usi.md), not a second engine. Design:
  [docs/design/usi-engine/index.md](../design/usi-engine/index.md) §4.
- **Identity is the pair (login name, trip)**. floodgate requires no
  registration: any login name works, and the trip (the secret half of the
  password field) is what distinguishes users who picked the same name.
  Ratings accumulate per identity, so reuse the same pair to build a rating
  for one engine version (e.g. `maou_v1`).
- **floodgate returns the client to a logged-out state after every game**, so
  consecutive games are played as one connection per game. The ONNX model is
  loaded and warmed up **once** and shared across all games in the session.
- floodgate arranges games at **:00 and :30 of every hour**, in the
  `floodgate-300-10F` room: 300 seconds per side plus a 10 second Fischer
  increment, drawn at 512 moves. Ratings settle after roughly 15 games.
- Time control is handled by the same `TimeStrategy` as USI play. The CSA
  transport only tracks the clock the server announced (`Total_Time`,
  `Byoyomi`, `Increment`) and converts it into the USI-shaped clock the
  strategy consumes; **the server's `,T<n>` on each move is authoritative**,
  not the client's own measurement.
- Pondering is **not** used on this transport (CSA has no `ponderhit`
  signal). Everything else — resign threshold, nyugyoku declaration
  (`%KACHI`), root-dfpn, leaf-mate, draw values — behaves as in USI play.

## Identity and the password field

floodgate's CSA-mode password field carries the room name:

```
LOGIN <login-name> <game-name>,<trip>
```

`--login-name` and `--password` (the trip) map onto that; `--game-name`
supplies the room and defaults to `floodgate-300-10F`.

- **Omitting `--password` generates a random trip and prints it on stdout.**
  Record it — passing that same value later resumes the same identity. It is
  written to stdout rather than the stderr protocol log precisely because it
  is the one output that cannot be recovered once lost.
- The server identifies a client as `<login-name>+<md5 of the trip>`; that
  string is what appears in the published game record's `'rating:` line, so
  it can be used to confirm that two sessions played under one identity.
- Passing `--password` reuses an existing identity, which is the normal way
  to keep accumulating rating for one engine version.
- For CSA servers that do not use the room-name convention, pass
  `--game-name ""` and the trip is sent as the whole password.

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--login-name TEXT` | yes | Login name on the server. Ratings are aggregated per name; no registration needed. Pick something unique (`maou_test`, `maou_v1`). |
| `--password TEXT` | | Trip distinguishing users with the same login name. Generated at random and printed when omitted. |
| `--game-name TEXT` | default `floodgate-300-10F` | Game room embedded in the CSA password field. Empty string for plain CSA servers. |
| `--host TEXT` | default `wdoor.c.u-tokyo.ac.jp` | CSA server hostname. |
| `--port INT` | default `4081` | CSA server port. |
| `--games INT` | default `1` | Number of consecutive games (0 = unlimited). One connection per game. |
| `--model-path PATH` | | ONNX model file path. When omitted a mock evaluator is used — connectivity checks only, it does not play meaningfully. |
| `--threads INT` | default `1` | Number of search threads. |
| `--batch-size INT` | default `8` | Evaluation batch size (around 64 on GPU, 8 on CPU). |
| `--node-capacity INT` | | Node pool capacity (default 2^20 nodes). |
| `--network-delay-ms INT` | default `1000` | Communication margin. The server measures the consumed time, so the round trip and the driver's own lag are kept outside the search budget. |
| `--min-think-ms INT` | default `100` | Minimum thinking time. |
| `--time-curve/--no-time-curve` | **default on** | Weight the per-move budget towards the **conversion phase**. The multiplier scales **only the discretionary share** (`remaining / horizon`), so on `floodgate-300-10F` the 10-second Fischer increment stays intact while the 300-second bank moves to where it matters. Enabled after a self-play A/B scored 65% over 20 games (+108 Elo); the interval still includes zero, so treat it as provisional — see [verification.md §4.4.1](../design/usi-engine/verification.md). |
| `--time-curve-peak-ply INT` | default `100` | Ply where the curve peaks. |
| `--time-curve-half-width-ply INT` | default `55` | Plies from the peak down to the floor weight. |
| `--time-curve-peak-permille INT` | default `2500` | Multiplier at the peak, per mille. |
| `--time-curve-opening-floor-permille INT` | default `300` | Multiplier before the peak, per mille. |
| `--time-curve-endgame-floor-permille INT` | default `1200` | Multiplier after the peak, per mille (above the flat 1000 on purpose). |
| `--keep-alive-sec INT` | default `60` | Keep-alive interval while waiting. The protocol forbids sending one more often than every 30 seconds, so smaller values are raised to that floor. |
| `--connect-timeout-sec INT` | default `30` | Timeout for connecting and for the login response. |
| `--game-wait-sec INT` | default `2400` | How long to wait for a game to be arranged (0 = forever). |
| `--reconnect-wait-sec INT` | default `5` | Delay between games before reconnecting. |
| `--draw-value-black INT` | default `500` | Draw value for Black in permille. |
| `--draw-value-white INT` | default `500` | Draw value for White in permille. |
| `--resign-value INT` | default `0` | Resign win-rate threshold in permille (0 = never resign). |
| `--resign-consecutive INT` | default `3` | Consecutive below-threshold moves required to resign. |
| `--opening-script TEXT` | | Forced opening moves in USI notation (space separated). |
| `--root-dfpn / --no-root-dfpn` | default on | Run dfpn mate search on the root position in parallel. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root dfpn mate search. |
| `--leaf-mate / --no-leaf-mate` | default on | Enable short mate search at MCTS leaves. |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate threads. |
| `--defensive-mate/--no-defensive-mate` | **default on** | Prove whether the side to move is *being* mated (root + leaves that are in check), and drop root moves that let the opponent force mate. Without it the engine only asks whether it can mate, so a forced mate against it stays invisible until the last ply. |
| `--defensive-mate-threads INT` | default `1` | Parallelism of the defensive filter that screens root moves. Each root move is judged independently, so raising this uses spare CPU to reach a larger budget within the same wall clock (memory scales with it, ~88MB/thread). |
| `--cuda / --no-cuda` | default off | Enable CUDA Execution Provider (needs an `onnx-cuda` wheel). |
| `--tensorrt / --no-tensorrt` | default off | Enable TensorRT Execution Provider (needs an `onnx-tensorrt` wheel). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |
| `--quiet / --no-quiet` | default off | Suppress the per-line protocol log on stderr. |

## Execution flow

1. Connect to `--host:--port` and send `LOGIN <login-name>
   <game-name>,<trip>`. A `LOGIN:incorrect` reply aborts immediately (it
   means the credentials are malformed, and retrying cannot fix it).
2. Wait for `BEGIN Game_Summary` … `END Game_Summary`, sending a blank-line
   keep-alive every `--keep-alive-sec` while idle. The summary carries the
   colour, the time rule, the move limit, and the starting position.
3. Reply `AGREE <game-id>` and wait for `START:<game-id>`.
4. Exchange moves. On its own turn the client searches and sends a CSA move
   (`+7776FU`), `%TORYO` (resign) or `%KACHI` (nyugyoku declaration). Every
   move is applied to the board only when the **server echoes it** with the
   consumed time, which is also what the clock is debited by.
5. The game ends on a `#WIN` / `#LOSE` / `#DRAW` / `#CENSORED` / `#CHUDAN`
   line, preceded by the reason (`#RESIGN`, `#SENNICHITE`, `#TIME_UP`,
   `#ILLEGAL_MOVE`, `#MAX_MOVES`, …).
6. Send `LOGOUT`, close the connection, wait `--reconnect-wait-sec`, and
   start the next game until `--games` games are done.

## Validation and guardrails

- An empty login name is rejected before connecting.
- `LOGIN:incorrect` is fatal; transient failures (connection refused, drop
  mid-game, rejected pairing) are retried on a new connection, and the
  session aborts after **3 consecutive failures** so a misconfiguration
  cannot spin forever.
- Keep-alive is clamped to the protocol's 30 second floor, because a server
  may treat more frequent traffic as a forfeit.
- Moves received from the server are resolved against generated legal moves,
  so a desynchronized board is detected instead of silently diverging.
- The password is never written to the protocol log.

## Outputs and usage

The per-game outcome is printed on stdout after the session, and the
protocol exchange goes to stderr (suppress with `--quiet`).

```bash
# 2 consecutive rated games on floodgate, generating a new identity
uv run maou floodgate \
  --login-name maou_test \
  --games 2 \
  --model-path model.onnx \
  --batch-size 8

# resume the same identity later (rating accumulates under it)
uv run maou floodgate \
  --login-name maou_test \
  --password 0123456789abcdef \
  --games 10 \
  --model-path model.onnx

# GPU (Colab L4)
uv run maou floodgate \
  --login-name maou_v1 --password <trip> --games 0 \
  --model-path model.onnx --threads 1 --batch-size 64 \
  --tensorrt --cuda --trt-cache-dir .trt-cache
```

Game records are published by the server at
<http://wdoor.c.u-tokyo.ac.jp/shogi/> and can be fetched later with
[`maou utility fetch-floodgate`](utility_fetch_floodgate.md).

## Implementation references

- `rust/maou_usi/src/csa/protocol.rs` — CSA line ⇔ typed messages (pure).
- `rust/maou_usi/src/csa/client.rs` — TCP session, clock tracking, game loop.
- `rust/maou_usi/tests/csa_client.rs` — E2E test against a mock CSA server
  (two consecutive games with reconnection).
- `rust/maou_rust/src/maou_usi.rs` — `run_csa` PyO3 binding.
- `src/maou/app/usi/floodgate.py` — use case and trip generation.
- `src/maou/interface/floodgate.py` — interface layer.
- `src/maou/infra/console/floodgate.py` — CLI.
