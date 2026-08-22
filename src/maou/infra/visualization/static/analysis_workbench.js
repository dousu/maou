/* 棋譜解析 GUI (analyze-gui) ワークベンチのクライアント側配線．

   ワークベンチは gr.HTML 1 枚として innerHTML で丸ごと差し替わるため，
   リスナーは永続する document に 1 回だけ付ける (委譲)．UI 上の操作は
   すべて data-action 文字列に符号化し，レーンごとのブリッジ
   (server_functions + trigger("change")) 経由で Python に渡す．

   レーン分割は docs/design/game-analysis/gui.md §11 に従う:
   engine / engineAll は concurrency_id="engine" で直列化され，
   cancel は実行中でも即応するよう別レーンに置く． */

(function () {
  if (window.__maouWorkbenchWired) return;
  window.__maouWorkbenchWired = true;

  var WORKBENCH_ID = 'maou-workbench';

  function lanes() {
    return window.__maou_wb || {};
  }

  function laneFor(action) {
    if (action === 'analyze' || action === 'reanalyze') return 'engine';
    if (action === 'analyzeall') return 'engineAll';
    if (action === 'cancel') return 'cancel';
    if (action === 'save') return 'download';
    return 'nav';
  }

  function send(action) {
    if (!action) return;
    var name = laneFor(action);
    var lane = lanes()[name];
    if (!lane || !lane.server) {
      console.warn('[maou] workbench bridge not ready:', name);
      return;
    }
    lane.server
      .handle_action(action)
      .then(function (ok) {
        if (ok) lane.trigger('change');
      })
      .catch(function (err) {
        console.error('[maou] workbench action failed:', action, err);
      });
  }

  /* ── クリック ───────────────────────────────────────── */

  // Gradio の読み込みアコーディオンを開いて視野に入れる
  function openFilePanel() {
    var panel = document.getElementById('maou-file-panel');
    if (!panel) return;
    var toggle = panel.querySelector('button, summary, .label-wrap');
    var open = panel.querySelector('input[type="file"]');
    if (toggle && !open) toggle.click();
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function copyText(text, el) {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      var original = el.textContent;
      el.textContent = 'コピーしました';
      setTimeout(function () {
        el.textContent = original;
      }, 1200);
    });
  }

  function onClick(e) {
    var root = document.getElementById(WORKBENCH_ID);
    if (!root || !root.contains(e.target)) return;

    var opener = e.target.closest('[data-open-file]');
    if (opener) {
      e.preventDefault();
      openFilePanel();
      return;
    }

    var copier = e.target.closest('[data-copy]');
    if (copier) {
      e.preventDefault();
      copyText(copier.getAttribute('data-copy'), copier);
      return;
    }

    // 盤面 SVG のクリック標的 (SVGBoardRenderer が敷く透明 rect)
    var square = e.target.closest('[data-click]');
    if (square) {
      e.preventDefault();
      send('board:' + square.getAttribute('data-click'));
      return;
    }

    var track = e.target.closest('[data-scrub]');
    if (track) {
      var total = parseInt(track.getAttribute('data-scrub'), 10);
      if (total > 0) {
        var rect = track.getBoundingClientRect();
        var ratio = (e.clientX - rect.left) / rect.width;
        ratio = Math.min(1, Math.max(0, ratio));
        send('goto:' + Math.round(ratio * total));
      }
      return;
    }

    var target = e.target.closest('[data-action]');
    if (!target || !root.contains(target)) return;
    e.preventDefault();
    var action = target.getAttribute('data-action');
    send(action);
    if (action === 'save') openFilePanel();
  }

  /* ── 数値入力 (解析予算) ────────────────────────────── */

  function onChange(e) {
    var input = e.target.closest('[data-action-input]');
    if (!input) return;
    send(input.getAttribute('data-action-input') + ':' + input.value);
  }

  /* ── キーボードショートカット ──────────────────────── */

  var KEY_ACTIONS = {
    ArrowLeft: 'nav:prev',
    ArrowRight: 'nav:next',
    Home: 'nav:first',
    End: 'nav:last',
    b: 'nav:mainline',
    B: 'nav:mainline',
    l: 'toggle:legal',
    L: 'toggle:legal',
    Escape: 'clear',
  };

  function isTypingTarget(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  }

  function onKeyDown(e) {
    if (!document.getElementById(WORKBENCH_ID)) return;
    if (isTypingTarget(e.target)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.shiftKey && e.key === 'ArrowLeft') {
      e.preventDefault();
      send('blunder:prev');
      return;
    }
    if (e.shiftKey && e.key === 'ArrowRight') {
      e.preventDefault();
      send('blunder:next');
      return;
    }
    if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      send('analyze');
      return;
    }
    if (e.key >= '1' && e.key <= '5') {
      e.preventDefault();
      send('cand:' + (parseInt(e.key, 10) - 1));
      return;
    }
    var action = KEY_ACTIONS[e.key];
    if (action) {
      e.preventDefault();
      send(action);
    }
  }

  /* ── 再描画後の追従 ─────────────────────────────────── */

  // ワークベンチは丸ごと差し替わるので，棋譜リストのスクロール位置は
  // 現在手の行を見える位置に寄せて復元する．
  function followCurrentRow() {
    var row = document.querySelector('#' + WORKBENCH_ID + ' .mw-kifu-row.is-current');
    if (!row) return;
    var list = row.closest('.mw-kifu-list');
    if (!list) return;
    var top = row.offsetTop - list.clientHeight / 2 + row.clientHeight / 2;
    list.scrollTop = Math.max(0, top);
  }

  var observer = new MutationObserver(function () {
    var root = document.getElementById(WORKBENCH_ID);
    if (!root) return;
    var stamp = root.getAttribute('data-render');
    if (root._maouLastRender === stamp) return;
    root._maouLastRender = stamp;
    followCurrentRow();
  });

  function start() {
    document.addEventListener('click', onClick);
    document.addEventListener('change', onChange);
    document.addEventListener('keydown', onKeyDown);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-render'],
    });
    followCurrentRow();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
