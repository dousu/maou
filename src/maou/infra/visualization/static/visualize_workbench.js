/* データ可視化 GUI (visualize) ワークベンチのクライアント側配線．

   ワークベンチは gr.HTML 1 枚として innerHTML で丸ごと差し替わるため，
   リスナーは永続する document に 1 回だけ付ける (委譲)．UI 上の操作は
   すべて data-action 文字列に符号化し，レーンごとのブリッジ
   (server_functions + trigger("change")) 経由で Python に渡す．
   analyze-gui の static/analysis_workbench.js と同じ方式．

   レーン分割:
     nav   — 選択・ページ送り・表示切替 (即応が要る軽い操作)
     load  — データソース読み込み・インデックス再構築 (時間がかかる)
   重い操作を nav と分けておかないと，読み込み中に行送りが詰まる． */

(function () {
  if (window.__mauVizWired) return;
  window.__mauVizWired = true;

  var ROOT_ID = 'viz-workbench';

  function lanes() {
    return window.__maou_viz || {};
  }

  function laneFor(action) {
    var verb = String(action).split(':')[0];
    if (verb === 'load' || verb === 'rebuild' || verb === 'type') {
      return 'load';
    }
    return 'nav';
  }

  function send(action) {
    if (!action) return;
    var name = laneFor(action);
    var lane = lanes()[name];
    if (!lane || !lane.server) {
      console.warn('[maou] viz bridge not ready:', name);
      return;
    }
    lane.server
      .handle_action(action)
      .then(function (ok) {
        if (ok) lane.trigger('change');
      })
      .catch(function (err) {
        console.error('[maou] viz action failed:', action, err);
      });
  }

  /* ── クリップボード ─────────────────────────────────── */

  function flash(el, message) {
    var original = el.textContent;
    el.textContent = message;
    setTimeout(function () {
      el.textContent = original;
    }, 1200);
  }

  // navigator.clipboard は secure context (https / localhost) でしか
  // 生えない．LAN 越しの http や --share の中継先では undefined に
  // なるので，隠しテキストエリア + execCommand に落とす
  function legacyCopy(text) {
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(area);
    return ok;
  }

  function copyText(text, el) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(
        function () {
          flash(el, 'コピーしました');
        },
        function () {
          flash(el, legacyCopy(text) ? 'コピーしました' : 'コピーできません');
        }
      );
      return;
    }
    flash(el, legacyCopy(text) ? 'コピーしました' : 'コピーできません');
  }

  /* ── クリック ───────────────────────────────────────── */

  function onClick(e) {
    var root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(e.target)) return;

    var copier = e.target.closest('[data-copy]');
    if (copier) {
      e.preventDefault();
      copyText(copier.getAttribute('data-copy'), copier);
      return;
    }

    var target = e.target.closest('[data-action]');
    if (!target || !root.contains(target)) return;
    e.preventDefault();

    // 行選択はサーバー往復を待たずに見た目を先に反映する．
    // 確定した選択状態はサーバーが返す HTML で上書きされる．
    if (target.classList.contains('vz-row')) {
      var host = target.parentNode;
      if (host) {
        var rows = host.querySelectorAll('.vz-row');
        for (var i = 0; i < rows.length; i++) {
          rows[i].classList.remove('on');
        }
      }
      target.classList.add('on');
    }

    send(target.getAttribute('data-action'));
  }

  /* ── 入力 ───────────────────────────────────────────── */

  // テキストは Enter と blur で確定する (1 文字ごとに往復させない)．
  // range は input の途中経過を拾わず change (離した時) だけ送る．
  function onChange(e) {
    var input = e.target.closest('[data-action-input]');
    if (!input) return;
    send(input.getAttribute('data-action-input') + ':' + input.value);
  }

  function onKeyDown(e) {
    var root = document.getElementById(ROOT_ID);
    if (!root) return;

    var input = e.target.closest && e.target.closest('[data-action-input]');
    if (input) {
      if (e.key === 'Enter') {
        e.preventDefault();
        send(input.getAttribute('data-action-input') + ':' + input.value);
      }
      return;
    }
    if (isTypingTarget(e.target)) return;
    if (e.altKey || e.metaKey) return;

    if (e.ctrlKey) {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        send('page:prev');
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        send('page:next');
      }
      return;
    }

    var action = KEY_ACTIONS[e.key];
    if (action) {
      e.preventDefault();
      send(action);
    }
  }

  var KEY_ACTIONS = {
    k: 'rec:prev',
    K: 'rec:prev',
    ArrowUp: 'rec:prev',
    j: 'rec:next',
    J: 'rec:next',
    ArrowDown: 'rec:next',
  };

  function isTypingTarget(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  }

  /* ── 再描画後の追従 ─────────────────────────────────── */

  // ワークベンチは丸ごと差し替わるので，結果一覧のスクロール位置は
  // 選択行が見える位置に寄せて復元する．
  function followSelectedRow() {
    var row = document.querySelector('#' + ROOT_ID + ' .vz-row.on');
    if (!row) return;
    var list = row.closest('.vz-table-body');
    if (!list) return;
    var offset =
      row.getBoundingClientRect().top - list.getBoundingClientRect().top;
    var top =
      list.scrollTop + offset - list.clientHeight / 2 + row.clientHeight / 2;
    list.scrollTop = Math.max(0, top);
  }

  var observer = new MutationObserver(function () {
    var root = document.getElementById(ROOT_ID);
    if (!root) return;
    var stamp = root.getAttribute('data-render');
    if (root._mauLastRender === stamp) return;
    root._mauLastRender = stamp;
    followSelectedRow();
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
    followSelectedRow();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
