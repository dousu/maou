/**
 * ワークベンチの行クリックブリッジ．
 *
 * 結果一覧と指し手一覧は gr.Dataframe ではなくサーバー生成の HTML
 * (game_graph_shared.build_row_table_html) で描いているため，行クリックを
 * Gradio に届ける経路が要る．gr.HTML の server_functions / js_on_load で
 * 公開されたブリッジ (window.__maou_row / window.__maou_movesel) を通し，
 * 行番号を送ってから trigger("change") で出力パイプラインを回す．
 *
 * イベントは document 単位の委譲で受けるので，HTML が差し替わっても
 * 張り直しは不要．
 */
(function () {
  "use strict";

  if (window.__maou_wb_rows_init) {
    return;
  }
  window.__maou_wb_rows_init = true;

  /**
   * ブリッジに行番号を送り，成功したら change を発火する．
   *
   * @param {object} bridge window.__maou_row などのブリッジオブジェクト
   * @param {string} fnName server_functions で公開した関数名
   * @param {number} rowIndex 0 始まりの行番号
   */
  function send(bridge, fnName, rowIndex) {
    if (!bridge || !bridge.server || !bridge.server[fnName]) {
      console.warn("[maou] row bridge not ready:", fnName);
      return;
    }
    bridge.server[fnName](String(rowIndex))
      .then(function (ok) {
        if (ok) bridge.trigger("change");
      })
      .catch(function (err) {
        console.error("[maou] " + fnName + " failed:", err);
      });
  }

  document.addEventListener("click", function (ev) {
    var row = ev.target && ev.target.closest
      ? ev.target.closest(".vz-row[data-row]")
      : null;
    if (!row) {
      return;
    }
    var index = parseInt(row.getAttribute("data-row"), 10);
    if (isNaN(index)) {
      return;
    }

    // 押した瞬間に選択を反映する (サーバー往復を待たない)．
    // 確定した選択状態はサーバーが返す HTML で上書きされる．
    var host = row.parentNode;
    if (host) {
      var siblings = host.querySelectorAll(".vz-row");
      for (var i = 0; i < siblings.length; i++) {
        siblings[i].classList.remove("on");
      }
    }
    row.classList.add("on");

    if (row.closest("#vz-result-list")) {
      send(window.__maou_row, "handle_row_select", index);
    } else if (row.closest("#vz-move-list")) {
      send(window.__maou_movesel, "handle_move_select", index);
    }
  });
})();
