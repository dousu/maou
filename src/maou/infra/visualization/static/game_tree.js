/**
 * Game Tree Viewer - Cytoscape.js Frontend
 *
 * Cytoscape.jsを使ったインタラクティブなゲームツリー表示
 */

/* global cytoscape */

(function () {
  "use strict";

  let cy = null;
  let tapTimer = null;

  /**
   * 先手視点の勝率(0.0-1.0)からノードの色を計算する
   * 先手有利(>0.55): 青系, 互角(0.45-0.55): グレー, 後手有利(<0.45): 赤系
   */
  function winRateToColor(resultValue) {
    if (resultValue > 0.55) {
      // 先手有利: 青
      const t = Math.min((resultValue - 0.55) / 0.35, 1.0);
      const r = Math.round(144 * (1 - t) + 25 * t);
      const g = Math.round(202 * (1 - t) + 118 * t);
      const b = Math.round(249 * (1 - t) + 210 * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else if (resultValue < 0.45) {
      // 後手有利: 赤 (Red 200 → Red 700)
      // g == b は意図的: 純粋な赤系統を維持するため
      const t = Math.min((0.45 - resultValue) / 0.35, 1.0);
      const r = Math.round(239 * (1 - t) + 211 * t);
      const g = Math.round(154 * (1 - t) + 47 * t);
      const b = Math.round(154 * (1 - t) + 47 * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
    // 互角: グレー
    return "#9E9E9E";
  }

  /**
   * 確率(0.0-1.0)からノードサイズを計算する
   */
  function probabilityToSize(probability) {
    const minSize = 25;
    const maxSize = 60;
    return minSize + (maxSize - minSize) * Math.sqrt(probability);
  }

  /**
   * 確率(0.0-1.0)からエッジの太さを計算する
   */
  function probabilityToWidth(probability) {
    const minWidth = 1;
    const maxWidth = 6;
    return minWidth + (maxWidth - minWidth) * probability;
  }

  /**
   * Gradio slider の現在値を DOM から読み取る
   *
   * elem_id は game_tree_shared.py の ELEM_ID_DEPTH_SLIDER /
   * ELEM_ID_MIN_PROB_SLIDER と同期している．
   */
  function readSlider(elemId) {
    const el = document.getElementById(elemId);
    if (!el) return null;
    const numInput = el.querySelector('input[type="number"]');
    if (numInput) return parseFloat(numInput.value);
    const rangeInput = el.querySelector('input[type="range"]');
    if (rangeInput) return parseFloat(rangeInput.value);
    return null;
  }

  /**
   * ノード選択をサーバーに通知し，Gradio コールバックを発火する
   *
   * gr.HTML の server_functions でPython関数を直接呼び出し，
   * 完了後に trigger("change") で .change() コールバックを発火して
   * Gradio の出力パイプラインで UI を更新する．
   */
  function notifyNodeSelected(nodeId) {
    const bridge = window.__maou_select;
    if (!bridge || !bridge.server) {
      console.warn("[maou] select bridge not ready");
      return;
    }
    bridge.server
      .handle_select(String(nodeId))
      .then(function (ok) {
        if (ok) bridge.trigger("change");
      })
      .catch(function (err) {
        console.error("[maou] handle_select failed:", err);
      });
  }

  /**
   * ノード展開をサーバーに通知し，Gradio コールバックを発火する
   *
   * depth / prob はスライダーの DOM から直接読み取る．
   * elem_id は game_tree_shared.py と同期．
   */
  function notifyNodeExpanded(nodeId) {
    const bridge = window.__maou_expand;
    if (!bridge || !bridge.server) {
      console.warn("[maou] expand bridge not ready");
      return;
    }
    // フォールバック値 (3, 0.01) は Python 側のスライダーデフォルト値
    // (game_tree_server.py: depth_slider value=3,
    //  min_prob_slider value=0.01) と同期すること．
    const depth = readSlider("gt-depth-slider") ?? 3;
    const prob = readSlider("gt-min-prob-slider") ?? 0.01;
    bridge.server
      .handle_expand(String(nodeId), depth, prob)
      .then(function (ok) {
        if (ok) bridge.trigger("change");
      })
      .catch(function (err) {
        console.error("[maou] handle_expand failed:", err);
      });
  }

  /**
   * Cytoscape.jsインスタンスを初期化・更新する
   */
  function renderTree(elementsJson, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Register dagre layout plugin if not yet registered
    if (typeof cytoscape !== "undefined" && typeof cytoscapeDagre !== "undefined") {
      try { cytoscape.use(cytoscapeDagre); } catch (_) { /* already registered */ }
    }

    let elements;
    try {
      elements = typeof elementsJson === "string"
        ? JSON.parse(elementsJson)
        : elementsJson;
    } catch (e) {
      console.error("Failed to parse elements JSON:", e);
      return;
    }

    if (!elements || !elements.nodes || elements.nodes.length === 0) {
      container.innerHTML = '<p style="text-align:center;color:#718096;padding:40px;">データがありません</p>';
      return;
    }

    // Destroy previous instance and cancel pending tap timer
    if (tapTimer) {
      clearTimeout(tapTimer);
      tapTimer = null;
    }
    if (cy) {
      cy.destroy();
      cy = null;
    }

    cy = cytoscape({
      container: container,
      elements: [
        ...elements.nodes,
        ...elements.edges,
      ],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            "font-family": "Noto Sans JP, sans-serif",
            "font-weight": "bold",
            color: "#fff",
            "text-outline-width": 1.5,
            "text-outline-color": function (ele) {
              return winRateToColor(ele.data("sente_result_value"));
            },
            "text-outline-opacity": 0.8,
            "background-color": function (ele) {
              return winRateToColor(ele.data("sente_result_value"));
            },
            width: function (ele) {
              return probabilityToSize(ele.data("probability"));
            },
            height: function (ele) {
              return probabilityToSize(ele.data("probability"));
            },
            "border-width": 2,
            "border-color": "#fff",
            "border-opacity": 0.8,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#0070f3",
            "border-opacity": 1,
            "overlay-padding": 4,
            "overlay-color": "#0070f3",
            "overlay-opacity": 0.15,
          },
        },
        {
          selector: "node[?is_depth_cutoff]",
          style: {
            "border-style": "dashed",
            "border-color": "#ff9800",
          },
        },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#b0b0b0",
            "line-color": "#b0b0b0",
            width: function (ele) {
              return probabilityToWidth(ele.data("probability"));
            },
            opacity: function (ele) {
              return 0.3 + 0.7 * ele.data("probability");
            },
            label: "data(label)",
            "font-size": "8px",
            "text-rotation": "autorotate",
            "text-margin-y": -8,
            color: "#718096",
          },
        },
      ],
      layout: {
        name: "dagre",
        rankDir: "TB",
        nodeSep: 40,
        rankSep: 60,
        edgeSep: 10,
        animate: false,
      },
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.3,
    });

    // Node click -> server_functions で詳細パネルを更新
    // dbltap 時に tap が2回余分に発火するのを防ぐため，タイマーで遅延させる
    cy.on("tap", "node", function (evt) {
      const nodeId = evt.target.id();
      if (tapTimer) clearTimeout(tapTimer);
      tapTimer = setTimeout(function () {
        tapTimer = null;
        notifyNodeSelected(nodeId);
      }, 250);
    });

    // Double click -> expand subtree
    // 現在のルートと同じノードの場合は再描画をスキップする
    cy.on("dbltap", "node", function (evt) {
      // tap タイマーをキャンセルして冗長な select を防止
      if (tapTimer) {
        clearTimeout(tapTimer);
        tapTimer = null;
      }
      var nodeId = evt.target.id();
      if (nodeId === readCurrentRoot()) {
        notifyNodeSelected(nodeId);
      } else {
        notifyNodeExpanded(nodeId);
      }
    });

    // Fit to view
    cy.fit(undefined, 30);
  }

  /**
   * 現在のルートハッシュを DOM から読み取る
   */
  function readCurrentRoot() {
    var el = document.getElementById("current-root");
    if (!el) return "";
    var input = el.querySelector("input") || el.querySelector("textarea");
    return input ? input.value : "";
  }

  /**
   * パンくずリスト要素のクリックハンドラ(イベント委譲)
   *
   * クリックしたノードが現在のルートと異なる場合のみツリーを再描画し，
   * 同じ場合は詳細パネルの更新(select)のみ行う．
   */
  document.addEventListener("click", function (e) {
    const item = e.target.closest(".breadcrumb-item[data-hash]");
    if (!item) return;
    const hash = item.getAttribute("data-hash");
    if (!hash) return;
    if (hash === readCurrentRoot()) {
      notifyNodeSelected(hash);
    } else {
      notifyNodeExpanded(hash);
    }
  });

  /**
   * ツリー画像をPNGとしてダウンロードする
   */
  function exportTreePNG() {
    if (!cy) return;
    const png = cy.png({ scale: 2, bg: "#ffffff", full: true });
    const link = document.createElement("a");
    link.href = png;
    link.download = "game_tree.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Expose to global scope for Gradio integration
  window.renderGameTree = renderTree;
  window.exportTreePNG = exportTreePNG;
})();
