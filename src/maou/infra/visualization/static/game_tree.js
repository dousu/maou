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
   * 勝率(0.0-1.0)からノードの色を計算する
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
   * hidden textboxの値を設定する(イベントディスパッチなし)
   */
  function setHiddenTextbox(selector, value) {
    const hiddenInput = document.querySelector(selector);
    if (!hiddenInput) return;
    const proto = hiddenInput instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const nativeSetter = Object.getOwnPropertyDescriptor(
      proto, "value"
    )?.set;
    if (nativeSetter) {
      nativeSetter.call(hiddenInput, value);
    } else {
      hiddenInput.value = value;
    }
    // nativeSetter で値を設定すると Svelte の内部状態が更新される．
    // イベントディスパッチは triggerHiddenInput() に一元化し，
    // ここでは値の設定のみを行う．
  }

  /**
   * hidden textboxにinputイベントをディスパッチしてGradio .input()を発火する
   *
   * Gradio 6ではプログラマティックなボタン.click()がイベントパイプライン
   * (jsプリプロセッサ含む)を正しく発火しない．代わりにtextboxの
   * inputイベントをディスパッチし，.input()ハンドラを発火させる．
   */
  function triggerHiddenInput(selector) {
    const el = document.querySelector(selector);
    if (!el) return;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
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
              return winRateToColor(ele.data("result_value"));
            },
            "text-outline-opacity": 0.8,
            "background-color": function (ele) {
              return winRateToColor(ele.data("result_value"));
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

    // Node click -> update hidden textbox for detail panel
    // dbltap 時に tap が2回余分に発火するのを防ぐため，タイマーで遅延させる
    cy.on("tap", "node", function (evt) {
      const nodeId = evt.target.id();
      if (tapTimer) clearTimeout(tapTimer);
      tapTimer = setTimeout(function () {
        tapTimer = null;
        // グローバル変数に保存(js パラメータから確実に読み取れる)
        window.__maou_selected_node_id = nodeId;
        setHiddenTextbox(
          "#selected-node-id textarea, #selected-node-id input",
          nodeId
        );
        // textbox の input イベントで Gradio .input() コールバックを発火
        triggerHiddenInput(
          "#selected-node-id textarea, #selected-node-id input"
        );
      }, 250);
    });

    // Double click -> expand subtree
    cy.on("dbltap", "node", function (evt) {
      // tap タイマーをキャンセルして冗長な select を防止
      if (tapTimer) {
        clearTimeout(tapTimer);
        tapTimer = null;
      }
      const nodeId = evt.target.id();
      // グローバル変数に保存(js パラメータから確実に読み取れる)
      window.__maou_expand_node_id = nodeId;
      setHiddenTextbox(
        "#expand-node-id textarea, #expand-node-id input",
        nodeId
      );
      // textbox の input イベントで Gradio .input() コールバックを発火
      triggerHiddenInput(
        "#expand-node-id textarea, #expand-node-id input"
      );
    });

    // Fit to view
    cy.fit(undefined, 30);
  }

  /**
   * パンくずリスト要素のクリックハンドラ(イベント委譲)
   */
  document.addEventListener("click", function (e) {
    const item = e.target.closest(".breadcrumb-item[data-hash]");
    if (!item) return;
    const hash = item.getAttribute("data-hash");
    if (!hash) return;
    // グローバル変数に保存(js パラメータから確実に読み取れる)
    window.__maou_expand_node_id = hash;
    // パンくずクリック → ツリー展開(expand は select の出力を包含する)
    setHiddenTextbox(
      "#expand-node-id textarea, #expand-node-id input",
      hash
    );
    // textbox の input イベントで Gradio .input() コールバックを発火
    triggerHiddenInput(
      "#expand-node-id textarea, #expand-node-id input"
    );
  });

  /**
   * ツリー画像をPNGとしてダウンロードする
   */
  function exportTreePNG() {
    if (!cy) return;
    var png = cy.png({ scale: 2, bg: "#ffffff", full: true });
    var link = document.createElement("a");
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
