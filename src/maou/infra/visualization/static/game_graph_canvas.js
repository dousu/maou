/**
 * Game Graph Viewer - Canvas 2D Frontend
 *
 * Canvas 2D による仮想描画ベースのゲームグラフ表示．
 * 旧 Cytoscape.js + dagre を置換し，以下を実現する:
 * - エッジ方向の正確な描画(source → target)
 * - 仮想描画による大規模グラフの高速表示
 * - サーバー側事前計算済み座標の利用
 */

(function () {
  "use strict";

  // ========================================
  // 定数
  // ========================================

  const MIN_ZOOM = 0.01;
  const MAX_ZOOM = 5.0;
  const GRID_CELL_SIZE = 100;
  const VIEWPORT_MARGIN = 200;
  const DEBOUNCE_MS = 200;

  // LOD 閾値
  const LOD_DOT_ZOOM = 0.3;
  const LOD_LABEL_ZOOM = 0.8;

  // ノード描画
  const NODE_MIN_SIZE = 25;
  const NODE_MAX_SIZE = 60;
  const NODE_DOT_SIZE = 3;

  // ========================================
  // 色計算
  // ========================================

  /**
   * 先手視点の勝率(0.0-1.0)からノードの色を計算する
   */
  function winRateToColor(resultValue) {
    let t, r, g, b;
    if (resultValue > 0.55) {
      t = Math.min((resultValue - 0.55) / 0.35, 1.0);
      r = Math.round(144 * (1 - t) + 25 * t);
      g = Math.round(202 * (1 - t) + 118 * t);
      b = Math.round(249 * (1 - t) + 210 * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    } else if (resultValue < 0.45) {
      t = Math.min((0.45 - resultValue) / 0.35, 1.0);
      r = Math.round(239 * (1 - t) + 211 * t);
      g = Math.round(154 * (1 - t) + 47 * t);
      b = Math.round(154 * (1 - t) + 47 * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    }
    return "#9E9E9E";
  }

  /**
   * 確率(0.0-1.0)からノード半径を計算する
   */
  function probabilityToRadius(probability) {
    return (NODE_MIN_SIZE + (NODE_MAX_SIZE - NODE_MIN_SIZE) * Math.sqrt(probability)) / 2;
  }

  // ========================================
  // SpatialGrid - グリッドベース空間インデックス
  // ========================================

  function SpatialGrid(cellSize) {
    this.cellSize = cellSize || GRID_CELL_SIZE;
    this.cells = {};
  }

  SpatialGrid.prototype.clear = function () {
    this.cells = {};
  };

  SpatialGrid.prototype._key = function (x, y) {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    return cx + "," + cy;
  };

  SpatialGrid.prototype.insert = function (id, x, y) {
    const key = this._key(x, y);
    if (!this.cells[key]) this.cells[key] = [];
    this.cells[key].push(id);
  };

  SpatialGrid.prototype.query = function (x, y, radius) {
    const results = [];
    const r = radius || 30;
    // 近傍のセルを探索
    const minCx = Math.floor((x - r) / this.cellSize);
    const maxCx = Math.floor((x + r) / this.cellSize);
    const minCy = Math.floor((y - r) / this.cellSize);
    const maxCy = Math.floor((y + r) / this.cellSize);
    for (let cx = minCx; cx <= maxCx; cx++) {
      for (let cy = minCy; cy <= maxCy; cy++) {
        const key = cx + "," + cy;
        const cell = this.cells[key];
        if (cell) {
          for (let i = 0; i < cell.length; i++) {
            results.push(cell[i]);
          }
        }
      }
    }
    return results;
  };

  // ========================================
  // GameGraphRenderer
  // ========================================

  function GameGraphRenderer(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    // State
    this.nodes = new Map(); // id -> {x, y, label, color, radius, ...}
    this.edges = new Map(); // "sourceId:targetId" -> {sx, sy, tx, ty, label, prob, sourceId, targetId}
    this.offsetX = 0;
    this.offsetY = 0;
    this.zoom = 1.0;
    this.selectedNode = null;

    // Spatial index
    this.grid = new SpatialGrid(GRID_CELL_SIZE);

    // Interaction state
    this.dragging = false;
    this._dragMoved = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.tapTimer = null;
    this.renderRequested = false;

    // Viewport query callback
    this.onViewportChange = null;
    this._viewportDebounce = null;

    // Bind event handlers
    this._bindEvents();

    // HiDPI support
    this._setupHiDPI();
  }

  GameGraphRenderer.prototype._setupHiDPI = function () {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this._displayWidth = rect.width;
    this._displayHeight = rect.height;
  };

  GameGraphRenderer.prototype._bindEvents = function () {
    const self = this;

    // Store handler references for cleanup in destroy()
    this._handlers = {
      mousedown: function (e) { self._onMouseDown(e); },
      mousemove: function (e) { self._onMouseMove(e); },
      mouseup: function (e) { self._onMouseUp(e); },
      wheel: function (e) { self._onWheel(e); },
      dblclick: function (e) { self._onDblClick(e); },
      touchstart: function (e) { self._onTouchStart(e); },
      touchmove: function (e) { self._onTouchMove(e); },
      touchend: function (e) { self._onTouchEnd(e); }
    };

    this.canvas.addEventListener("mousedown", this._handlers.mousedown);
    this.canvas.addEventListener("mousemove", this._handlers.mousemove);
    this.canvas.addEventListener("mouseup", this._handlers.mouseup);
    this.canvas.addEventListener("wheel", this._handlers.wheel, { passive: false });
    this.canvas.addEventListener("dblclick", this._handlers.dblclick);

    // Touch support
    this.canvas.addEventListener("touchstart", this._handlers.touchstart, { passive: false });
    this.canvas.addEventListener("touchmove", this._handlers.touchmove, { passive: false });
    this.canvas.addEventListener("touchend", this._handlers.touchend);

    // Resize observer
    this._resizeObserver = new ResizeObserver(function () {
      self._setupHiDPI();
      self.requestRender();
    });
    this._resizeObserver.observe(this.canvas);
  };

  /**
   * リソースを解放する．
   *
   * canvas が動的に再生成される場合(Gradio の re-render 等)に，
   * 旧 canvas のイベントリスナーと ResizeObserver をクリーンアップする．
   */
  GameGraphRenderer.prototype.destroy = function () {
    if (this._handlers) {
      this.canvas.removeEventListener("mousedown", this._handlers.mousedown);
      this.canvas.removeEventListener("mousemove", this._handlers.mousemove);
      this.canvas.removeEventListener("mouseup", this._handlers.mouseup);
      this.canvas.removeEventListener("wheel", this._handlers.wheel);
      this.canvas.removeEventListener("dblclick", this._handlers.dblclick);
      this.canvas.removeEventListener("touchstart", this._handlers.touchstart);
      this.canvas.removeEventListener("touchmove", this._handlers.touchmove);
      this.canvas.removeEventListener("touchend", this._handlers.touchend);
      this._handlers = null;
    }
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
  };


  // ========================================
  // Data management
  // ========================================

  GameGraphRenderer.prototype.setData = function (canvasData) {
    if (!canvasData) return;

    const nodes = canvasData.nodes || [];
    const edges = canvasData.edges || [];

    // Merge nodes (add new, update existing)
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      this.nodes.set(n.id, {
        x: n.x,
        y: n.y,
        label: n.label,
        color: winRateToColor(n.sente_result_value),
        radius: probabilityToRadius(n.probability),
        probability: n.probability,
        depth: n.depth,
        numBranches: n.num_branches,
        isDepthCutoff: n.is_depth_cutoff,
        senteResultValue: n.sente_result_value,
      });
    }

    // Merge edges (keep previously loaded edges outside current viewport)
    for (let j = 0; j < edges.length; j++) {
      const e = edges[j];
      const edgeKey = e.source_id + ":" + e.target_id;
      this.edges.set(edgeKey, {
        sourceId: e.source_id,
        targetId: e.target_id,
        sx: e.source_x,
        sy: e.source_y,
        tx: e.target_x,
        ty: e.target_y,
        label: e.label,
        prob: e.probability,
      });
    }

    // Rebuild spatial index
    this.grid.clear();
    this.nodes.forEach(function (nd, id) {
      this.grid.insert(id, nd.x, nd.y);
    }.bind(this));

    this.requestRender();
  };

  GameGraphRenderer.prototype.clearData = function () {
    this.nodes.clear();
    this.edges.clear();
    this.grid.clear();
    this.requestRender();
  };

  GameGraphRenderer.prototype.fitToView = function () {
    if (this.nodes.size === 0) return;

    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    this.nodes.forEach(function (n) {
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    });

    const padding = 50;
    const dataWidth = maxX - minX + padding * 2;
    const dataHeight = maxY - minY + padding * 2;

    const zoomX = this._displayWidth / dataWidth;
    const zoomY = this._displayHeight / dataHeight;
    this.zoom = Math.min(zoomX, zoomY, 2.0);
    this.zoom = Math.max(this.zoom, MIN_ZOOM);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    this.offsetX = this._displayWidth / 2 - centerX * this.zoom;
    this.offsetY = this._displayHeight / 2 - centerY * this.zoom;

    this.requestRender();
  };

  // ========================================
  // Coordinate transforms
  // ========================================

  GameGraphRenderer.prototype.screenToWorld = function (sx, sy) {
    return {
      x: (sx - this.offsetX) / this.zoom,
      y: (sy - this.offsetY) / this.zoom,
    };
  };

  GameGraphRenderer.prototype.worldToScreen = function (wx, wy) {
    return {
      x: wx * this.zoom + this.offsetX,
      y: wy * this.zoom + this.offsetY,
    };
  };

  // ========================================
  // Rendering
  // ========================================

  GameGraphRenderer.prototype.requestRender = function () {
    if (this.renderRequested) return;
    this.renderRequested = true;
    const self = this;
    requestAnimationFrame(function () {
      self.renderRequested = false;
      self._render();
    });
  };

  GameGraphRenderer.prototype._render = function () {
    const ctx = this.ctx;
    const w = this._displayWidth;
    const h = this._displayHeight;

    ctx.clearRect(0, 0, w, h);

    // Visible bounds in world coordinates
    const topLeft = this.screenToWorld(0, 0);
    const bottomRight = this.screenToWorld(w, h);
    const margin = VIEWPORT_MARGIN / this.zoom;
    const visLeft = topLeft.x - margin;
    const visTop = topLeft.y - margin;
    const visRight = bottomRight.x + margin;
    const visBottom = bottomRight.y + margin;

    // Draw edges
    this._renderEdges(ctx, visLeft, visTop, visRight, visBottom);

    // Draw nodes
    this._renderNodes(ctx, visLeft, visTop, visRight, visBottom);
  };

  GameGraphRenderer.prototype._renderEdges = function (ctx, vl, vt, vr, vb) {
    const zoom = this.zoom;
    const showLabels = zoom >= LOD_LABEL_ZOOM;

    const edgesIter = this.edges.values();
    let edgeEntry = edgesIter.next();
    while (!edgeEntry.done) {
      const e = edgeEntry.value;
      edgeEntry = edgesIter.next();

      // Frustum culling: skip if edge bounding box does not intersect viewport.
      // This correctly handles edges that cross the viewport without either
      // endpoint being inside it.
      const eMinX = e.sx < e.tx ? e.sx : e.tx;
      const eMaxX = e.sx > e.tx ? e.sx : e.tx;
      const eMinY = e.sy < e.ty ? e.sy : e.ty;
      const eMaxY = e.sy > e.ty ? e.sy : e.ty;
      if (eMaxX < vl || eMinX > vr || eMaxY < vt || eMinY > vb) continue;

      // Transform to screen
      const s = this.worldToScreen(e.sx, e.sy);
      const t = this.worldToScreen(e.tx, e.ty);

      // Line
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.strokeStyle = "rgba(176,176,176," + (0.3 + 0.7 * e.prob) + ")";
      ctx.lineWidth = (1 + 5 * e.prob) * Math.min(zoom, 1.5);
      ctx.stroke();

      // Arrowhead at target
      const angle = Math.atan2(t.y - s.y, t.x - s.x);
      const targetNode = this.nodes.get(e.targetId);
      const nodeRadius = targetNode ? targetNode.radius * zoom : 15 * zoom;
      const headLen = (8 + 4 * e.prob) * Math.min(zoom, 1.5);
      const ax = t.x - nodeRadius * Math.cos(angle);
      const ay = t.y - nodeRadius * Math.sin(angle);

      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(
        ax - headLen * Math.cos(angle - Math.PI / 6),
        ay - headLen * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        ax - headLen * Math.cos(angle + Math.PI / 6),
        ay - headLen * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = "rgba(176,176,176," + (0.3 + 0.7 * e.prob) + ")";
      ctx.fill();

      // Edge label
      if (showLabels && e.label) {
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        ctx.save();
        ctx.font = "8px 'Noto Sans JP', sans-serif";
        ctx.fillStyle = "#718096";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(e.label, mx, my - 4);
        ctx.restore();
      }
    }
  };

  GameGraphRenderer.prototype._renderNodes = function (ctx, vl, vt, vr, vb) {
    const zoom = this.zoom;
    const isDot = zoom < LOD_DOT_ZOOM;
    const showLabel = zoom >= LOD_LABEL_ZOOM;
    const self = this;

    this.nodes.forEach(function (n, id) {
      // Frustum culling (半径分の余裕を持たせる)
      const worldRadius = n.radius;
      if (n.x + worldRadius < vl || n.x - worldRadius > vr || n.y + worldRadius < vt || n.y - worldRadius > vb) return;

      const screen = self.worldToScreen(n.x, n.y);

      if (isDot) {
        // LOD: dot mode
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, NODE_DOT_SIZE, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
        return;
      }

      const screenRadius = n.radius * Math.min(zoom, 2.0);

      // Node circle
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, screenRadius, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.fill();

      // Border
      if (id === self.selectedNode) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#0070f3";
      } else if (n.isDepthCutoff) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ff9800";
        ctx.setLineDash([4, 2]);
      } else {
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      if (showLabel && n.label) {
        ctx.save();
        const fontSize = Math.max(8, Math.min(12, screenRadius * 0.7));
        ctx.font = "bold " + fontSize + "px 'Noto Sans JP', sans-serif";
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // Text outline for readability
        ctx.strokeStyle = n.color;
        ctx.lineWidth = 2;
        ctx.strokeText(n.label, screen.x, screen.y);
        ctx.fillText(n.label, screen.x, screen.y);
        ctx.restore();
      }
    });
  };

  // ========================================
  // Hit testing
  // ========================================

  GameGraphRenderer.prototype.hitTest = function (screenX, screenY) {
    const world = this.screenToWorld(screenX, screenY);
    const searchRadius = 50 / this.zoom;
    const candidates = this.grid.query(world.x, world.y, searchRadius);

    let bestId = null;
    let bestDist = Infinity;

    for (let i = 0; i < candidates.length; i++) {
      const id = candidates[i];
      const n = this.nodes.get(id);
      if (!n) continue;
      const dx = n.x - world.x;
      const dy = n.y - world.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < n.radius + 5 / this.zoom && dist < bestDist) {
        bestDist = dist;
        bestId = id;
      }
    }
    return bestId;
  };

  // ========================================
  // Event handlers
  // ========================================

  GameGraphRenderer.prototype._onMouseDown = function (e) {
    this.dragging = true;
    this.dragStartX = e.offsetX;
    this.dragStartY = e.offsetY;
    this._dragMoved = false;
  };

  GameGraphRenderer.prototype._onMouseMove = function (e) {
    if (!this.dragging) return;
    const dx = e.offsetX - this.dragStartX;
    const dy = e.offsetY - this.dragStartY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      this._dragMoved = true;
    }
    this.offsetX += dx;
    this.offsetY += dy;
    this.dragStartX = e.offsetX;
    this.dragStartY = e.offsetY;
    this.requestRender();
  };

  GameGraphRenderer.prototype._onMouseUp = function (e) {
    this.dragging = false;
    if (!this._dragMoved) {
      this._handleClick(e.offsetX, e.offsetY);
    } else {
      this._notifyViewportChange();
    }
  };

  GameGraphRenderer.prototype._onWheel = function (e) {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const mouseWorld = this.screenToWorld(e.offsetX, e.offsetY);
    this.zoom *= zoomFactor;
    this.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, this.zoom));
    this.offsetX = e.offsetX - mouseWorld.x * this.zoom;
    this.offsetY = e.offsetY - mouseWorld.y * this.zoom;
    this.requestRender();
    this._notifyViewportChange();
  };

  GameGraphRenderer.prototype._onDblClick = function (e) {
    // Cancel any pending tap timer
    if (this.tapTimer) {
      clearTimeout(this.tapTimer);
      this.tapTimer = null;
    }
    const nodeId = this.hitTest(e.offsetX, e.offsetY);
    if (nodeId) {
      const currentRoot = readCurrentRoot();
      if (nodeId === currentRoot) {
        notifyNodeSelected(nodeId);
      } else {
        notifyNodeExpanded(nodeId);
      }
    }
  };

  GameGraphRenderer.prototype._handleClick = function (sx, sy) {
    const nodeId = this.hitTest(sx, sy);
    if (nodeId) {
      // Use timer to differentiate click from double-click
      const self = this;
      if (this.tapTimer) clearTimeout(this.tapTimer);
      this.tapTimer = setTimeout(function () {
        self.tapTimer = null;
        self.selectedNode = nodeId;
        self.requestRender();
        notifyNodeSelected(nodeId);
      }, 250);
    }
  };

  // Touch events
  GameGraphRenderer.prototype._onTouchStart = function (e) {
    if (e.touches.length === 1) {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = this.canvas.getBoundingClientRect();
      this.dragStartX = touch.clientX - rect.left;
      this.dragStartY = touch.clientY - rect.top;
      this.dragging = true;
      this._dragMoved = false;
    }
  };

  GameGraphRenderer.prototype._onTouchMove = function (e) {
    if (!this.dragging || e.touches.length !== 1) return;
    e.preventDefault();
    const touch = e.touches[0];
    const rect = this.canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    const dx = x - this.dragStartX;
    const dy = y - this.dragStartY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) this._dragMoved = true;
    this.offsetX += dx;
    this.offsetY += dy;
    this.dragStartX = x;
    this.dragStartY = y;
    this.requestRender();
  };

  GameGraphRenderer.prototype._onTouchEnd = function () {
    this.dragging = false;
    if (!this._dragMoved) {
      const x = this.dragStartX;
      const y = this.dragStartY;
      this._handleClick(x, y);
    } else {
      this._notifyViewportChange();
    }
  };

  // ========================================
  // Viewport change notification
  // ========================================

  GameGraphRenderer.prototype._notifyViewportChange = function () {
    if (!this.onViewportChange) return;
    const self = this;
    if (this._viewportDebounce) clearTimeout(this._viewportDebounce);
    this._viewportDebounce = setTimeout(function () {
      self._viewportDebounce = null;
      const topLeft = self.screenToWorld(0, 0);
      const bottomRight = self.screenToWorld(self._displayWidth, self._displayHeight);
      const margin = 500 / self.zoom;
      self.onViewportChange(
        topLeft.x - margin,
        bottomRight.x + margin,
        topLeft.y - margin,
        bottomRight.y + margin
      );
    }, DEBOUNCE_MS);
  };

  // ========================================
  // PNG export
  // ========================================

  GameGraphRenderer.prototype.exportPNG = function () {
    const png = this.canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = png;
    link.download = "game_graph.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ========================================
  // Gradio bridge helpers (same as game_graph.js)
  // ========================================

  function readSlider(elemId) {
    const el = document.getElementById(elemId);
    if (!el) return null;
    const numInput = el.querySelector('input[type="number"]');
    if (numInput) return parseFloat(numInput.value);
    const rangeInput = el.querySelector('input[type="range"]');
    if (rangeInput) return parseFloat(rangeInput.value);
    return null;
  }

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

  function notifyNodeExpanded(nodeId) {
    const bridge = window.__maou_expand;
    if (!bridge || !bridge.server) {
      console.warn("[maou] expand bridge not ready");
      return;
    }
    const depth = readSlider("gt-depth-slider") || 3;
    const prob = readSlider("gt-min-prob-slider") || 0.01;
    bridge.server
      .handle_expand(String(nodeId), depth, prob)
      .then(function (ok) {
        if (ok) bridge.trigger("change");
      })
      .catch(function (err) {
        console.error("[maou] handle_expand failed:", err);
      });
  }

  function readCurrentRoot() {
    const el = document.getElementById("current-root");
    if (!el) return "";
    const input = el.querySelector("input") || el.querySelector("textarea");
    return input ? input.value : "";
  }

  // ========================================
  // Initialization and rendering
  // ========================================

  let renderer = null;

  function renderGraph(canvasDataJson, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let data;
    try {
      data = typeof canvasDataJson === "string"
        ? JSON.parse(canvasDataJson)
        : canvasDataJson;
    } catch (e) {
      console.error("[maou] Failed to parse canvas data:", e);
      return;
    }

    if (!data || !data.nodes || data.nodes.length === 0) {
      container.innerHTML =
        '<p style="text-align:center;color:#718096;padding:40px;">データがありません</p>';
      return;
    }

    // Ensure canvas element exists
    let canvas = container.querySelector("canvas");
    if (!canvas) {
      container.innerHTML = "";
      canvas = document.createElement("canvas");
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.display = "block";
      container.appendChild(canvas);
    }

    // Create renderer if not exists or canvas changed
    if (!renderer || renderer.canvas !== canvas) {
      if (renderer) {
        renderer.destroy();
      }
      renderer = new GameGraphRenderer(canvas);

      // Set up viewport query callback
      renderer.onViewportChange = function (minX, maxX, minY, maxY) {
        const bridge = window.__maou_viewport;
        if (!bridge || !bridge.server) return;
        bridge.server
          .handle_viewport(minX, maxX, minY, maxY)
          .then(function (result) {
            if (result) {
              const viewportData = typeof result === "string" ? JSON.parse(result) : result;
              renderer.setData(viewportData);
            }
          })
          .catch(function (err) {
            console.error("[maou] viewport query failed:", err);
          });
      };
    }

    renderer.setData(data);
    renderer.fitToView();
  }

  function exportGraphPNG() {
    if (renderer) renderer.exportPNG();
  }

  // Breadcrumb click handler (event delegation)
  function handleBreadcrumbClick(e) {
    const item = e.target.closest(".breadcrumb-item[data-hash]");
    if (!item) return;
    const hash = item.getAttribute("data-hash");
    if (!hash) return;
    if (hash === readCurrentRoot()) {
      notifyNodeSelected(hash);
    } else {
      notifyNodeExpanded(hash);
    }
  }
  document.addEventListener("click", handleBreadcrumbClick);

  // Expose to global scope
  window.renderGameGraph = renderGraph;
  window.exportGraphPNG = exportGraphPNG;
  window.destroyGameGraph = function () {
    if (renderer) {
      renderer.destroy();
      renderer = null;
    }
    document.removeEventListener("click", handleBreadcrumbClick);
  };
})();
