# Maou将棋データ可視化ツール - UI/UXリデザイン

## 概要

Maou将棋データ可視化ツールの完全なUI/UXリデザインプロジェクト．
モダン・ミニマリストデザイン（Apple/Vercel風）を採用し，エンジニア向けの開発・デバッグツールとして美しく使いやすいインターフェースを実現．

## デザイン哲学

- **モダン・ミニマリスト**: Apple/Vercel風のクリーンなデザイン
- **暖色系ニュートラル**: 快適性を重視した色使い
- **モダンブルーアクセント**: 技術的印象を与えるアクセント
- **洗練されたタイポグラフィ**: システムフォントスタックの活用
- **控えめなアニメーション**: 必要最小限の動き

## 実装フェーズ

### Phase 1: Visual Foundation（ビジュアル基盤）

**実装内容**:
- モダンなカラーパレット定義
- CSS変数システムの構築
- タイポグラフィシステム
- スペーシングスケール
- トランジション設定

**主要ファイル**:
- `src/maou/infra/visualization/static/theme.css` - テーマ変数
- `src/maou/infra/visualization/static/components.css` - コンポーネントスタイル

**カラーパレット**:
```css
--bg-primary: #fafafa;       /* メイン背景 */
--bg-secondary: #f5f5f5;     /* パネル背景 */
--bg-tertiary: #ffffff;      /* カード/浮いた面 */
--accent-primary: #0070f3;   /* Vercelブルー */
--text-primary: #1a1a1a;     /* メインテキスト */
```

### Phase 2: Layout Redesign（レイアウト再設計）

**実装内容**:
- アコーディオンからタブUIへ移行
- アイコンによる視覚的階層の改善
- セクションヘッダーの統一
- モダンなバッジコンポーネント

**変更点**:
- 📋 概要タブ - レコード詳細表示
- 📊 検索結果タブ - 結果一覧テーブル
- セクションにアイコン追加（🔍 検索，🎯 ナビゲーション，📄 ページネーション）

### Phase 3: Board Enhancements（盤面強化）

**実装内容**:
- 座標ラベルの洗練（バッジスタイル）
- 持ち駒エリアのヘッダーバー追加
- 駒のドロップシャドウ効果
- ホバー時の拡大・輝度効果

**視覚効果**:
```css
.piece {
    font-weight: 700;
    filter: url(#piece-shadow);
    transition: transform 0.2s ease, filter 0.2s ease;
}
.piece:hover {
    transform: scale(1.15);
    filter: url(#piece-shadow) brightness(1.1);
}
```

### Phase 4: Keyboard Shortcuts（キーボードショートカット）

**実装内容**:
- 効率的なキーボードナビゲーション
- ヘルプモーダル（`?`キーで表示）
- elem_id属性でJavaScript連携

**ショートカット一覧**:
| キー | アクション |
|------|-----------|
| `/` | 検索にフォーカス |
| `Esc` | 検索クリア/閉じる |
| `J` / `↓` | 次のレコード |
| `K` / `↑` | 前のレコード |
| `Ctrl+→` | 次のページ |
| `Ctrl+←` | 前のページ |
| `?` | ヘルプ表示 |

**技術実装**:
- グローバルkeydownイベントリスナー
- 入力フィールド内では無効化
- モーダルの動的DOM生成

### Phase 5: Data Visualization（データ可視化）

**実装内容**:
- Plotlyインタラクティブチャート
- 📈 データ分析タブの追加
- データ型別の分析チャート

**チャート種類**:

**HCPE（棋譜データ）**:
- 評価値分布ヒストグラム
- 手数分布ヒストグラム
- 2つのサブプロット並列表示

**Stage1（到達可能マス）**:
- 到達可能マス数の分布

**Stage2（合法手）**:
- 合法手数の分布

**Preprocessing（訓練データ）**:
- 勝率（Result Value）の分布

**実装パターン**:
```python
def generate_analytics(self, records: List[Dict[str, Any]]) -> str:
    """レコード群からPlotlyチャートHTMLを生成"""
    import plotly.graph_objects as go

    # データ抽出
    values = [r.get("field") for r in records]

    # チャート生成
    fig = go.Figure(data=[go.Histogram(x=values)])
    fig.update_layout(template="plotly_white")

    return fig.to_html(include_plotlyjs="cdn")
```

### Phase 6: Loading States（ローディング状態）

**実装内容**:
- スケルトンスクリーン
- プログレスバー
- トースト通知システム
- ローディングオーバーレイ

**スケルトンスクリーン**:
```css
.skeleton {
    background: linear-gradient(90deg,
        var(--bg-secondary) 0%,
        var(--bg-primary) 50%,
        var(--bg-secondary) 100%);
    animation: shimmer 1.5s ease-in-out infinite;
}
```

**トースト通知API**:
```javascript
window.showToast(title, message, type, duration)

// 使用例
showToast('保存成功', 'データが正常に保存されました', 'success');
showToast('エラー', '読み込みに失敗しました', 'error');
```

**通知タイプ**:
- `success` - 成功（✓，グリーン）
- `error` - エラー（✕，レッド）
- `warning` - 警告（⚠，オレンジ）
- `info` - 情報（ℹ，ブルー）

### Phase 7: Responsive Design（レスポンシブデザイン）

**実装内容**:
- タブレット/モバイル対応
- タッチフレンドリーUI
- アクセシビリティ対応
- プリントスタイル

**ブレークポイント**:
```css
--breakpoint-sm: 640px;   /* スマートフォン */
--breakpoint-md: 768px;   /* タブレット */
--breakpoint-lg: 1024px;  /* ラップトップ */
--breakpoint-xl: 1280px;  /* デスクトップ */
```

**タブレット対応（768px - 1199px）**:
- パディング調整
- サイドバー幅削減
- フォントサイズ縮小
- コンパクトなテーブル

**モバイル対応（600px - 767px）**:
- カラムを縦積み
- サイドバー非表示
- フルワイドコンテンツ
- 横スクロールテーブル

**タッチフレンドリー**:
- 最小タップサイズ 44x44px（Apple/Googleガイドライン）
- 大きめの入力フィールド（16px，iOS拡大防止）
- active状態でscale(0.95)効果
- スムーズスクロール

**アクセシビリティ**:
```css
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

## ファイル構成

```
/workspaces/maou/
├── src/maou/
│   ├── domain/visualization/
│   │   └── board_renderer.py          # SVG描画（Phase 3更新）
│   ├── app/visualization/
│   │   └── record_renderer.py         # レンダラー（Phase 5更新）
│   ├── interface/
│   │   └── visualization.py           # インターフェース（Phase 5更新）
│   └── infra/visualization/
│       ├── gradio_server.py           # Gradioサーバー（全フェーズ）
│       └── static/
│           ├── theme.css              # Phase 1, 7
│           └── components.css         # Phase 1-7
└── docs/visualization/
    └── UI_UX_REDESIGN.md             # 本ドキュメント
```

## デザインシステム

### タイポグラフィスケール

```css
--text-xs: 11px;      /* ラベル */
--text-sm: 13px;      /* セカンダリ */
--text-base: 15px;    /* 本文 */
--text-lg: 17px;      /* 強調 */
--text-xl: 20px;      /* サブ見出し */
--text-2xl: 24px;     /* セクション見出し */
--text-3xl: 30px;     /* ページタイトル */
```

### スペーシングスケール

```css
--space-1: 4px;
--space-2: 8px;
--space-3: 12px;
--space-4: 16px;
--space-6: 24px;
--space-8: 32px;
--space-12: 48px;
```

### Border Radius

```css
--radius-sm: 4px;
--radius-md: 6px;
--radius-lg: 8px;
--radius-xl: 12px;
```

### シャドウ

```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
```

## パフォーマンス最適化

### CSS最適化

- CSS変数の活用で一貫性と保守性向上
- トランジションの適切な使用（0.15s - 0.3s）
- アニメーションのGPU加速（transform使用）

### レスポンシブ最適化

- モバイルでのシャドウ削減（パフォーマンス向上）
- 高DPI画面での細いボーダー（0.5px）
- 動き削減設定の尊重

### アクセシビリティ

- WCAG準拠のコントラスト比
- キーボードナビゲーション完備
- タップターゲットサイズ遵守（44x44px）
- アニメーション削減設定対応

## 今後の拡張

### ダークモード（未実装）

プレースホルダー準備済み:
```css
@media (prefers-color-scheme: dark) {
    /* ダークモード対応用 */
}
```

### 追加の可視化

- 時系列チャート（評価値推移）
- 散布図（複数指標の相関）
- ヒートマップ（盤面の傾向）

### 高度なインタラクション

- ドラッグ&ドロップでのレコード比較
- リアルタイム検索（入力中にフィルタリング）
- カスタムダッシュボード

## テスト結果

✅ **全57テスト通過**
✅ **mypy型チェック通過**
✅ **Clean Architecture準拠**

## トラブルシューティング

### CSSが反映されない

**原因**: Gradioのデフォルトスタイルが優先されている

**解決策**: セレクタの詳細度を上げる，または`!important`を使用

```css
/* 詳細度を上げる */
.gradio-container .my-element {
    /* ... */
}

/* または !important */
.my-element {
    color: var(--text-primary) !important;
}
```

### JavaScriptイベントが動作しない

**原因**: Gradio DOMの遅延読み込み

**解決策**: イベント委譲またはDOMContentLoaded待機

```javascript
document.addEventListener('DOMContentLoaded', function() {
    // 初期化コード
});

// または
document.addEventListener('click', function(e) {
    if (e.target.id === 'my-button') {
        // イベント処理
    }
});
```

### チャートが表示されない

**原因1**: Plotly未インストール

**解決策**: `uv add plotly`

**原因2**: CDN読み込み失敗

**解決策**: `include_plotlyjs='cdn'`を確認

```python
fig.to_html(include_plotlyjs='cdn')  # CDN使用
# または
fig.to_html(include_plotlyjs=True)   # 埋め込み (ファイルサイズ増)
```

## 参考資料

- [Gradio Documentation](https://www.gradio.app/)
- [Plotly Python](https://plotly.com/python/)
- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [Material Design - Touch Targets](https://material.io/design/usability/accessibility.html#layout-and-typography)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

## 変更履歴

- 2025-12-29: 全7フェーズ完了（Phase 1-7）
- 初期バージョン: モダン・ミニマリストデザインへの完全リデザイン
