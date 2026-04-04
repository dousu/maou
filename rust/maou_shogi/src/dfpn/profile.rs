//! プロファイリング用マクロと統計構造体．


/// `profile` feature が有効な場合のみ計測し，結果をフィールドに加算する．
#[cfg(feature = "profile")]
macro_rules! profile_timed {
    ($self:expr, $ns_field:ident, $count_field:ident, $body:expr) => {{
        let _t = Instant::now();
        let _r = $body;
        let _elapsed = _t.elapsed().as_nanos() as u64;
        $self.profile_stats.$ns_field += _elapsed;
        $self.profile_stats.$count_field += 1;
        _r
    }};
}

/// プロファイリング無効時は素通し．
#[cfg(not(feature = "profile"))]
macro_rules! profile_timed {
    ($self:expr, $ns_field:ident, $count_field:ident, $body:expr) => {
        $body
    };
}

/// プロファイリング統計情報．
///
/// `profile` feature が有効な場合に収集される．
/// 各フィールドは `mid()` 内の主要操作の累積時間(ナノ秒)と呼び出し回数を保持する．
#[cfg(feature = "profile")]
#[derive(Debug, Clone, Default)]
pub struct ProfileStats {
    /// position_key() の累積時間(ナノ秒)．
    pub position_key_ns: u64,
    /// position_key() の呼び出し回数．
    pub position_key_count: u64,
    /// ループ検出(path.contains)の累積時間(ナノ秒)．
    pub loop_detect_ns: u64,
    /// ループ検出の呼び出し回数．
    pub loop_detect_count: u64,
    /// TT 参照(look_up_pn_dn)の累積時間(ナノ秒)．
    pub tt_lookup_ns: u64,
    /// TT 参照の呼び出し回数．
    pub tt_lookup_count: u64,
    /// TT 格納(store)の累積時間(ナノ秒)．
    pub tt_store_ns: u64,
    /// TT 格納の呼び出し回数．
    pub tt_store_count: u64,
    /// 王手生成(generate_check_moves)の累積時間(ナノ秒)．
    pub movegen_check_ns: u64,
    /// 王手生成の呼び出し回数．
    pub movegen_check_count: u64,
    /// 応手生成(generate_defense_moves)の累積時間(ナノ秒)．
    pub movegen_defense_ns: u64,
    /// 応手生成の呼び出し回数．
    pub movegen_defense_count: u64,
    /// do_move の累積時間(ナノ秒)．
    pub do_move_ns: u64,
    /// do_move の呼び出し回数．
    pub do_move_count: u64,
    /// undo_move の累積時間(ナノ秒)．
    pub undo_move_ns: u64,
    /// undo_move の呼び出し回数．
    pub undo_move_count: u64,
    /// 子ノード初期化フェーズの累積時間(ナノ秒)．
    pub child_init_ns: u64,
    /// 子ノード初期化の呼び出し回数．
    pub child_init_count: u64,
    /// 子ノード初期化内の do_move/undo_move の累積時間(ナノ秒)．
    pub child_init_domove_ns: u64,
    /// 子ノード初期化内の do_move/undo_move の呼び出し回数．
    pub child_init_domove_count: u64,
    /// メインループの pn/dn 収集の累積時間(ナノ秒)．
    pub main_loop_collect_ns: u64,
    /// メインループの pn/dn 収集回数．
    pub main_loop_collect_count: u64,
    /// 深さ制限時の終端処理(`depth_limit_all_checks_refutable` を含む)の累積時間(ナノ秒)．
    pub depth_limit_terminal_ns: u64,
    /// 深さ制限時の終端処理の呼び出し回数．
    pub depth_limit_terminal_count: u64,
    /// NM 昇格のための `depth_limit_all_checks_refutable` の累積時間(ナノ秒)．
    pub nm_promotion_refutable_ns: u64,
    /// NM 昇格のための `depth_limit_all_checks_refutable` の呼び出し回数．
    pub nm_promotion_refutable_count: u64,
    /// 合駒 TT 先読み(`generate_check_moves` + `try_capture_tt_proof`)の累積時間(ナノ秒)．
    pub capture_tt_lookahead_ns: u64,
    /// 合駒 TT 先読みの呼び出し回数．
    pub capture_tt_lookahead_count: u64,
    /// `cross_deduce_children` の累積時間(ナノ秒)．
    pub cross_deduce_ns: u64,
    /// `cross_deduce_children` の呼び出し回数．
    pub cross_deduce_count: u64,
    /// `try_prefilter_block` の累積時間(ナノ秒)．
    pub prefilter_ns: u64,
    /// `try_prefilter_block` の呼び出し回数．
    pub prefilter_count: u64,
    /// MID 全体のウォール時間(ナノ秒)．mid_fallback 内のみ計測．
    pub mid_total_ns: u64,
    /// MID トップレベル呼び出し回数．
    pub mid_total_count: u64,
    /// PNS フェーズのウォール時間(ナノ秒)．
    pub pns_total_ns: u64,
    // === child_init 内訳 ===
    /// child_init 内: TT ミス時のインライン判定(movegen + heuristic + store)の累積時間(ナノ秒)．
    pub ci_inline_ns: u64,
    pub ci_inline_count: u64,
    /// child_init 内: 解決チェック(look_up + 証明/反証伝播)の累積時間(ナノ秒)．
    pub ci_resolve_ns: u64,
    pub ci_resolve_count: u64,

    /// TT エントリ溢れ(置換)の発生回数．
    pub tt_overflow_count: u64,
    /// ProvenTT での overflow 回数．
    pub tt_proven_overflow_count: u64,
    /// WorkingTT での overflow 回数．
    pub tt_working_overflow_count: u64,
    /// TT エントリ溢れで置換対象が見つからなかった回数．
    pub tt_overflow_no_victim_count: u64,
    /// TT エントリ数の最大値(1局面あたり)．
    pub tt_max_entries_per_position: usize,
}

#[cfg(feature = "profile")]
impl std::fmt::Display for ProfileStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // child_init_domove は child_init の内訳だが，early return で child_init が
        // 記録されないケースがあるため，未計上分を補正する．
        let child_init_uncaptured = self.child_init_domove_ns
            .saturating_sub(self.child_init_ns);
        let total_ns = self.position_key_ns
            + self.loop_detect_ns
            + self.tt_lookup_ns
            + self.tt_store_ns
            + self.movegen_check_ns
            + self.movegen_defense_ns
            + self.do_move_ns
            + self.undo_move_ns
            + self.child_init_ns
            + child_init_uncaptured
            + self.main_loop_collect_ns
            + self.depth_limit_terminal_ns
            + self.nm_promotion_refutable_ns
            + self.capture_tt_lookahead_ns
            + self.cross_deduce_ns
            + self.prefilter_ns;
        let total_us = total_ns as f64 / 1000.0;

        writeln!(f, "=== DFPN Profile Stats ===")?;
        writeln!(f, "{:<25} {:>12} {:>10} {:>10} {:>6}",
            "Operation", "Total(µs)", "Count", "Avg(ns)", "%")?;
        writeln!(f, "{}", "-".repeat(65))?;

        let items: Vec<(&str, u64, u64)> = vec![
            ("position_key", self.position_key_ns, self.position_key_count),
            ("loop_detect", self.loop_detect_ns, self.loop_detect_count),
            ("tt_lookup", self.tt_lookup_ns, self.tt_lookup_count),
            ("tt_store", self.tt_store_ns, self.tt_store_count),
            ("movegen_check", self.movegen_check_ns, self.movegen_check_count),
            ("movegen_defense", self.movegen_defense_ns, self.movegen_defense_count),
            ("do_move", self.do_move_ns, self.do_move_count),
            ("undo_move", self.undo_move_ns, self.undo_move_count),
            ("child_init", self.child_init_ns, self.child_init_count),
            ("  ci_do/undo_move", self.child_init_domove_ns, self.child_init_domove_count),
            ("  ci_inline", self.ci_inline_ns, self.ci_inline_count),
            ("  ci_resolve", self.ci_resolve_ns, self.ci_resolve_count),
            ("  ci_early_domove", child_init_uncaptured, 0),
            ("main_loop_collect", self.main_loop_collect_ns, self.main_loop_collect_count),
            ("depth_limit_terminal", self.depth_limit_terminal_ns, self.depth_limit_terminal_count),
            ("nm_promotion_refut", self.nm_promotion_refutable_ns, self.nm_promotion_refutable_count),
            ("capture_tt_lookahead", self.capture_tt_lookahead_ns, self.capture_tt_lookahead_count),
            ("cross_deduce", self.cross_deduce_ns, self.cross_deduce_count),
            ("prefilter", self.prefilter_ns, self.prefilter_count),
        ];

        for (name, ns, count) in &items {
            let us = *ns as f64 / 1000.0;
            let avg_ns = if *count > 0 { *ns / *count } else { 0 };
            let pct = if total_ns > 0 {
                *ns as f64 / total_ns as f64 * 100.0
            } else {
                0.0
            };
            writeln!(f, "{:<25} {:>12.1} {:>10} {:>10} {:>5.1}%",
                name, us, count, avg_ns, pct)?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f, "{:<25} {:>12.1}", "Total measured", total_us)?;
        if self.tt_overflow_count > 0 || self.tt_max_entries_per_position > 0 {
            writeln!(f, "  tt_overflow: {} (proven: {}, working: {}, no_victim: {}), max_entries/pos: {}",
                self.tt_overflow_count,
                self.tt_proven_overflow_count,
                self.tt_working_overflow_count,
                self.tt_overflow_no_victim_count,
                self.tt_max_entries_per_position)?;
        }
        let solve_wall_ns = self.mid_total_ns + self.pns_total_ns;
        if solve_wall_ns > 0 {
            let mid_us = self.mid_total_ns as f64 / 1000.0;
            let pns_us = self.pns_total_ns as f64 / 1000.0;
            let coverage_pct = if self.mid_total_ns > 0 {
                total_ns as f64 / self.mid_total_ns as f64 * 100.0
            } else {
                0.0
            };
            writeln!(f, "  MID wall: {:.1}µs ({} calls), PNS wall: {:.1}µs",
                mid_us, self.mid_total_count, pns_us)?;
            writeln!(f, "  MID profiled coverage: {:.1}%", coverage_pct)?;
        }
        Ok(())
    }
}
