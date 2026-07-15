//! partitioningKey (開始日時) の日付パースと Date32 (epoch からの日数) 変換．
//!
//! Python 側の `_start_date` を移植する:
//! - CSA: `$START_TIME` を `%Y/%m/%d %H:%M:%S` で厳密パース (strptime 相当)．
//!   欠損は `None`，フォーマット不一致は**エラー** (現行 strptime 例外 →
//!   error status を維持)．
//! - KIF: `開始日時` を `%Y/%m/%d %H:%M:%S` / `%Y/%m/%d %H:%M` / `%Y/%m/%d` の
//!   順で試す．欠損・全失敗はいずれも `None` (null)．
//!
//! chrono に依存せず，Howard Hinnant の days_from_civil で epoch 日数に変換する．

/// 日付パースエラー (CSA で START_TIME が不正フォーマットのとき)．
#[derive(Debug, thiserror::Error)]
#[error("invalid START_TIME format: {0:?}")]
pub struct DateParseError(pub String);

/// グレゴリオ暦 (y, m, d) を 1970-01-01 からの日数に変換する．
///
/// <http://howardhinnant.github.io/date_algorithms.html#days_from_civil>
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe as i32 - 719468
}

/// "YYYY/MM/DD" 形式の日付部分を (y, m, d) にパースする．
///
/// 妥当な範囲 (月 1-12，日 1-31) でなければ `None`．
fn parse_ymd(date_part: &str) -> Option<(i32, u32, u32)> {
    let mut it = date_part.split('/');
    let y: i32 = it.next()?.parse().ok()?;
    let m: u32 = it.next()?.parse().ok()?;
    let d: u32 = it.next()?.parse().ok()?;
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some((y, m, d))
}

/// CSA `$START_TIME` → Date32 日数．
///
/// `%Y/%m/%d %H:%M:%S` に厳密一致しない場合は `Err` (error status)．
pub fn csa_start_date(start_time: &str) -> Result<i32, DateParseError> {
    let err = || DateParseError(start_time.to_string());
    let mut parts = start_time.split(' ');
    let date_part = parts.next().ok_or_else(err)?;
    let time_part = parts.next().ok_or_else(err)?;
    if parts.next().is_some() {
        return Err(err());
    }
    // 時刻は HH:MM:SS を要求 (strptime %H:%M:%S 相当)
    let mut t = time_part.split(':');
    let (h, mi, s) = (t.next(), t.next(), t.next());
    if t.next().is_some()
        || h.and_then(|x| x.parse::<u32>().ok()).is_none()
        || mi.and_then(|x| x.parse::<u32>().ok()).is_none()
        || s.and_then(|x| x.parse::<u32>().ok()).is_none()
    {
        return Err(err());
    }
    let (y, m, d) = parse_ymd(date_part).ok_or_else(err)?;
    Ok(days_from_civil(y, m, d))
}

/// KIF `開始日時` → Date32 日数．パースできなければ `None` (null)．
pub fn kif_start_date(start_time: &str) -> Option<i32> {
    // 日付部分 (先頭トークン) だけ取れれば良い (3 フォーマットは日付部分共通)．
    let date_part = start_time.split(' ').next()?;
    let (y, m, d) = parse_ymd(date_part)?;
    Some(days_from_civil(y, m, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_days_from_civil_epoch() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
    }

    #[test]
    fn test_csa_start_date_ok() {
        assert_eq!(
            csa_start_date("2025/01/05 12:34:56").unwrap(),
            days_from_civil(2025, 1, 5)
        );
    }

    #[test]
    fn test_csa_start_date_malformed_errors() {
        assert!(csa_start_date("2025/01/05").is_err()); // 時刻なし
        assert!(csa_start_date("2025-01-05 12:00:00").is_err()); // 区切り違い
        assert!(csa_start_date("garbage").is_err());
    }

    #[test]
    fn test_kif_start_date_formats() {
        let expected = Some(days_from_civil(2025, 1, 5));
        assert_eq!(kif_start_date("2025/01/05 12:34:56"), expected);
        assert_eq!(kif_start_date("2025/01/05 12:34"), expected);
        assert_eq!(kif_start_date("2025/01/05"), expected);
    }

    #[test]
    fn test_kif_start_date_malformed_is_none() {
        assert_eq!(kif_start_date("not a date"), None);
        assert_eq!(kif_start_date("2025/13/05"), None); // 月 out of range
                                                        // 曜日サフィックス付きは Python strptime も失敗 → None (現行挙動一致)
        assert_eq!(kif_start_date("2025/01/05(日) 12:34:56"), None);
    }
}
