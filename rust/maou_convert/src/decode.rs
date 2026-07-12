//! 棋譜ファイルのバイト列 → 文字列デコード (UTF-8 → cp932 fallback)．
//!
//! Python 側は `file.read_text()` が UTF-8 固定で cp932 の .kif を読めなかった．
//! Rust 側でファイルを直読みする設計に合わせ，内容ベースでエンコーディングを
//! 判定する: UTF-8 として妥当ならそのまま，不正なら Shift_JIS (cp932) で
//! 置換なしデコードを試みる．
//!
//! 判定順序 (UTF-8 先行) が本質的: cp932 デコードはほぼ任意のバイト列で
//! 「成功」してしまうため，先に cp932 を試すと UTF-8 テキストを誤解釈する．

/// デコードエラー: UTF-8 でも cp932 でも解釈できなかった．
#[derive(Debug, thiserror::Error)]
#[error("failed to decode as UTF-8 or Shift_JIS (cp932)")]
pub struct DecodeError;

/// バイト列を UTF-8 → cp932 の順で文字列へデコードする．
///
/// 先頭の UTF-8 BOM は除去する．
pub fn decode_kifu_bytes(bytes: &[u8]) -> Result<String, DecodeError> {
    // UTF-8 BOM を除去
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);

    if let Ok(s) = std::str::from_utf8(body) {
        return Ok(s.to_string());
    }

    // Shift_JIS (cp932) で置換文字を許さずデコード
    let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(body);
    if had_errors {
        return Err(DecodeError);
    }
    Ok(cow.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_passthrough() {
        let s = "開始日時：2025/01/05\n+7776FU";
        assert_eq!(decode_kifu_bytes(s.as_bytes()).unwrap(), s);
    }

    #[test]
    fn test_utf8_bom_stripped() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice("abc".as_bytes());
        assert_eq!(decode_kifu_bytes(&bytes).unwrap(), "abc");
    }

    #[test]
    fn test_shift_jis_fallback() {
        let utf8 = "先手：テスト";
        let (sjis, _, had_errors) = encoding_rs::SHIFT_JIS.encode(utf8);
        assert!(!had_errors);
        // cp932 バイト列は UTF-8 として不正 → fallback で復元される
        assert!(std::str::from_utf8(&sjis).is_err());
        assert_eq!(decode_kifu_bytes(&sjis).unwrap(), utf8);
    }

    #[test]
    fn test_invalid_bytes_error() {
        // 0x81 は Shift_JIS のリードバイト．不正なトレイル (0x20) が続くと
        // WHATWG Shift_JIS デコーダはエラーを立てる (UTF-8 としても不正)．
        // 注: encoding_rs は寛容で単独 0x80 等は置換なしで通ることがある．
        assert!(decode_kifu_bytes(&[0x81, 0x20]).is_err());
    }
}
