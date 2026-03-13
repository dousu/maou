use maou_shogi::board::Board;
use maou_shogi::feature;
use maou_shogi::hcp;
use maou_shogi::movegen;
use maou_shogi::moves::Move;
use maou_shogi::types::FEATURES_NUM;

use serde::Deserialize;

fn load_fixture<T: serde::de::DeserializeOwned>(filename: &str) -> T {
    let path = format!(
        "{}/tests/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        filename
    );
    let content = std::fs::read_to_string(&path).expect(&format!("Failed to read: {}", path));
    serde_json::from_str(&content).expect(&format!("Failed to parse: {}", path))
}

// === SFEN Fixtures ===

#[derive(Deserialize)]
struct SfenFixture {
    name: String,
    sfen: String,
    pieces: Vec<u8>,
    hand_black: Vec<u8>,
    hand_white: Vec<u8>,
    turn: u8,
}

#[test]
fn test_sfen_fixtures() {
    let fixtures: Vec<SfenFixture> = load_fixture("sfen_fixtures.json");

    for fixture in &fixtures {
        let mut board = Board::empty();
        board
            .set_sfen(&fixture.sfen)
            .unwrap_or_else(|e| panic!("{}: {}", fixture.name, e));

        let pieces = board.pieces();
        assert_eq!(
            pieces.to_vec(),
            fixture.pieces,
            "{}: pieces mismatch",
            fixture.name
        );

        let (hand_b, hand_w) = board.pieces_in_hand();
        assert_eq!(
            hand_b.to_vec(),
            fixture.hand_black,
            "{}: hand_black mismatch",
            fixture.name
        );
        assert_eq!(
            hand_w.to_vec(),
            fixture.hand_white,
            "{}: hand_white mismatch",
            fixture.name
        );

        assert_eq!(
            board.turn as u8, fixture.turn,
            "{}: turn mismatch",
            fixture.name
        );

        // SFEN round-trip
        let sfen_out = board.sfen();
        assert_eq!(sfen_out, fixture.sfen, "{}: SFEN roundtrip failed", fixture.name);
    }
}

// === Move Fixtures ===

#[derive(Deserialize)]
struct MoveFixture {
    #[serde(rename = "move")]
    #[allow(dead_code)]
    move_val: u32,
    move16: u16,
    to_sq: u8,
    #[allow(dead_code)]
    from_sq: u8,
    usi: String,
    is_drop: bool,
    is_promotion: bool,
}

#[test]
fn test_move_fixtures() {
    let fixtures: Vec<MoveFixture> = load_fixture("move_fixtures.json");

    for fixture in &fixtures {
        // USIからMoveを生成して16-bitエンコーディングを比較
        let m = Move::from_usi(&fixture.usi).expect(&format!("Invalid USI: {}", fixture.usi));

        assert_eq!(
            m.to_move16(),
            fixture.move16,
            "move16 mismatch for USI: {}",
            fixture.usi
        );
        assert_eq!(
            m.to_sq().0,
            fixture.to_sq,
            "to_sq mismatch for USI: {}",
            fixture.usi
        );
        assert_eq!(
            m.is_drop(),
            fixture.is_drop,
            "is_drop mismatch for USI: {}",
            fixture.usi
        );
        assert_eq!(
            m.is_promotion(),
            fixture.is_promotion,
            "is_promotion mismatch for USI: {}",
            fixture.usi
        );

        // USI round-trip
        assert_eq!(m.to_usi(), fixture.usi, "USI roundtrip failed: {}", fixture.usi);
    }
}

// === Legal Move Fixtures ===

#[derive(Deserialize)]
struct LegalMoveFixture {
    name: String,
    sfen: String,
    legal_moves_count: usize,
    legal_moves_usi: Vec<String>,
}

#[test]
fn test_legal_move_fixtures() {
    let fixtures: Vec<LegalMoveFixture> = load_fixture("legal_move_fixtures.json");

    for fixture in &fixtures {
        let mut board = Board::empty();
        board
            .set_sfen(&fixture.sfen)
            .unwrap_or_else(|e| panic!("{}: {}", fixture.name, e));

        let moves = movegen::generate_legal_moves(&mut board);
        let mut usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        usi_moves.sort();

        assert_eq!(
            moves.len(),
            fixture.legal_moves_count,
            "{}: legal moves count mismatch (expected {}, got {})\nexpected: {:?}\ngot: {:?}",
            fixture.name,
            fixture.legal_moves_count,
            moves.len(),
            fixture.legal_moves_usi,
            usi_moves
        );

        assert_eq!(
            usi_moves, fixture.legal_moves_usi,
            "{}: legal moves USI mismatch",
            fixture.name
        );
    }
}

// === Special Rule Fixtures ===

#[derive(Deserialize)]
struct SpecialRuleFixture {
    name: String,
    sfen: String,
    description: String,
    #[serde(default)]
    #[allow(dead_code)]
    pawn_drops_on_file5: Vec<String>,
    #[serde(default)]
    p_drop_5a_is_legal: bool,
    total_legal_moves: usize,
    #[serde(default)]
    #[allow(dead_code)]
    all_moves_usi: Vec<String>,
}

#[test]
fn test_special_rule_fixtures() {
    let fixtures: Vec<SpecialRuleFixture> = load_fixture("special_rule_fixtures.json");

    for fixture in &fixtures {
        let mut board = Board::empty();
        board
            .set_sfen(&fixture.sfen)
            .unwrap_or_else(|e| panic!("{}: {}", fixture.name, e));

        let moves = movegen::generate_legal_moves(&mut board);
        let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();

        assert_eq!(
            moves.len(),
            fixture.total_legal_moves,
            "{}: total legal moves mismatch (expected {}, got {})\ndescription: {}\ngot: {:?}",
            fixture.name,
            fixture.total_legal_moves,
            moves.len(),
            fixture.description,
            {
                let mut sorted = usi_moves.clone();
                sorted.sort();
                sorted
            }
        );

        match fixture.name.as_str() {
            "nifu_check" => {
                // 5筋に歩を打てない
                let pawn_drops_file5: Vec<&String> = usi_moves
                    .iter()
                    .filter(|u| u.starts_with("P*5"))
                    .collect();
                assert!(
                    pawn_drops_file5.is_empty(),
                    "nifu: should not have pawn drops on file 5, but found: {:?}",
                    pawn_drops_file5
                );
            }
            "uchifuzume_check" => {
                // P*5aは打ち歩詰めなので合法手に含まれない
                let has_p5a = usi_moves.iter().any(|u| u == "P*5a");
                assert_eq!(
                    has_p5a, fixture.p_drop_5a_is_legal,
                    "{}: P*5a legality mismatch",
                    fixture.name
                );
            }
            _ => {}
        }
    }
}

// === Feature Plane Fixtures ===

#[derive(Deserialize)]
struct FeatureFixture {
    name: String,
    sfen: String,
    piece_planes: Vec<Vec<Vec<f32>>>,
    piece_planes_rotate: Vec<Vec<Vec<f32>>>,
}

#[test]
fn test_feature_fixtures() {
    let fixtures: Vec<FeatureFixture> = load_fixture("feature_fixtures.json");

    for fixture in &fixtures {
        let mut board = Board::empty();
        board
            .set_sfen(&fixture.sfen)
            .unwrap_or_else(|e| panic!("{}: {}", fixture.name, e));

        // piece_planes
        let mut buf = vec![0.0f32; FEATURES_NUM * 9 * 9];
        feature::piece_planes(&board, &mut buf);

        // 3D配列に変換して比較
        for ch in 0..FEATURES_NUM {
            for col in 0..9 {
                for row in 0..9 {
                    let rust_val = buf[ch * 81 + col * 9 + row];
                    let cshogi_val = fixture.piece_planes[ch][col][row];
                    assert!(
                        (rust_val - cshogi_val).abs() < 1e-6,
                        "{}: piece_planes mismatch at ch={}, col={}, row={}: rust={}, cshogi={}",
                        fixture.name,
                        ch,
                        col,
                        row,
                        rust_val,
                        cshogi_val
                    );
                }
            }
        }

        // piece_planes_rotate
        let mut buf_rot = vec![0.0f32; FEATURES_NUM * 9 * 9];
        feature::piece_planes_rotate(&board, &mut buf_rot);

        for ch in 0..FEATURES_NUM {
            for col in 0..9 {
                for row in 0..9 {
                    let rust_val = buf_rot[ch * 81 + col * 9 + row];
                    let cshogi_val = fixture.piece_planes_rotate[ch][col][row];
                    assert!(
                        (rust_val - cshogi_val).abs() < 1e-6,
                        "{}: piece_planes_rotate mismatch at ch={}, col={}, row={}: rust={}, cshogi={}",
                        fixture.name,
                        ch,
                        col,
                        row,
                        rust_val,
                        cshogi_val
                    );
                }
            }
        }
    }
}

// === HCP Fixtures ===

#[derive(Deserialize)]
struct HcpFixture {
    name: String,
    sfen: String,
    hcp_hex: String,
    roundtrip_sfen: String,
}

#[test]
fn test_hcp_fixtures() {
    let fixtures: Vec<HcpFixture> = load_fixture("hcp_fixtures.json");

    for fixture in &fixtures {
        let mut board = Board::empty();
        board
            .set_sfen(&fixture.sfen)
            .unwrap_or_else(|e| panic!("{}: SFEN parse error: {}", fixture.name, e));

        // maou_shogi -> HCP
        let hcp_bytes = hcp::to_hcp(&board)
            .unwrap_or_else(|e| panic!("{}: HCP encode error: {}", fixture.name, e));
        let hcp_hex: String = hcp_bytes.iter().map(|b| format!("{:02x}", b)).collect();

        assert_eq!(
            hcp_hex, fixture.hcp_hex,
            "{}: HCP mismatch\n  maou_shogi: {}\n  cshogi:     {}",
            fixture.name, hcp_hex, fixture.hcp_hex
        );

        // HCP -> Board (roundtrip)
        let decoded = hcp::from_hcp(&hcp_bytes)
            .unwrap_or_else(|e| panic!("{}: HCP decode error: {}", fixture.name, e));
        let decoded_sfen = decoded.sfen();

        // cshogiのroundtrip SFENと比較
        assert_eq!(
            decoded_sfen, fixture.roundtrip_sfen,
            "{}: roundtrip SFEN mismatch\n  maou_shogi: {}\n  cshogi:     {}",
            fixture.name, decoded_sfen, fixture.roundtrip_sfen
        );
    }
}
