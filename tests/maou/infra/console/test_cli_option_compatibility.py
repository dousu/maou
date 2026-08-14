"""Test CLI option compatibility between commands.

このテストは，learn-modelコマンドのオプションが
benchmark-trainingコマンドでも全て使えることを確認する．
"""

from __future__ import annotations

import click
import pytest

from maou.infra.console import (
    learn_model,
    pre_process,
    utility,
)
from maou.interface.learn import normalize_lr_scheduler_name


def get_command_options(command: click.Command) -> set[str]:
    """Clickコマンドから全てのオプション名を抽出する．

    Args:
        command: 解析対象のClickコマンドオブジェクト

    Returns:
        オプション名のセット（"--"プレフィックスなし）
    """
    options = set()
    for param in command.params:
        if isinstance(param, click.Option):
            # 全ての名前を追加（長いオプション名のみを対象）
            for opt_name in param.opts:
                if opt_name.startswith("--"):
                    # "--"を除去してオプション名を追加
                    options.add(opt_name[2:])
    return options


def test_learn_model_options_available_in_benchmark_training() -> (
    None
):
    """learn-modelのオプションが全てbenchmark-trainingで使えることを確認する．

    このテストは，学習コマンド(learn-model)で使えるオプションが
    ベンチマークコマンド(benchmark-training)でも全て使えることを保証する．
    これにより，ユーザーは同じパラメータでベンチマークと実際の学習を
    実行できる．

    ただし，以下のオプションはbenchmark-trainingでは不要なため除外する:
    - epoch: benchmarkでは--max-batchesで制御される
    - resume-from: 学習再開機能（benchmark不要）
    - log-dir, model-dir: 出力ディレクトリ（benchmark不要）
    - output-gcs, gcs-*, output-s3, s3-*: クラウド出力（benchmark不要）
    """
    learn_options = get_command_options(learn_model.learn_model)
    benchmark_options = get_command_options(
        utility.benchmark_training
    )

    # benchmark-trainingに不要なオプション（除外リスト）
    excluded_options = {
        "epoch",  # benchmarkでは--max-batchesで制御
        "early-stopping-patience",  # エポック単位の停止判定（benchmark不要）
        "early-stopping-metric",  # 停止判定の監視指標（benchmark不要）
        "stage3-validation-data-path",  # 検証データ分割（benchmark不要）
        "resume-from",  # チェックポイント再開（不要）
        "resume-backbone-from",  # コンポーネント別再開（不要）
        "stage1-threshold",  # Stage 1閾値（不要）
        "stage2-threshold",  # Stage 2閾値（不要）
        "stage1-max-epochs",  # Stage 1最大エポック（不要）
        "stage1-learning-rate",  # Stage 1学習率（不要）
        "stage2-learning-rate",  # Stage 2学習率（不要）
        "stage2-max-epochs",  # Stage 2最大エポック（不要）
        "log-dir",  # TensorBoardログディレクトリ（不要）
        "model-dir",  # モデル出力ディレクトリ（不要）
        "output-gcs",  # GCS出力（不要）
        "gcs-bucket-name",  # GCS（不要）
        "gcs-base-path",  # GCS（不要）
        "output-s3",  # S3出力（不要）
        "s3-bucket-name",  # S3（不要）
        "s3-base-path",  # S3（不要）
        "save-split-params",  # データ分割パラメータ保存（不要）
        # benchmark-training の Stage 3 (policy+value) は主経路で --batch-size を
        # 直接使うため個別上書きは不要 (stage1/2 は補助経路なので override あり)．
        "stage3-batch-size",
    }

    # 除外リストを適用してチェック対象を絞る
    required_options = learn_options - excluded_options

    # learn-modelの必須オプションが全てbenchmark-trainingにも含まれているか確認
    missing_options = required_options - benchmark_options

    # エラーメッセージを構築
    if missing_options:
        missing_list = sorted(missing_options)
        error_msg = (
            f"以下のlearn-modelオプションがbenchmark-trainingで使用できません:\n"
            f"  {', '.join('--' + opt for opt in missing_list)}\n\n"
            f"learn-modelで使えるオプションは全てbenchmark-trainingでも\n"
            f"使用可能であるべきです．不足している{len(missing_options)}個の\n"
            f"オプションをbenchmark-trainingコマンドに追加してください．"
        )
        pytest.fail(error_msg)


def test_benchmark_training_has_all_learn_model_training_params() -> (
    None
):
    """benchmark-trainingがlearn-modelの学習関連パラメータを全て持つことを確認．

    入力，データソース，モデル，最適化に関連するパラメータが
    両方のコマンドで一致していることを確認する．
    """
    learn_options = get_command_options(learn_model.learn_model)
    benchmark_options = get_command_options(
        utility.benchmark_training
    )

    # 学習パラメータとして共通であるべきオプション
    # （出力，ロギング，チェックポイント関連を除く）
    training_params = {
        # 入力関連
        "stage3-data-path",
        # GPU/デバイス
        "gpu",
        # モデル
        "model-architecture",
        "compilation",
        "detect-anomaly",
        # データ分割
        "test-ratio",
        # バッチ/DataLoader
        "batch-size",
        "dataloader-workers",
        "pin-memory",
        "prefetch-factor",
        # 損失関数
        "policy-loss-ratio",
        "value-loss-ratio",
        # 最適化
        "learning-ratio",
        "momentum",
        "optimizer",
        "optimizer-beta1",
        "optimizer-beta2",
        "optimizer-eps",
    }

    # learn-modelにあることを確認
    learn_missing = training_params - learn_options
    assert not learn_missing, (
        f"learn-modelに以下の学習パラメータがありません: "
        f"{', '.join('--' + opt for opt in sorted(learn_missing))}"
    )

    # benchmark-trainingにもあることを確認
    benchmark_missing = training_params - benchmark_options
    assert not benchmark_missing, (
        f"benchmark-trainingに以下の学習パラメータがありません: "
        f"{', '.join('--' + opt for opt in sorted(benchmark_missing))}"
    )


def test_option_consistency_documentation() -> None:
    """オプションの一貫性に関するドキュメント確認テスト．

    このテストは常に成功するが，現在のオプション状況を
    ドキュメントとして出力する．
    """
    learn_options = get_command_options(learn_model.learn_model)
    benchmark_options = get_command_options(
        utility.benchmark_training
    )

    # 除外リスト（benchmark-trainingに不要なオプション）
    excluded_options = {
        "epoch",
        "early-stopping-patience",
        "early-stopping-metric",
        "stage3-validation-data-path",
        "resume-from",
        "resume-backbone-from",
        "log-dir",
        "model-dir",
        "output-gcs",
        "gcs-bucket-name",
        "gcs-base-path",
        "output-s3",
        "s3-bucket-name",
        "s3-base-path",
        # Stage 3 は benchmark-training の主経路で --batch-size を直接使う
        "stage3-batch-size",
    }

    learn_only = sorted(learn_options - benchmark_options)
    benchmark_only = sorted(benchmark_options - learn_options)
    common = sorted(learn_options & benchmark_options)

    # 除外されたオプションと実際に不足しているオプションを分類
    excluded_and_missing = sorted(
        set(learn_only) & excluded_options
    )
    actually_missing = sorted(
        set(learn_only) - excluded_options
    )

    print("\n=== CLI Option Compatibility Report ===")
    print(f"\n共通オプション数: {len(common)}")
    print(f"learn-model専用: {len(learn_only)}")
    print(f"benchmark-training専用: {len(benchmark_only)}")

    if excluded_and_missing:
        print("\nlearn-modelのみ（benchmark不要として除外）:")
        for opt in excluded_and_missing:
            print(f"  --{opt}")

    if actually_missing:
        print(
            "\nlearn-modelのみ（benchmark-trainingに追加が必要）:"
        )
        for opt in actually_missing:
            print(f"  --{opt}")

    if benchmark_only:
        print("\nbenchmark-trainingのみ:")
        for opt in benchmark_only:
            print(f"  --{opt}")


def _find_option(
    command: click.Command, option_name: str
) -> click.Option | None:
    """Clickコマンドから指定名のオプションを検索する．

    Args:
        command: 検索対象のClickコマンドオブジェクト
        option_name: 検索するオプション名（例: "--input-cache-mode"）

    Returns:
        見つかったオプション，見つからない場合はNone
    """
    for param in command.params:
        if (
            isinstance(param, click.Option)
            and option_name in param.opts
        ):
            return param
    return None


# ------------------------------------------------------------------
# --input-cache-mode が全コマンドから消えていること
# ------------------------------------------------------------------


def _all_commands_that_had_input_cache_mode() -> list[
    tuple[str, click.Command]
]:
    """かつて --input-cache-mode を持っていた全コマンド．"""
    return [
        ("learn-model", learn_model.learn_model),
        ("benchmark-dataloader", utility.benchmark_dataloader),
        ("benchmark-training", utility.benchmark_training),
    ]


@pytest.mark.parametrize(
    "name,command",
    _all_commands_that_had_input_cache_mode(),
    ids=[
        c[0] for c in _all_commands_that_had_input_cache_mode()
    ],
)
def test_input_cache_mode_is_gone_everywhere(
    name: str, command: click.Command
) -> None:
    """--input-cache-mode がどのコマンドにも残っていないこと．

    `/audit-backlog` 2026-08-14 / backlog 行 D5 + O5(d)．

    ``cache_mode`` は「全ファイルをメモリへ 1 本に結合するか」の
    ノブだったが，両モードとも ``__init__`` で全ロードする点は同じで，
    ``memory`` 側は結合の前後で入力を保持するためピークが 2 倍になる
    footgun だった．結合した単一配列を必要とする caller はゼロだったので
    ノブごと廃止した (2026-08-14 のユーザ判断)．

    ``learn-model`` にだけ無いという層をまたいだ非対称 (O5(d)) も，
    公開するノブが無くなったことで自動的に解消する．
    """
    assert (
        _find_option(command, "--input-cache-mode") is None
    ), f"{name} に廃止済みの --input-cache-mode が残っています"


def test_no_command_advertises_a_cache_mode_option() -> None:
    """``cache-mode`` を名前に含むオプションがそもそも無いこと．"""
    for (
        name,
        command,
    ) in _all_commands_that_had_input_cache_mode():
        offenders = [
            opt
            for opt in get_command_options(command)
            if "cache-mode" in opt
        ]
        assert offenders == [], (
            f"{name} に cache-mode 系オプションが残っています: {offenders}"
        )


# ------------------------------------------------------------------
# --stage12-lr-scheduler の一貫性テスト
# ------------------------------------------------------------------


def _get_stage12_lr_scheduler_commands() -> list[
    tuple[str, click.Command]
]:
    """--stage12-lr-scheduler を持つ全コマンドを取得する．"""
    return [
        ("learn-model", learn_model.learn_model),
        ("benchmark-training", utility.benchmark_training),
    ]


@pytest.mark.parametrize(
    "name,command",
    _get_stage12_lr_scheduler_commands(),
    ids=[c[0] for c in _get_stage12_lr_scheduler_commands()],
)
def test_stage12_lr_scheduler_choices_are_all_resolvable(
    name: str, command: click.Command
) -> None:
    """広告した選択肢が全て正規化を通ることを確認する．

    回帰テスト: benchmark-training は
    ``["warmup_cosine_decay", "cosine_annealing", "step"]`` を
    ハードコードしていたが，別名表に無い ``cosine_annealing`` /
    ``step`` は ``normalize_lr_scheduler_name`` で ValueError になり，
    CLI が決して動かない選択肢を2つ広告していた．
    """
    option = _find_option(command, "--stage12-lr-scheduler")
    assert option is not None, (
        f"{name} に --stage12-lr-scheduler オプションがありません"
    )
    assert isinstance(option.type, click.Choice)
    for choice in option.type.choices:
        # 'auto'/'none' は learn-model 側で解釈される制御値であり，
        # スケジューラ名として正規化されない
        if choice in ("auto", "none"):
            continue
        assert normalize_lr_scheduler_name(choice), (
            f"{name} の --stage12-lr-scheduler が"
            f"解決不能な選択肢 {choice!r} を広告しています"
        )


def test_learn_model_removed_options_not_in_help() -> None:
    """learn-modelから削除されたオプションがヘルプに表示されないことを確認する．"""
    removed_options = {
        "cache-transforms",
        "no-cache-transforms",
        "input-cache-mode",
        "input-file-packed",
    }
    learn_options = get_command_options(learn_model.learn_model)
    for opt in removed_options:
        assert opt not in learn_options, (
            f"learn-model に削除済みオプション --{opt} が残っています"
        )


# ------------------------------------------------------------------
# ローカルキャッシュの有効化は dir に一本化されている
# ------------------------------------------------------------------


def _commands_with_cloud_input() -> list[
    tuple[str, click.Command]
]:
    """クラウド入力を受け付ける全コマンド．"""
    return [
        ("pre-process", pre_process.pre_process),
        ("benchmark-dataloader", utility.benchmark_dataloader),
        ("benchmark-training", utility.benchmark_training),
    ]


@pytest.mark.parametrize(
    "name,command",
    _commands_with_cloud_input(),
    ids=[c[0] for c in _commands_with_cloud_input()],
)
def test_input_local_cache_flag_is_gone(
    name: str, command: click.Command
) -> None:
    """bool flag ``--input-local-cache`` が残っていないこと．

    `/audit-backlog` 2026-08-14 / backlog 行 O5(a)．

    flag は BigQuery の ``use_local_cache=`` にしか渡っておらず，
    S3/GCS の分岐は ``--input-local-cache-dir`` だけを見ていた．
    ``--input-s3`` に flag だけを渡すと S3 の elif を外れ，入力ソースが
    未指定であるかのような誤誘導エラーで停止していた．
    2026-08-14 のユーザ判断で dir に一本化した．
    """
    options = get_command_options(command)
    assert "input-local-cache" not in options, (
        f"{name} に廃止済みの --input-local-cache が残っています"
    )


@pytest.mark.parametrize(
    "name,command",
    _commands_with_cloud_input(),
    ids=[c[0] for c in _commands_with_cloud_input()],
)
def test_input_local_cache_dir_is_the_only_switch(
    name: str, command: click.Command
) -> None:
    """キャッシュ系オプションが ``--input-local-cache-dir`` 1 本であること．"""
    options = get_command_options(command)
    assert "input-local-cache-dir" in options, (
        f"{name} に --input-local-cache-dir がありません"
    )
    local_cache_options = {
        opt
        for opt in options
        if opt.startswith("input-local-cache")
    }
    assert local_cache_options == {"input-local-cache-dir"}, (
        f"{name} のローカルキャッシュ系オプションが "
        f"1 本になっていません: {sorted(local_cache_options)}"
    )
