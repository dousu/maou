"""Test CLI option compatibility between commands.

このテストは，learn-modelコマンドのオプションが
benchmark-trainingコマンドでも全て使えることを確認する．
"""

from __future__ import annotations

import types

import click
import pytest

import maou.infra.console.learn_model as learn_model
import maou.infra.console.utility as utility


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
    - resume-from, start-epoch: 学習再開機能（benchmark不要）
    - log-dir, model-dir: 出力ディレクトリ（benchmark不要）
    - tensorboard-*: TensorBoard関連（benchmark不要）
    - output-gcs, gcs-*, output-s3, s3-*: クラウド出力（benchmark不要）
    """
    learn_options = get_command_options(learn_model.learn_model)
    benchmark_options = get_command_options(
        utility.benchmark_training
    )

    # benchmark-trainingに不要なオプション（除外リスト）
    excluded_options = {
        "epoch",  # benchmarkでは--max-batchesで制御
        "resume-from",  # チェックポイント再開（不要）
        "start-epoch",  # 開始エポック（不要）
        "resume-backbone-from",  # コンポーネント別再開（不要）
        "resume-policy-head-from",  # コンポーネント別再開（不要）
        "resume-value-head-from",  # コンポーネント別再開（不要）
        "freeze-backbone",  # バックボーン凍結（不要）
        "trainable-layers",  # バックボーン凍結制御（不要）
        "vit-embed-dim",  # ViTアーキテクチャ設定（不要）
        "vit-num-layers",  # ViTアーキテクチャ設定（不要）
        "vit-num-heads",  # ViTアーキテクチャ設定（不要）
        "vit-mlp-ratio",  # ViTアーキテクチャ設定（不要）
        "vit-dropout",  # ViTアーキテクチャ設定（不要）
        "gradient-checkpointing",  # ViTメモリ最適化（不要）
        "stage",  # マルチステージ学習（不要）
        "stage1-data-path",  # Stage 1データパス（不要）
        "stage2-data-path",  # Stage 2データパス（不要）
        "stage3-data-path",  # Stage 3データパス（不要）
        "stage1-threshold",  # Stage 1閾値（不要）
        "stage2-threshold",  # Stage 2閾値（不要）
        "stage1-max-epochs",  # Stage 1最大エポック（不要）
        "stage1-batch-size",  # Stage 1バッチサイズ（不要）
        "stage2-batch-size",  # Stage 2バッチサイズ（不要）
        "stage1-learning-rate",  # Stage 1学習率（不要）
        "stage2-learning-rate",  # Stage 2学習率（不要）
        "stage2-max-epochs",  # Stage 2最大エポック（不要）
        "resume-reachable-head-from",  # Stage 1ヘッド再開（不要）
        "resume-legal-moves-head-from",  # Stage 2ヘッド再開（不要）
        "stage12-lr-scheduler",  # Stage 1/2 LRスケジューラ（不要）
        "stage12-compilation",  # Stage 1/2 torch.compile（不要）
        "no-stage12-compilation",  # Stage 1/2 torch.compile無効化（不要）
        "log-dir",  # TensorBoardログディレクトリ（不要）
        "model-dir",  # モデル出力ディレクトリ（不要）
        "tensorboard-histogram-frequency",  # TensorBoard（不要）
        "tensorboard-histogram-module",  # TensorBoard（不要）
        "output-gcs",  # GCS出力（不要）
        "gcs-bucket-name",  # GCS（不要）
        "gcs-base-path",  # GCS（不要）
        "output-s3",  # S3出力（不要）
        "s3-bucket-name",  # S3（不要）
        "s3-base-path",  # S3（不要）
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
        "gce-parameter",
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
        "resume-from",
        "start-epoch",
        "resume-backbone-from",
        "resume-policy-head-from",
        "resume-value-head-from",
        "freeze-backbone",
        "log-dir",
        "model-dir",
        "tensorboard-histogram-frequency",
        "tensorboard-histogram-module",
        "output-gcs",
        "gcs-bucket-name",
        "gcs-base-path",
        "output-s3",
        "s3-bucket-name",
        "s3-base-path",
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
# --input-cache-mode の一貫性テスト
# ------------------------------------------------------------------


# テスト対象コマンドの遅延取得（pretrain_cli の import を分離）
def _get_input_cache_mode_commands() -> list[
    tuple[str, click.Command]
]:
    """--input-cache-mode を持つ全コマンドを取得する．"""
    import maou.infra.console.pretrain_cli as pretrain_cli

    return [
        ("benchmark-dataloader", utility.benchmark_dataloader),
        ("benchmark-training", utility.benchmark_training),
        ("pretrain", pretrain_cli.pretrain),
    ]


def _get_input_cache_mode_commands_with_modules() -> list[
    tuple[str, click.Command, types.ModuleType]
]:
    """--input-cache-mode を持つ全コマンドとそのモジュールを取得する．

    デコレータで包まれたコールバックからモジュールを特定できないため，
    コマンドとモジュールの対応を明示的に定義する．
    """
    import maou.infra.console.pretrain_cli as pretrain_cli

    return [
        (
            "benchmark-dataloader",
            utility.benchmark_dataloader,
            utility,
        ),
        (
            "benchmark-training",
            utility.benchmark_training,
            utility,
        ),
        ("pretrain", pretrain_cli.pretrain, pretrain_cli),
    ]


@pytest.mark.parametrize(
    "name,command",
    _get_input_cache_mode_commands(),
    ids=[c[0] for c in _get_input_cache_mode_commands()],
)
def test_input_cache_mode_choices_consistency(
    name: str, command: click.Command
) -> None:
    """全コマンドの--input-cache-modeがfile, memory, mmapを受け付けることを確認する．"""
    option = _find_option(command, "--input-cache-mode")
    assert option is not None, (
        f"{name} に --input-cache-mode オプションがありません"
    )
    assert isinstance(option.type, click.Choice)
    choices = set(option.type.choices)
    assert choices == {"file", "memory", "mmap"}, (
        f"{name} の --input-cache-mode choices が不正です: {choices}"
    )


@pytest.mark.parametrize(
    "name,command",
    _get_input_cache_mode_commands(),
    ids=[c[0] for c in _get_input_cache_mode_commands()],
)
def test_input_cache_mode_default_is_file(
    name: str, command: click.Command
) -> None:
    """全コマンドの--input-cache-modeのデフォルト値がfileであることを確認する．"""
    option = _find_option(command, "--input-cache-mode")
    assert option is not None
    assert option.default == "file", (
        f"{name} の --input-cache-mode default が 'file' ではありません: {option.default!r}"
    )


@pytest.mark.parametrize(
    "name,command,module",
    _get_input_cache_mode_commands_with_modules(),
    ids=[
        c[0]
        for c in _get_input_cache_mode_commands_with_modules()
    ],
)
def test_input_cache_mode_has_mmap_deprecation(
    name: str,
    command: click.Command,
    module: types.ModuleType,
) -> None:
    """全コマンドにmmap→file変換のdeprecation warningが含まれることを確認する．"""
    import inspect

    # command.callback はデコレータ(handle_exception)で包まれており，
    # inspect.getmodule が common.py を返すため，モジュールを明示的に渡す
    module_source = inspect.getsource(module)
    assert "'mmap' is deprecated" in module_source, (
        f"{name} のモジュールに mmap→file の deprecation warning ロジックがありません"
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
