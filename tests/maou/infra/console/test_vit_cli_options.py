"""Tests for ViT CLI options in learn_model command."""

from __future__ import annotations

import click.testing

import maou.infra.console.learn_model as learn_model


class TestViTCLIOptions:
    """ViT CLIオプションの登録テスト."""

    def _get_option_names(self) -> list[str]:
        """learn_model commandの全オプション名を取得."""
        return [
            param.name
            for param in learn_model.learn_model.params
            if isinstance(param, click.Option)
            and param.name is not None
        ]

    def test_vit_embed_dim_option_registered(self) -> None:
        """--vit-embed-dimオプションが登録されている."""
        names = self._get_option_names()
        assert "vit_embed_dim" in names

    def test_vit_num_layers_option_registered(self) -> None:
        """--vit-num-layersオプションが登録されている."""
        names = self._get_option_names()
        assert "vit_num_layers" in names

    def test_vit_num_heads_option_registered(self) -> None:
        """--vit-num-headsオプションが登録されている."""
        names = self._get_option_names()
        assert "vit_num_heads" in names

    def test_vit_mlp_ratio_option_registered(self) -> None:
        """--vit-mlp-ratioオプションが登録されている."""
        names = self._get_option_names()
        assert "vit_mlp_ratio" in names

    def test_vit_dropout_option_registered(self) -> None:
        """--vit-dropoutオプションが登録されている."""
        names = self._get_option_names()
        assert "vit_dropout" in names

    def test_vit_options_default_to_none(self) -> None:
        """ViTオプションのデフォルト値がNone."""
        param_map = {
            param.name: param
            for param in learn_model.learn_model.params
            if isinstance(param, click.Option)
        }
        for name in (
            "vit_embed_dim",
            "vit_num_layers",
            "vit_num_heads",
            "vit_mlp_ratio",
            "vit_dropout",
        ):
            assert param_map[name].default is None, (
                f"{name} should default to None"
            )

    def test_help_text_contains_vit_options(self) -> None:
        """--helpテキストにViTオプションが含まれる."""
        runner = click.testing.CliRunner()
        result = runner.invoke(
            learn_model.learn_model, ["--help"]
        )
        assert "--vit-embed-dim" in result.output
        assert "--vit-num-layers" in result.output
        assert "--vit-num-heads" in result.output
        assert "--vit-mlp-ratio" in result.output
        assert "--vit-dropout" in result.output
