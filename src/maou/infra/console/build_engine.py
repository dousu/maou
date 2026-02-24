from pathlib import Path

import click

import maou.interface.build_engine as build_engine_iface
from maou.infra.console.common import handle_exception


@click.command("build-engine")
@click.option(
    "--model-path",
    help="ONNX model file path.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "-o",
    help="Output TensorRT engine file path.",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--trt-workspace-size",
    help="TensorRT workspace size in MB. "
    "Default is sufficient for this project's models. "
    "Increase for larger models or max speed. "
    "Decrease if GPU memory is limited.",
    type=int,
    default=256,
    show_default=True,
    required=False,
)
@handle_exception
def build_engine(
    model_path: Path,
    output: Path,
    trt_workspace_size: int,
) -> None:
    """Build a TensorRT engine from an ONNX model and save to file."""
    click.echo(
        build_engine_iface.build_engine(
            model_path=model_path,
            output=output,
            trt_workspace_size=trt_workspace_size,
        )
    )
