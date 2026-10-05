"""Command-line entry points for training recipes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from minirl.common.config import ConfigError, dump_yaml_config
from minirl.recipes.sft import load_sft_config, prepare_sft_data, run_sft


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="minirl", description="Train miniRL recipes from YAML configuration"
    )
    commands = parser.add_subparsers(dest="recipe", required=True)
    sft = commands.add_parser(
        "sft", help="Supervised fine-tuning for RL initialization"
    )
    sft.add_argument(
        "--config", type=Path, required=True, help="Path to a recipe YAML file"
    )
    actions = sft.add_mutually_exclusive_group()
    actions.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print resolved configuration without downloads",
    )
    actions.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate/tokenize data and print statistics without loading model weights",
    )
    actions.add_argument(
        "--resume",
        action="store_true",
        help="Resume trainer_state.pt in the configured output directory",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    try:
        config = load_sft_config(args.config)
    except ConfigError as error:
        parser.error(str(error))
    if args.check_config:
        print(dump_yaml_config(config), end="")
    elif args.prepare_only:
        _, _, _, statistics = prepare_sft_data(config)
        print(json.dumps(statistics, indent=2))
    else:
        run_sft(config, resume=args.resume)
