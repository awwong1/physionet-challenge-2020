#!/usr/bin/env python3
from argparse import ArgumentParser
from util.config import parse_config
from traceback import format_exc


def main():
    parser = ArgumentParser("Research Experiment Runner")
    parser.add_argument(
        "config", metavar="config_json", help="Experiment configuration JSON file"
    )
    parser.add_argument(
        "--override",
        metavar="override_json",
        default=None,
        type=str,
        help="Serialized JSON object to merge into configuration (overrides config)",
    )
    args = parser.parse_args()

    config, agent = parse_config(args.config, raw_config_override=args.override)

    try:
        agent.run()
    except BaseException:
        print(format_exc())
    finally:
        agent.finalize()


if __name__ == "__main__":
    main()
