import json
import logging
from logging.handlers import RotatingFileHandler
from os import makedirs, path
from pprint import pformat
from sys import stderr

__all__ = ["parse_config", "fetch_class", "init_class"]


def parse_config(config_fp, raw_config_override=None):
    with open(config_fp, "r") as config_f:
        config = json.load(config_f)

    if raw_config_override is not None:
        override = json.loads(raw_config_override)
        print("Overriding configuration. Deep merge in:\n{}".format(pformat(override)))
        config = deep_merge_dict(config, override)

    print("Configuration:\n{}".format(pformat(config)))

    exp_name = config.get("exp_name")
    if exp_name is None:
        print("FATAL: Configuration 'exp_name' must be set.", file=stderr)
        exit(-1)
    else:
        print("=" * 10 + "\nExperiment: {}\n".format(exp_name) + "=" * 10)

    # setup the experiment required directories
    # output directory root
    od_root = config.get("od_root", "experiments")
    # tensorboard summary writer directory
    od_summary = config.get("tb_dir", path.join(od_root, exp_name, "tb"))
    # model checkpoint directory
    od_chkpnt = config.get("chkpt_dir", path.join(od_root, exp_name, "checkpoints"))
    # experiment output directory
    od_out = config.get("out_dir", path.join(od_root, exp_name, "out"))
    # experiment logs directory
    od_logs = config.get("log_dir", path.join(od_root, exp_name, "logs"))

    create_dirs([od_summary, od_chkpnt, od_chkpnt, od_out, od_logs])
    config["tb_dir"] = od_summary
    config["chkpt_dir"] = od_chkpnt
    config["out_dir"] = od_out
    config["log_dir"] = od_logs

    # setup project wide logging
    configure_logging(config["log_dir"])
    logger = logging.getLogger()

    logger.warn(pformat(config))
    logger.info("tensorboard output: {}".format(config["tb_dir"]))
    logger.info("checkpoint output: {}".format(config["chkpt_dir"]))
    logger.info("output directory: {}".format(config["out_dir"]))
    logger.info("log output: {}".format(config["log_dir"]))

    # Instantiate the agent instance
    agent_query = config.get("agent", "agents.BaseAgent")
    agent_class = fetch_class(agent_query)
    agent_instance = agent_class(config)

    return config, agent_instance


def configure_logging(log_dir, log_max_bytes=10 * 1024 * 1024, log_backup_count=5):
    log_file_format = (
        "[%(levelname)s] %(asctime)s %(name)s: %(message)s @ %(pathname)s:%(lineno)d"
    )
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    # Console handles INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    # File handles INFO
    exp_info_file_handler = RotatingFileHandler(
        path.join(log_dir, "info.log"),
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
    )
    exp_info_file_handler.setLevel(logging.INFO)
    exp_info_file_handler.setFormatter(logging.Formatter(log_file_format))

    # Warning file handles WARN
    exp_warn_file_handler = RotatingFileHandler(
        path.join(log_dir, "warn.log"),
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
    )
    exp_warn_file_handler.setLevel(logging.WARNING)
    exp_warn_file_handler.setFormatter(logging.Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_info_file_handler)
    main_logger.addHandler(exp_warn_file_handler)


def create_dirs(dirs):
    for d in dirs:
        makedirs(d, exist_ok=True)


def deep_merge_dict(a, b, path=[], overwrite=True):
    """Deeply merges b into a"""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deep_merge_dict(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            elif overwrite:
                a[key] = b[key]
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def fetch_class(query):
    """Given a query string, return the class"""
    *module, _class = query.split(".")
    mod = __import__(".".join(module), fromlist=[_class])
    return getattr(mod, _class)


def init_class(config, *args, **kwargs):
    """Given a class definition dictionary, return an instance of the class"""
    _class = fetch_class(config["name"])
    return _class(*args, *config.get("args", []), **kwargs, **config.get("kwargs", {}))
