import argparse
import ast


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Override Config Items", allow_abbrev=False)
    parser.add_argument(
        "opts",
        help='Modify config use "key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Parse.
    args = parser.parse_args()
    configs = {}
    for o in args.opts:
        if "=" not in o:
            continue
        key, value = o.split("=")
        try:
            value = ast.literal_eval(value)
        except:
            pass
        configs[key] = value

    return configs
