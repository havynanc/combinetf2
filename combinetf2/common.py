import pathlib
import re
import importlib.util
import sys

base_dir = f"{pathlib.Path(__file__).parent}/../"


def natural_sort_key(s):
    # Sort string in a number aware way by plitting the string into alphabetic and numeric parts
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def natural_sort(strings):
    return sorted(strings, key=natural_sort_key)


def natural_sort_dict(dictionary):
    sorted_keys = natural_sort(dictionary.keys())
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict


def load_config(config_path):
    if config_path is None:
        return {}
    # load a python module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def read_axis_label(x, labels, with_unit=True):
    if x in labels:
        label = labels[x]
        if isinstance(label, str):
            return label
        elif with_unit:
            return f'{label["label"]} ({label["unit"]})'
        else:
            return label["label"]
    else:
        return x


def get_axis_label(config, default_keys=None, label=None, is_bin=False):
    if label is not None:
        return label

    if default_keys is None:
        return "Bin index"

    labels = getattr(config, "axis_labels", {})

    if len(default_keys) == 1:        
        if is_bin:
            return f"{read_axis_label(default_keys[0], labels, False)} bin"
        else:
            return read_axis_label(default_keys[0], labels)
    else:
        return f"({', '.join([read_axis_label(a, labels, False) for a in default_keys])}) bin"
    