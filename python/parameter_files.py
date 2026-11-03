"""Shared parameter-file output naming for standalone programs."""
from pathlib import Path


def resolve_output_settings(settings, default_output):
    """Map combined-style output_dir/output_prefix to the legacy output path."""
    if not any(name in settings for name in ('output_dir', 'output_prefix')):
        return settings
    if 'output' in settings:
        raise ValueError('Specify output or output_dir/output_prefix, not both')
    default = Path(default_output)
    directory = settings.pop('output_dir', str(default.parent))
    prefix = settings.pop('output_prefix', default.stem)
    if not directory or not prefix or prefix in ('.', '..') or Path(prefix).name != prefix or '\\' in prefix:
        raise ValueError('Require nonempty output_dir and a simple output_prefix without a path')
    settings['output'] = str(Path(directory)/(prefix if prefix.endswith('.npz') else prefix+'.npz'))
    return settings
