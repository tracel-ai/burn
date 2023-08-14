#!/bin/python 

import json
import subprocess
from typing import List, Any

def get_metadata():
    """Gets the metadata from the cargo metadata command."""
    proc = subprocess.run(['cargo', 'metadata', '--no-deps', '--format-version=1'], capture_output=True)
    if proc.returncode != 0:
        raise Exception('Cargo metadata command failed')
    return json.loads(proc.stdout)

def generate_launch_json(metadata):
    """Generates the launch.json file."""
    launch_json = {
        'version': '0.2.0',
        'configurations': []
    }
    def add_config(name: str, cargo_args: List[str], filter: Any, pkg_name: str):
            launch_json['configurations'].append({
                'type': 'lldb',
                'request': 'launch',
                'name': name,
                'cargo': {
                    'args': cargo_args + [f'--package={pkg_name}'],
                    'filter': filter
                },
                'args': [],
                'cwd': '${workspaceFolder}'
            })
    for pkg in metadata['packages']:
        for target in pkg['targets']:
            lib_added = False
            for kind in target['kind']:
                if kind == 'lib' or kind == 'rlib' or kind == 'staticlib' or kind == 'dylib' or kind == 'cstaticlib':
                    if not lib_added:
                        add_config(
                            'Debug unit tests in library \'{}\''.format(target["name"]),
                            ['test', '--no-run', '--lib'],
                            { 'name': target["name"], 'kind': 'lib' },
                            pkg["name"]
                        )
                        lib_added = True
                    continue

                if kind == 'bin' or kind == 'example':
                    pretty_kind = 'executable' if kind == 'bin' else kind
                    add_config(
                        'Debug {} \'{}\''.format(pretty_kind, target["name"]),
                        ['build', '--{}={}'.format(kind, target["name"])],
                        { 'name': target["name"], 'kind': kind },
                        pkg["name"]
                    )
                    add_config(
                        'Debug unit tests in {} \'{}\''.format(pretty_kind, target["name"]),
                        ['test', '--no-run', '--{}={}'.format(kind, target["name"])],
                        { 'name': target["name"], 'kind': kind },
                        pkg["name"]
                    )
                    continue

                if kind == 'bench' or kind == 'test':
                    pretty_kind = 'benchmark' if kind == 'bench' else (
                        'integration test' if kind == 'test' else kind
                    )
                    add_config(
                        'Debug {} \'{}\''.format(pretty_kind, target["name"]),
                        ['test', '--no-run', '--{}={}'.format(kind, target["name"])],
                        { 'name': target["name"], 'kind': kind },
                        pkg["name"]
                    )
                    continue

    with open('.vscode/launch.json', 'w') as f:
        json.dump(launch_json, f, indent=4)

if __name__ == '__main__':
    metadata = get_metadata()
    generate_launch_json(metadata)