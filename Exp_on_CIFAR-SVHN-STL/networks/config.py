InfoPro = {
    'resnet32': {
        1: [[3,4]],  # End-to-end
        2: [[2, 1],[3,4]],
        3: [[1, 4], [2, 4],[3,4]],
        4: [[1, 2], [2, 1], [3, 0],[3,4]],
        8: [[1, 0], [1, 2], [1, 4],
            [2, 1], [2, 3],
            [3, 0], [3, 2],[3,4]],
        16: [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
             [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    },
    'resnet110': {
        1: [[3, 17]],  # End-to-end
        2: [[2, 8],[3,17]],
        3: [[1, 17], [2, 17],[3,17]],
        4: [[1, 11], [2, 7], [3, 3],[3,17]],
        8: [[1, 4], [1, 11],
            [2, 0], [2, 7], [2, 14],
            [3, 3], [3, 10],[3, 17]],
        16: [[1, 1], [1, 4], [1, 7], [1, 10], [1, 13], [1, 16],
             [2, 1], [2, 4], [2, 7], [2, 11], [2, 15],
             [3, 1], [3, 5], [3, 9], [3, 13], [3, 17]],
        32: [[1,1],[1,2],[1,4],[1,5],[1,7],[1,9],[1,11],[1,13],[1,14],[1,16],[1,17],
             [2,1],[2,2],[2,4],[2,5],[2,7],[2,9],[2,11],[2,13],[2,14],[2,16],[2,17],
             [3,1],[3,3],[3,4],[3,6],[3,7],[3,9],[3,11],[3,13],[3,15],[3,17]],
        55: [[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],
             [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],[2,11],[2,12],[2,13],[2,14],[2,15],[2,16],[2,17],
             [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[3,15],[3,16],[3,17]],
    }
}

InfoPro_balanced_memory = {
    'resnet110': {
        'cifar10': {
            1: [[3, 17]],  # End-to-end
            2: [[1, 14], [3, 17]],
            3: [[1, 9], [2, 2], [3, 17]],
            4: [[1, 6], [1, 14], [2, 8], [3, 17]],
        },
        'stl10': {
            1: [[3, 17]],  # End-to-end
            2: [[1, 14], [3, 17]],
            3: [[1, 8], [2, 1], [3, 17]],
            4: [[1, 6], [1, 14], [2, 8], [3, 17]],
        }

    }
}


MANPP_PAPER_LOCAL_MODULES = {
    'resnet32': (8, 16),
    'resnet110': (32, 55),
}


def _paper_settings_text():
    return ', '.join(
        f'{arch} K={list(values)}'
        for arch, values in MANPP_PAPER_LOCAL_MODULES.items()
    )


def supported_local_module_nums(arch, dataset='cifar10', balanced_memory=False):
    """Return configured local-module counts for a ResNet variant."""
    if balanced_memory:
        configs = InfoPro_balanced_memory.get(arch, {}).get(dataset, {})
    else:
        configs = InfoPro.get(arch, {})
    return tuple(sorted(configs))


def get_infopro_config(arch, dataset, local_module_num, balanced_memory=False):
    """Fetch MAN++ split config with an explicit error for unsupported K."""
    supported = supported_local_module_nums(arch, dataset, balanced_memory)
    if local_module_num not in supported:
        mode = 'balanced_memory=True' if balanced_memory else 'balanced_memory=False'
        supported_text = list(supported) if supported else 'none'
        raise ValueError(
            f'Unsupported local_module_num={local_module_num} for {arch} '
            f'on {dataset} ({mode}). Supported K values: {supported_text}. '
            f'MAN++ paper settings: {_paper_settings_text()}.'
        )

    configs = InfoPro_balanced_memory[arch][dataset] if balanced_memory else InfoPro[arch]
    return configs[local_module_num]
