from PyInstaller.utils.hooks import collect_submodules, copy_metadata

hiddenimports = collect_submodules('jaraco') + [
    'jaraco.text',
    'jaraco.context',
    'jaraco.functools',
    'jaraco.classes',
]

datas = copy_metadata('jaraco.text')