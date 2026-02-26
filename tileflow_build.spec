# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['tileflow/cli.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('tileflow', 'tileflow'),
    ],
    hiddenimports=[
        'clui',
        'clui.clui',
        'numpy',
        'huggingface_hub',
        'huggingface_hub.utils',
        'huggingface_hub.file_download',
        'huggingface_hub.snapshot_download',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tileflow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='tileflow',
)
