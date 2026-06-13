# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ultralytics_data, ultralytics_binaries, ultralytics_hiddenimports = collect_all("ultralytics")
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=ultralytics_binaries,
    datas=ultralytics_data + [("settings.json.sample", "."), ("assets/app_icon.png", "assets")],
    hiddenimports=ultralytics_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NSFW Auto Mosaic Flow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="assets/app_icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NSFW Auto Mosaic Flow",
)
