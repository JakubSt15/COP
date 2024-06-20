# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['LoginWindow.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('AddUser','AddUser'),
        ('DoctorMenu','DoctorMenu'),
        ('LoginWindow','LoginWindow'),
        ('Model','Model'),
        ('ShowEDF','ShowEDF'),
        ('UserListWindow','UserListWindow'),
        ('CommonTools','CommonTools'),
        ('ModelRun','ModelRun'),
        ('PN00-4.edf','.'),
        ('Users.csv','.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EpilepsyPredictionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
