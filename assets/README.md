# X2Real public ID asset bundle

Git intentionally tracks only this file and `manifest.json` under `assets/`.
Robot models, object models, textures, task catalogs, asset pools, grasp-label
arrays, and all other runtime payloads are distributed as one versioned asset
bundle.

## Prepare the local bundle

Obtain the versioned bundle through the approved local distribution channel.
Copy its contents into this repository's `assets/` directory, or keep the same
layout in another local directory. For an external location, point ManaEnv at
that `assets/` directory:

```bash
export MANAENV_ASSET_ROOT=/path/assets
```

ManaEnv resolves evaluation assets only from local filesystem paths.

## Verify the version

`manifest.json` identifies the expected bundle by file count, byte count, and a
content-addressed tree digest. The digest covers every payload file except the
two Git control files, `README.md` and `manifest.json`. Paths are relative to the
asset root.

Update the manifest whenever any runtime asset, catalog, pool, or grasp label
changes. A checkout and asset bundle are compatible only when the local payload
matches the manifest version.

## Required top-level layout

```text
art_assets/
benchmark/
catalogs/
grasp_labels/
manipulation/
objects/
pools/
render_settings/
robots/
simready/
```

ArtiXon Arm-6A and Quanta X1 use the legacy `ex001` asset namespace under
`simready/robots/ex001/`; optimized IK URDF assets live under
`art_assets/urdf/optimized/ex001/`. Keep these paths unchanged for compatibility
with the versioned asset bundle. ARX-R5 and Franka remain under `robots/`.
