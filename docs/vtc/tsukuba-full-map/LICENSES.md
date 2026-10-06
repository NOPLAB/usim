# Asset licensing audit

Audit date: 2026-10-05. This distinguishes verified declarations from unresolved
permissions. It does not treat a repository license as ownership of imported
third-party content. The public PCD-only artifact excludes the mixed-art
classes below; existing local archives are not a redistribution clearance.

## Primary sources

- VTC revision `c503a8827cb1134a4931fd93cb1af093df5ba35f`:
  [license exceptions](https://github.com/furo-org/VTC/blob/c503a8827cb1134a4931fd93cb1af093df5ba35f/readme.md),
  [required asset packs](https://github.com/furo-org/VTC/blob/c503a8827cb1134a4931fd93cb1af093df5ba35f/docs/editor.md),
  [packaged terms](https://github.com/furo-org/VTC/blob/c503a8827cb1134a4931fd93cb1af093df5ba35f/docs/readme_packaged.md).
- [Official fuRo dataset catalog](https://github.com/tsukubachallenge/tc-datasets).
- [CC BY-NC-SA 4.0 legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
- [Epic Content License Agreement](https://www.unrealengine.com/en-US/eula/content).
- [Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal).
- [Content historical changes](https://www.unrealengine.com/eula-change-log/content)
  and [Engine historical changes](https://www.unrealengine.com/eula-change-log/unreal).
- [Pinned Blender source declaration](https://github.com/Field-Robotics-Japan/vtc_world_blender/blob/2e6087aea800876e673e1e713a720728481e3f84/README.md).
- [CagePlugin declaration](https://github.com/furo-org/CagePlugin/blob/e2572be58b5fd25acada749550f1fb9b786b481e/readme.md)
  and [license](https://github.com/furo-org/CagePlugin/blob/e2572be58b5fd25acada749550f1fb9b786b481e/LICENSE).
- [GSI reuse procedures](https://www.gsi.go.jp/LAW/2930-qa.html).

## Per-class decisions

Paths in this table are from the local `assets/vtc-full/` audit inventory.
Excluded classes are not dependencies of the public point-cloud-derived map.

| Class | Evidence and conditions | Public artifact decision |
| --- | --- | --- |
| Official 2018/2019 fuRo PCDs | Dataset-specific CC BY-NC-SA 4.0 declarations; 17,002,094 and 22,356,688 records independently hashed. NC applies to reproduction/adaptation, not just distribution. | Included as separately licensed map data; never blanket MIT. |
| VTC split voxel clouds | VTC explicitly identifies the 2019 fuRo source and its CC conditions. | Not copied; regenerate from the official PCD. |
| Packaged unsplit cloud | Same count is not independent bytewise lineage proof. Unchanged-package permission does not grant extracted-USD rights. | Excluded; directly licensed PCD replaces this source. |
| Earlier `TsukubaChallenge_2018.ply` | Pinned Blender README identifies Stencil2 data, attributes 防衛大学校ソフトウェア工学講座 and declares Apache-2.0. It has 13,964,093 points, not the public fuRo PCD's 17,002,094. Underlying lineage and registration remain unresolved. | Excluded; do not mislabel it as fuRo/CC solely by location/year. |
| City Hall, chair and table originals | VTC names six model originals under CC BY-NC-SA 4.0; attribution includes Tomoaki Yoshida. Blender's Apache declaration conflicts. Converted tables and concrete materials also reference Epic wood/concrete textures. | Excluded from this map; textured converted assets are not blanket CC or MIT. |
| VTC-owned placement/BSP/custom geometry and logic | Otherwise-unspecified author-owned VTC contributions are Apache-2.0, copyright 2017-2020 Tomoaki Yoshida. Imported art does not inherit this ownership. | Excluded to keep the map's provenance limited to official PCDs. |
| Landscape, heights and masks | VTC excludes these from the general license; cites GSI map/5 m DEM sheets 544000/544010, approval R 2JHs 231, plus unpublished-at-the-time 2020 fuRo measurements. GSI source/processing notices and relevant reuse procedure still need checking. | Excluded; approval number alone is not downstream clearance. |
| StarterContent textures/shapes | Current Engine EULA section 1 treats StarterContent as Licensed Technology. Section 4 distribution and section 4(a)(i)'s source-format restriction matter. PNG, GLTF or editable USD is not automatically permitted by using a binary container. | Excluded; no extracted source art in the map or software wheel. |
| Engine sky/basic meshes | Actual sky placement references `/Engine/EngineSky/SM_SkySphere`; copying a Blueprint does not transfer ownership. Engine EULA section 5(b)'s Examples permission must be established asset-by-asset. | Excluded; owned sky/visualization instead. |
| KiteDemo/Open World Demo foliage | VTC setup requires this pack; local assembly uses it. Current listing does not establish historical account entitlement or UE-only designation. | Excluded; neither unrestricted portability nor blanket UE-only status is assumed from the name. |
| Paragon rocks | VTC setup requires Paragon Agora/Monolith; official listing identifies Unreal projects and naming/advertising restrictions. An Isaac USD redistribution permission was not verified. | Excluded. |
| Megascans asphalt/tiles | VTC requires MS_AsphaltEss/MS_StTilesEss. Megascans Addendum varies by acquisition plan; UE-plan assets are UE-Only. Original plan, entitlement and accepted terms are unknown. Modern Fab listings do not automatically relicense 2020 bytes. | Excluded. |
| Custom leaf/bark images | Filenames suggest cgbookcase Autumn Leaf 06/Bark 03; the site states CC0, but exact local-byte provenance was not verified. | Excluded; candidate CC0 is not verified CC0. |
| CagePlugin reference marker | Pinned submodule declares MPL-2.0, copyright 2019 Tomoaki Yoshida. Model-specific scope still needs confirmation. Hidden assets matter if distributed. | Excluded. |
| Reference-map images | OSM-like filename does not establish tile supplier/style terms. Two other map images lack verified original-source permission. OSM database rights and rendered-map rights differ. | Excluded. |

Archived Mannequin, Puffin and Velodyne exports are not assembled into the
displayed scene. Infinity Blade is mentioned in upstream setup, but was not
found as an exported/assembled source class. Do not incidentally ship the
entire extraction directory.

## Clause distinctions

CC sections 2(a)(1), 3(a), 3(b) and 4 govern NC use, supplied attribution,
modification indication, shared adaptations and database extraction. A
separate MIT converter does not automatically become CC; a format-only
conversion is not automatically Adapted Material under section 2(a)(4).

Epic Content sections 3(a) and 4 distinguish inseparable object-code Projects
from source-format sharing. Section 3(b)'s rendered-linear-media permission
does not grant texture/mesh distribution or waive underlying UE-only terms.
Sections 2, 5(a), 5(c)(i) and 8(c) mean "local-only" is not a universal grant
for extraction, translation or non-Unreal use.

Epic Content section 5(b) explicitly identifies CC Attribution-ShareAlike
among incompatible combinations. CC BY-NC-SA also has SA obligations. This
does not establish that every independent co-presence is incompatible, but
prevents asserting that a combined Epic-textured CC model can simply be
licensed wholesale CC. Separability and permission need specific analysis.

Current terms do not prove which historical agreement an account accepted.
Content section 7(b) and the official change logs distinguish existing and
subsequently accepted terms. Required unresolved evidence includes account
entitlement, acquisition plan/date, UE-only designation, accepted agreement
version and any permission for extraction or non-Unreal runtime use.

The packaged VTC clause is narrow:

> 本パッケージはパッケージの内容を変更しなければ再配布して構いません。

It permits redistribution of the unchanged package; the extracted and
converted USD directory is not that package, and the clause does not
sublicense third-party artwork.

## Additional authoritative asset links

- [KiteDemo listing](https://www.fab.com/listings/3262ab8f-f64a-4124-8efd-82cb19df6249).
- [Paragon Agora/Monolith listing](https://www.unrealengine.com/marketplace/en-US/product/paragon-agora-and-monolith-environment).
- [cgbookcase license declaration](https://www.cgbookcase.com/textures/),
  [leaf candidate](https://www.cgbookcase.com/textures/autumn-leaf-06),
  [bark candidate](https://www.cgbookcase.com/textures/bark-03).
- [OSM copyright and data terms](https://www.openstreetmap.org/copyright).

The PCD-only variant avoids these mixed-art permissions gaps. It remains
NonCommercial/ShareAlike map data, not a commercially unrestricted map.
