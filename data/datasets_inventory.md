# Blade Defect Datasets Inventory (T1)

候选风机叶片缺陷数据集清单。每行字段说明：

- **ID**: 计划编号
- **Name**: 数据集名称
- **URL**: 已通过 WebSearch / WebFetch 验证可达的页面；不可验证的填 `not-found`
- **License**: 公开页面声明的许可（未声明 → `unspecified`）
- **Estimated sample count**: 公开页面声称的样本量
- **Label format**: 标注格式
- **Has severity?**: 是否含严重度标签（yes/no/unknown）
- **Access**: `accessible` / `requires-request` / `unverified`

| ID | Name | URL | License | Estimated sample count | Label format | Has severity? | Access |
|----|------|-----|---------|------------------------|--------------|---------------|--------|
| D1 | DTU NordTank Drone Inspection Images | https://data.mendeley.com/datasets/hd96prn3nc/2 | CC BY-NC 3.0 | ~701 high-res images (2017+2018) | Bounding boxes + multi-class (VG panel, LE erosion, cracks, lightning receptor 等) via DTU-annotations | unknown | accessible |
| D2 | Kaggle "YOLO Annotated Wind Turbine Surface Damage" | https://www.kaggle.com/datasets/ajifoster3/yolo-annotated-wind-turbines-586x371 | unspecified (Kaggle metadata) | hundreds (586x371 patches) | YOLO (.txt bbox) | no | accessible (Kaggle login required) |
| D2b | Kaggle "Wind Turbine Blade Surfaces Dataset" | https://www.kaggle.com/datasets/kooaslansefat/wind-turbine-blade-surfaces-dataset | unspecified | unknown | image + masks (also mirrored on Mendeley jrmm82m4mv) | unknown | accessible (Kaggle login required) |
| D2c | Kaggle "Damage Detection in Wind Blades Dataset" | https://www.kaggle.com/datasets/ziya07/damage-detection-in-wind-blades-dataset | unspecified | unknown | classification / detection (page metadata not fully scrapeable) | unknown | accessible (Kaggle login required) |
| D3 | Roboflow Universe — nanjing/wind-turbine-blades | https://universe.roboflow.com/nanjing/wind-turbine-blades | CC BY 4.0 | 273 images | YOLO / COCO (Roboflow export, 4 classes: breakage, corrosion, fissure, macrolesion) | no | accessible |
| D3b | Roboflow Universe — abc/wind-turbine-faults-detection | https://universe.roboflow.com/abc-bdhsn/wind-turbine-faults-detection-computer-vision-project | CC BY 4.0 | 5288 images | YOLO / COCO (Crack, Damage, Erosion 等) | no | accessible |
| D4 | Mendeley — DTU Wind Turbine Blade Damage Inspection (Thermography) | https://data.mendeley.com/datasets/jmm33c6dny/1 | CC BY 4.0 | unspecified (thermal sequences) | thermal images for AQUADA/AQUADA+ damage quantification | unknown | accessible |
| D4b | Mendeley — Wind Turbine Blade Surfaces Dataset | https://data.mendeley.com/datasets/jrmm82m4mv/1 | unspecified | unknown | RGB images for SfM + damage detection | no | accessible |
| D4c | Mendeley — Vibration Fault Diagnosis | https://data.mendeley.com/datasets/5d7vbdp8f7/4 | CC BY 4.0 | 6 vibration sequences (3 healthy + 3 faulty) | uniaxial vibration CSV; faults: erosion, crack, mass imbalance, twist | no (binary fault type) | accessible |
| D5 | GitHub — cong-yang/Blade30 | https://github.com/cong-yang/Blade30 | unspecified (download via Baidu/OneDrive/Google Drive) | 1,302 drone images covering 30 blades | JSON (defects, contaminations, segmentation masks) | no | accessible |
| D5b | GitHub — DTU-annotations (annotations for D1) | https://github.com/imadgohar/DTU-annotations | unspecified | 701 (matches D1 images) | train/val/test 70:15:15 bbox annotations | unknown | accessible |
| D5c | Figshare — WTBD Multiclass UAV Dataset (Sci Data 2026) | https://springernature.figshare.com/articles/dataset/Multiclass_Dataset_for_Intelligent_Detection_of_Wind_Turbine_Blade_Defects_Using_Drone_Imagery/30210175 | CC BY 4.0 (Springer Nature Figshare default) | 1,065 images, 6 classes | PASCAL VOC XML | no (fine-grained defect taxonomy, not severity) | accessible |
| D6 | AIRA / WindGuard public demo | not-found | unspecified | unspecified | unspecified | unknown | unverified |
| D7 | PHM Society Data Challenge — wind turbine blade | not-found (repository lists only NASA mirror + ballscrew sets) | unspecified | unspecified | unspecified | unknown | unverified |
| D8 | IEEE DataPort — Drone Optical & Thermal Videos of Rotor Blades | https://ieee-dataport.org/documents/drone-based-optical-and-thermal-videos-rotor-blades-taken-normal-wind-turbine-operation | IEEE DataPort terms (subscriber download) | unspecified | optical + thermal video | unknown | requires-request |
| D9 | Mendeley — DTU Risø Blade Inspection Video Dataset (2024) | https://data.mendeley.com/datasets/6nzbdvjn87/1 | unspecified (Mendeley default CC BY) | 29 videos | video-level metadata | unknown | accessible |
| D10 | Mendeley — Sandpaper Wind Turbine Blade Benchmark | https://data.mendeley.com/datasets/hcgcnm269w/2 | unspecified | unknown | SfM benchmark images | no | accessible |
| D11 | GitHub — adions025/Damage_Detection_MaskRCNN (uses DTU images) | https://github.com/adions025/Damage_Detection_MaskRCNN | code MIT-style; data inherits DTU | inherits D1 | Mask R-CNN COCO masks | unknown | accessible |

## 验证结论

已确认 accessible 数量 ≥ 3: YES (count = 14: D1, D2, D2b, D2c, D3, D3b, D4, D4b, D4c, D5, D5b, D5c, D9, D10, D11 — D11 borderline)

## 备注

- **严重度标签缺失普遍**：现有公开集多为多类别缺陷（crack / erosion / VG panel 等），鲜少含 1–5 级严重度。EPRI 白皮书提出过 1–5 级评分但未公开数据。后续 T14 需要在 D1 / D5 / D5c 上人工补充严重度标注，或采用规则映射。
- **Kaggle 条目**需登录，但账号免费即可下载，归类为 accessible；如要无人值守批量下载需配置 `kaggle.json` 凭据。
- **D6 / D7** 未在公开索引中找到匹配项；保留为占位，后续如发现可更新。
- **D8** IEEE DataPort 部分集需要订阅或机构访问，标 requires-request。
