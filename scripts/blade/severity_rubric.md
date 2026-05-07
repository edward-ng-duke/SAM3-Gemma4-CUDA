# Blade Defect Severity Rubric (4-level)

This rubric defines a deterministic 4-level severity scale (`none` / `mild` / `moderate` / `severe`) used by VLM raters and human annotators when scoring wind-turbine blade defects. Geometric features are computed from the predicted defect mask relative to the blade's chord length (or absolute length for elongated cracks); visual features are read from the RGB frame. The thresholds below are the project-wide defaults and will be calibrated against the host operator's labels once a labelled subset becomes available.

## none

- **几何特征**: 无缺陷掩膜（mask 面积为 0 像素，或低于 IoU 噪声阈值）。
- **视觉特征**: 表面均匀完整，无色斑、起皮、裂纹或穿孔。

## mild

- **几何特征**: 缺陷面积 < 1% 弦长²；若为线状裂纹，单条长度 < 5 cm。
- **视觉特征**: 浅表色斑、微小起皮、单条 < 5 cm 微裂纹；无可见基材外露。

## moderate

- **几何特征**: 缺陷面积占弦长² 的 1%–5%，或单一缺陷长度在 5–30 cm 之间。
- **视觉特征**: 明显裂纹、漆面成片剥落、可见基材（玻纤/复合层）。

## severe

- **几何特征**: 缺陷面积 > 5% 弦长²，或存在多处独立缺陷，或单一缺陷长度 > 30 cm。
- **视觉特征**: 贯穿性裂纹、雷击穿孔、前缘大面积脱落、结构性损伤。

## Examples

1. A 30 m blade with a 3 cm hairline crack near the trailing edge and no exposed base material → **mild** (single crack < 5 cm, 几何特征 in mild range; surface intact otherwise).
2. A 50 m blade with a 15 cm leading-edge erosion patch exposing glass fibre, defect area ≈ 2% of chord² → **moderate** (length 5–30 cm and 1%–5% area; visible substrate matches moderate visual cue).
3. A blade with a through-thickness lightning puncture plus two separate trailing-edge cracks each > 40 cm → **severe** (multiple defects and a single defect > 30 cm; perforation is a severe visual cue).

> Calibration note: when host-provided labels are available, these thresholds (1%, 5%, 5 cm, 30 cm) will be re-fit per fleet to maximise agreement with operator severity calls.
