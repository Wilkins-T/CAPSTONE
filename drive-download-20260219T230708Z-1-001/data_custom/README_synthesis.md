# Synthetic Data Report

Generated: 2026-02-24 17:57:36

## Source Data
- Years: 2012 + 2014
- Combined samples: 12000 (benign=6000 malicious=6000)
- Features: 1159

## Generation Targets
- Synthetic samples: 6000 (benign=3000 malicious=3000)
- Output dir: /home/t/Downloads/Code/drive-download-20260219T230708Z-1-001/data_custom

## LLM Policy (raw)
```
Based on the provided data and interpretation rules, I will design a controlled synthetic data generation policy to improve robustness to temporal drift.

**Benign Adjustments**

To reduce overfitting to highly year-specific features, we will decrease the probability of benign-leaning features that are likely to change over time. We will also introduce some malicious-leaning features with increased probabilities to maintain class separability.

```json
"benign_adjustments": [
  {"feature_index": 53, "delta": -0.10}, // restrictedapilist_android.widget.videoview.setvideopath
  {"feature_index": 52, "delta": -0.08}, // restrictedapilist_android.widget.videoview.start
  {"feature_index": 49, "delta": -0.07}, // restrictedapilist_android.widget.videoview.stopplayback
  {"feature_index": 50, "delta": -0.06}, // restrictedapilist_android.widget.videoview.pause
  {"feature_index": 35, "delta": -0.05}, // activitylist_com.google.ads.adactivity
  {"feature_index": 548, "delta": -0.04}, // suspiciousapilist_landroid/support/v4/app/fragmentactivity.getsystemservice
  {"feature_index": 549, "delta": -0.03}, // restrictedapilist_android.support.v4.view.accessibility.accessibilitynodeprovidercompat.performaction
  {"feature_index": 550, "delta": -0.02} // restrictedapilist_android.support.v4.view.accessibility.accessibilitynodeprovidercompatjellybean$accessibilitynodeinfobridge.performaction
],
```

**Malicious Adjustments**

To increase robustness to features likely to drift over time, we will decrease the probability of malicious-leaning features that are highly year-specific. We will also introduce some benign-leaning features with increased probabilities to maintain class separability.

```json
"malicious_adjustments": [
  {"feature_index": 44, "delta": -0.10}, // requestedpermissionlist_android.permission.read_phone_state
  {"feature_index": 4, "delta": -0.08}, // usedpermissionslist_android.permission.read_phone_state
  {"feature_index": 12, "delta": -0.07}, // suspiciousapilist_landroid/telephony/telephonymanager.getdeviceid
  {"feature_index": 26, "delta": -0.06}, // intentfilterlist_android.intent.action.boot_completed
  {"feature_index": 16, "delta": -0.05}, // suspiciousapilist_landroid/telephony/telephonymanager.getsubscriberid
  {"feature_index": 39, "delta": -0.04}, // requestedpermissionlist_android.permission.access_wifi_state
  {"feature_index": 87, "delta": -0.03}, // requestedpermissionlist_android.permission.send_sms
  {"feature_index": 37, "delta": -0.02} // requestedpermissionlist_android.permission.receive_boot_completed
],
```

**Global Flip Probability**

To maintain approximate class separability and avoid collapsing global feature density, we will introduce a small global flip probability.

```json
"global_flip_prob": 0.005,
```

**Novelty Rate**

To introduce rare co-occurring feature patterns, we will set the novelty rate to a moderate value.

```json
"novelty_rate": 0.01,
```

**Notes**

This drift strategy aims to reduce overfitting to highly year-specific features by decreasing the probability of benign-leaning and malicious-leaning features that are likely to change over time. It also introduces some malicious-leaning and benign-leaning features with increased probabilities to maintain class separability.

```json
"notes": "Reducing overfitting to year-specific features while maintaining class separability through controlled synthetic data generation."
```

The final JSON object is:

```json
{
  "benign_adjustments": [
    {"feature_index": 53, "delta": -0.10},
    {"feature_index": 52, "delta": -0.08},
    {"feature_index": 49, "delta": -0.07},
    {"feature_index": 50, "delta": -0.06},
    {"feature_index": 35, "delta": -0.05},
    {"feature_index": 548, "delta": -0.04},
    {"feature_index": 549, "delta": -0.03},
    {"feature_index": 550, "delta": -0.02}
  ],
  "malicious_adjustments": [
    {"feature_index": 44, "delta": -0.10},
    {"feature_index": 4, "delta": -0.08},
    {"feature_index": 12, "delta": -0.07},
    {"feature_index": 26, "delta": -0.06},
    {"feature_index": 16, "delta": -0.05},
    {"feature_index": 39, "delta": -0.04},
    {"feature_index": 87, "delta": -0.03},
    {"feature_index": 37, "delta": -0.02}
  ],
  "global_flip_prob": 0.005,
  "novelty_rate": 0.01,
  "notes": "Reducing overfitting to year-specific features while maintaining class separability through controlled synthetic data generation."
}
```
```

## Parsed Policy
```
{
  "feature_index": 53,
  "delta": -0.1
}
```

## Summary Stats
- benign mean ones: 0.0197
- malicious mean ones: 0.0307
- 2012 class counts: benign=3000 malicious=3000
- 2014 class counts: benign=3000 malicious=3000
- flip_prob: 0.0
- novelty_rate: 0.0
- prob_smoothing: 0.5
- beta_kappa: 15.0
- drift_horizon_min: 1.0
- drift_horizon_max: 2.0
- drift_strength: 0.85
- base_mutation_rate: 0.01
- drift_mutation_scale: 2.0
- max_mutation_rate: 0.3
- ensure_uniqueness: True
