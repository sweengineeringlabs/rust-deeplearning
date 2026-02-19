# Feature Spec: Time Series Data Pipeline

**Version:** 1.0
**Status:** Draft
**Section:** 2.7

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-700 | `OHLCVCandle` struct: timestamp, open, high, low, close, volume | Must | Test | — |
| REQ-002 | FR-701 | `TimeSeriesDataset`: windowed access over candle data, configurable window size and prediction horizon | Must | Test | — |
| REQ-003 | FR-702 | Configurable target columns: single, multi, or close (default) | Must | Test | — |
| REQ-004 | FR-703 | `DataLoader` implementing `Iterator` — yields `(input_batch, target_batch)` | Must | Test | — |
| REQ-005 | FR-704 | `DataLoader` shuffle support with epoch reset | Must | Test | — |
| REQ-006 | FR-705 | `FeatureEngineer` with pluggable `Feature` trait | Must | Test | — |
| REQ-007 | FR-706 | Built-in features: log returns, moving average, volatility | Must | Test | — |
| REQ-008 | FR-707 | RSI feature using Wilder's exponential moving average (not SMA) | Must | Test | — |
| REQ-009 | FR-708 | `Scaler` with MinMax, StandardScaler, RobustScaler | Must | Test | — |
| REQ-010 | FR-709 | `Scaler::inverse_transform` for denormalization | Must | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-700): To be defined
- **REQ-002** (FR-701): To be defined
- **REQ-003** (FR-702): To be defined
- **REQ-004** (FR-703): To be defined
- **REQ-005** (FR-704): To be defined
- **REQ-006** (FR-705): To be defined
- **REQ-007** (FR-706): To be defined
- **REQ-008** (FR-707): To be defined
- **REQ-009** (FR-708): To be defined
- **REQ-010** (FR-709): To be defined

