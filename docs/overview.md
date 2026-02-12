# rust-ui Overview

Full-stack Rust UI framework with SPI-based architecture for building cross-platform applications.

---

## What

A component library and architecture pattern for building apps that run on:
- 🖥️ **Desktop** (Tauri + WebView)
- 🌐 **Web** (Axum + Browser)
- 🦀 **Native** (Iced - pure Rust, no WebView)

## Why

- **Write once, run everywhere** - Same components on all platforms
- **Type-safe** - Full Rust type safety across the stack
- **Swappable backends** - SPI architecture for different platforms
- **No JavaScript** - Pure Rust stack

## How

Built on **Dioxus 0.6** with an SPI pattern for provider abstraction.

---

## Components

| Component | Description |
|-----------|-------------|
| `Button` | Clickable button with variants |
| `Input` | Text input field |
| `Select` | Dropdown/combobox |
| `Card` | Container with styling |
| `Modal` | Dialog overlay |
| `Badge` | Status/tag indicator |
| `Spinner` | Loading indicator |
| `Alert` | Notification message |

## Variants & Sizes

```rust
// Variants
Variant::Default | Primary | Secondary | Success | Warning | Danger | Ghost | Link

// Sizes
Size::Xs | Sm | Md | Lg | Xl
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPI-Based Architecture                        │
│                                                                  │
│                       ┌─────────────────┐                        │
│                       │      core       │                        │
│                       │  (ports/traits) │                        │
│                       └────────▲────────┘                        │
│                                │                                  │
│        ┌───────────────┬───────┴───────┬───────────────┐        │
│        │               │               │               │         │
│   ┌────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐  │
│   │  tauri   │   │   axum    │   │   iced    │   │   mock    │  │
│   │ provider │   │  provider │   │  provider │   │  provider │  │
│   └────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘  │
│        │               │               │               │         │
│        ▼               ▼               ▼               ▼         │
│   ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐    │
│   │ Desktop │   │    Web    │   │  Native  │   │   Tests   │    │
│   └─────────┘   └───────────┘   └──────────┘   └───────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Provider Comparison

| Provider | Target | UI Tech | Binary Size |
|----------|--------|---------|-------------|
| **Tauri** | Desktop | WebView (WASM) | ~3-10MB |
| **Axum** | Web | Browser (WASM) | N/A |
| **Iced** | Desktop | Native (wgpu) | ~5-15MB |

## Project Structure

```
rust-ui/
├── crates/
│   └── components/         # UI component library
└── doc/
    ├── 0-ideation/         # Research & ideation
    └── 3-design/           # Architecture & design docs
```

## Related Documents

| Document | Description |
|----------|-------------|
| [3-design/architecture.md](3-design/architecture.md) | Full SPI architecture |
| [3-design/swe-cloud-ui.md](3-design/swe-cloud-ui.md) | CloudEmu UI design |
| [0-ideation/framework-benchmarks.md](0-ideation/framework-benchmarks.md) | Performance comparisons |
