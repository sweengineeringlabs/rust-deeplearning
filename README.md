# rust-ui

Full-stack Rust UI framework with SPI-based architecture.

## Quick Start

```rust
use components::prelude::*;

fn App() -> Element {
    rsx! {
        Button { variant: Variant::Primary, "Click Me" }
        Input { placeholder: "Enter text..." }
    }
}
```

## Documentation

See **[doc/overview.md](doc/overview.md)** for full documentation.

## License

MIT OR Apache-2.0
