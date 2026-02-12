# WebAssembly, Tauri & Full Rust Stack

Research notes on building desktop/web applications with Rust.

---

## What is WASM (WebAssembly)?

**WASM** is a binary instruction format for a stack-based virtual machine.

| Aspect | Description |
|--------|-------------|
| **What** | Low-level bytecode that runs in browsers (and elsewhere) |
| **Speed** | Near-native performance (~1.2x native speed) |
| **Languages** | Compile from Rust, C, C++, Go, AssemblyScript, etc. |
| **Portable** | Runs on any platform with a WASM runtime |
| **Secure** | Sandboxed execution, no direct memory access |

### Use Cases

```
┌─────────────────────────────────────────────────┐
│  Browser                                        │
│  ├── Games (Unity, Unreal)                     │
│  ├── Image/Video editing (Figma, Photoshop)    │
│  ├── CAD applications                          │
│  └── Crypto/compression libraries              │
├─────────────────────────────────────────────────┤
│  Server-side                                    │
│  ├── Edge computing (Cloudflare Workers)       │
│  ├── Plugin systems                            │
│  └── Serverless functions                      │
├─────────────────────────────────────────────────┤
│  Embedded                                       │
│  ├── IoT devices                               │
│  └── Blockchain smart contracts                │
└─────────────────────────────────────────────────┘
```

### Rust → WASM Example

```rust
// Compile with: cargo build --target wasm32-unknown-unknown
#[no_mangle]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

```javascript
// Use in browser
const { add } = await WebAssembly.instantiate(wasmBytes);
console.log(add(2, 3)); // 5
```

### Popular WASM Runtimes

- **Browser**: V8, SpiderMonkey, JavaScriptCore
- **Standalone**: Wasmtime, Wasmer, WasmEdge, WAMR

---

## Is WASM Open Source?

**Yes, fully open source and open standard.**

| Aspect | Details |
|--------|---------|
| **Spec** | W3C standard (like HTML/CSS) |
| **License** | Apache 2.0 |
| **Governance** | W3C WebAssembly Working Group |
| **Development** | GitHub: [webassembly/spec](https://github.com/WebAssembly/spec) |

### Major Open Source Runtimes

| Runtime | License | Maintainer |
|---------|---------|------------|
| **Wasmtime** | Apache 2.0 | Bytecode Alliance |
| **Wasmer** | MIT | Wasmer Inc |
| **WasmEdge** | Apache 2.0 | CNCF |
| **wasm3** | MIT | Community |
| **V8** | BSD | Google |
| **SpiderMonkey** | MPL 2.0 | Mozilla |

No vendor lock-in, no licensing fees.

---

## Can WASM Build a Browser?

**Yes, but with caveats.**

| Approach | Feasible? | Notes |
|----------|-----------|-------|
| **Browser engine in WASM** | Yes | Render HTML/CSS, run JS |
| **Full browser app** | Partial | Needs native shell for OS integration |
| **Browser inside browser** | Yes | Already done (see below) |

### Real Examples

1. **WebVM** - Full x86 Linux in browser via WASM: https://webvm.io
2. **Firefox in WASM** - Mozilla experimented with this
3. **Aspect** - Chromium compiled to WASM (research project)

### Architecture

```
┌─────────────────────────────────────────┐
│  Native Shell (Electron/Tauri/Native)  │  ← Window, networking, GPU
├─────────────────────────────────────────┤
│  WASM Browser Engine                    │  ← HTML parser, CSS, layout
│  ├── Servo (Rust) → compiles to WASM   │
│  ├── Custom engine                      │
│  └── JS engine (wasm3, QuickJS)        │
├─────────────────────────────────────────┤
│  Rendered Output (Canvas/WebGL)         │
└─────────────────────────────────────────┘
```

### Challenges

| Challenge | Why |
|-----------|-----|
| **Networking** | WASM can't open raw sockets (needs host APIs) |
| **File system** | Sandboxed, needs virtual FS or host access |
| **GPU** | Limited to WebGL/WebGPU when in browser |
| **Performance** | DOM manipulation still faster in native JS |
| **Size** | Full engine = 50-100MB+ WASM binary |

---

## WASM with Tauri

**Great combo for lightweight desktop apps.**

### How They Work Together

```
┌─────────────────────────────────────────────────────────┐
│  Tauri App                                              │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐  │
│  │  Frontend (WebView)                               │  │
│  │  ├── HTML/CSS/JS                                  │  │
│  │  ├── WASM modules ← Heavy computation here       │  │
│  │  │   ├── Image processing                        │  │
│  │  │   ├── Crypto                                  │  │
│  │  │   ├── Parsing/Compilation                     │  │
│  │  │   └── Game logic                              │  │
│  │  └── Framework (Yew, Leptos, Dioxus, Sycamore)   │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↕ IPC (invoke)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Rust Backend                                     │  │
│  │  ├── File system access                          │  │
│  │  ├── Native APIs                                 │  │
│  │  ├── System tray, notifications                  │  │
│  │  └── Database, networking                        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Two Approaches

#### 1. Rust Frontend (Full WASM)

```rust
// Using Leptos, Yew, or Dioxus - compiles to WASM
#[component]
fn App() -> impl IntoView {
    let (count, set_count) = create_signal(0);
    view! {
        <button on:click=move |_| set_count.update(|n| *n += 1)>
            "Clicked: " {count}
        </button>
    }
}
```

#### 2. JS Frontend + WASM Modules

```javascript
// Load WASM for heavy lifting
import init, { process_image } from './pkg/my_wasm.js';

await init();
const result = process_image(imageData); // Fast!
```

### Comparison

| Approach | Bundle Size | Performance | DX |
|----------|-------------|-------------|-----|
| **Tauri + JS** | ~3MB | Good | Familiar |
| **Tauri + WASM libs** | ~5MB | Great for compute | Mixed |
| **Tauri + Full Rust UI** | ~4MB | Excellent | All Rust |
| **Electron** | ~150MB | Good | Familiar |

### Popular Rust UI Frameworks for Tauri

| Framework | Style | Maturity |
|-----------|-------|----------|
| **Leptos** | Signals (like SolidJS) | ⭐⭐⭐⭐ |
| **Dioxus** | React-like | ⭐⭐⭐⭐ |
| **Yew** | React-like | ⭐⭐⭐⭐⭐ |
| **Sycamore** | Signals | ⭐⭐⭐ |

### Example: Tauri + Leptos

```toml
# Cargo.toml
[dependencies]
leptos = { version = "0.6", features = ["csr"] }
wasm-bindgen = "0.2"
```

```rust
// src/app.rs
use leptos::*;
use tauri_wasm::api::invoke;

#[component]
fn App() -> impl IntoView {
    let save_file = move |_| {
        spawn_local(async {
            invoke("save_file", &()).await.unwrap();
        });
    };

    view! {
        <button on:click=save_file>"Save"</button>
    }
}
```

### When to Use WASM in Tauri

| Use Case | WASM? | Why |
|----------|-------|-----|
| Simple UI | No | JS/HTML faster to develop |
| Image/video processing | Yes | 10-100x faster than JS |
| Crypto/hashing | Yes | Security + speed |
| Shared logic (web + desktop) | Yes | One codebase |
| Full Rust stack | Yes | Type safety, no JS |
| Complex state management | Yes | Rust's ownership model |

---

## What is "Full Rust"?

**"Full Rust" = Write everything in Rust, no JavaScript.**

### Traditional Web App Stack

```
┌─────────────────────────────────┐
│  Frontend: JavaScript/TypeScript │  ← Different language
│  (React, Vue, Svelte)            │
├─────────────────────────────────┤
│  Backend: Rust                   │  ← Rust
└─────────────────────────────────┘
```

### "Full Rust" Stack

```
┌─────────────────────────────────┐
│  Frontend: Rust → WASM          │  ← Rust
│  (Leptos, Yew, Dioxus)          │
├─────────────────────────────────┤
│  Backend: Rust                   │  ← Rust
└─────────────────────────────────┘
```

### What You Avoid

| No More | Replaced By |
|---------|-------------|
| JavaScript | Rust |
| TypeScript | Rust (already typed) |
| npm/node_modules | Cargo |
| package.json | Cargo.toml |
| React/Vue/Svelte | Leptos/Yew/Dioxus |
| Runtime type errors | Compile-time checks |

### Benefits

```
✓ One language everywhere
✓ Share types between frontend & backend
✓ No "undefined is not a function"
✓ Cargo instead of npm
✓ Smaller bundles (no JS runtime overhead)
✓ Memory safety everywhere
```

### Tradeoffs

```
✗ Steeper learning curve
✗ Slower compile times
✗ Smaller ecosystem than JS
✗ Fewer UI component libraries
✗ Harder to hire developers
```

### Example: Shared Types

```rust
// shared/src/lib.rs - Used by BOTH frontend and backend
#[derive(Serialize, Deserialize, Clone)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

// Frontend (WASM) - Same struct!
let user: User = invoke("get_user", &id).await?;

// Backend (Tauri) - Same struct!
#[tauri::command]
fn get_user(id: u64) -> User {
    db.find_user(id)
}
```

No more mismatched types between frontend and backend.

---

## Summary

| Technology | Best For |
|------------|----------|
| **WASM** | Compute-heavy tasks, portability |
| **Tauri** | Lightweight desktop apps |
| **Full Rust** | Type safety, single-language teams |
| **Tauri + WASM** | Best of both worlds |

### Recommended Stack for New Projects

```
Tauri 2.0 + Leptos + Tailwind CSS
├── Small bundle (~4MB vs Electron's 150MB)
├── Native performance
├── Full Rust (optional)
└── Cross-platform (Windows, macOS, Linux, iOS, Android)
```
