# Full-Stack Rust Architecture

SPI-based design with three interchangeable providers:
- **Tauri** - Desktop app (WebView + WASM UI)
- **Axum** - Web server (Browser + WASM UI)
- **Iced** - Native desktop (Pure Rust GUI, no WebView)

## WASM vs Native: Understanding the Layers

WASM (WebAssembly) does not create desktop applications by itself. It's a compile target that runs inside a WebView. Tauri provides the native desktop shell.

```
┌─────────────────────────────────────┐
│          Tauri Desktop Shell        │  ← Native binary (creates window)
│  ┌───────────────────────────────┐  │
│  │         WebView               │  │  ← Browser engine (WRY/WebKit)
│  │  ┌─────────────────────────┐  │  │
│  │  │   Your WASM UI Code     │  │  │  ← Leptos/Yew/Dioxus compiled
│  │  │   (Rust → WASM)         │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │   Rust Backend (native)       │  │  ← Full system access (FS, OS APIs)
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Technology Roles

| Technology | Role | Creates Desktop? |
|------------|------|------------------|
| **WASM** | Compile target for UI code | No (runs in WebView) |
| **Tauri** | Native shell + system APIs | Yes |
| **Leptos/Yew/Dioxus** | UI framework → compiles to WASM | No (UI only) |
| **WebView (WRY)** | Renders HTML/CSS/WASM | No (rendering engine) |

### Alternative: Pure Native GUI (No WASM)

For apps without WebView, use native Rust GUI frameworks:

| Framework | Rendering | Use Case |
|-----------|-----------|----------|
| `iced` | wgpu/OpenGL | Cross-platform native |
| `egui` | Immediate mode | Tools, debug UIs |
| `slint` | Native + embedded | Desktop + embedded |
| `gpui` | Metal/Vulkan | High-performance (Zed editor) |

```rust
// Example: iced (no WASM, pure native)
iced::application("App", App::update, App::view).run()
```

### Why WASM + Tauri?

| Benefit | Description |
|---------|-------------|
| **Web skills** | Use HTML/CSS for styling |
| **Code sharing** | Same UI runs in browser and desktop |
| **Ecosystem** | Access to web libraries |
| **Hot reload** | Faster UI development cycle |

## Project Structure

```
rust-ui/
├── Cargo.toml
│
├── core/                    # Domain logic (provider-agnostic)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── domain/          # Entities, value objects
│       ├── services/        # Business logic
│       ├── ports/           # SPI traits (interfaces)
│       │   ├── mod.rs
│       │   ├── storage.rs   # trait StoragePort
│       │   ├── auth.rs      # trait AuthPort
│       │   └── api.rs       # trait ApiPort
│       └── usecases/        # Application usecases
│
├── providers/
│   ├── tauri-provider/      # Tauri SPI implementation
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── adapter.rs   # impl ApiPort for TauriAdapter
│   │       └── commands.rs
│   │
│   ├── axum-provider/       # Axum SPI implementation
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── adapter.rs   # impl ApiPort for AxumAdapter
│   │       └── routes.rs
│   │
│   └── iced-provider/       # Iced SPI implementation (native GUI)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── adapter.rs   # impl ApiPort for IcedAdapter
│           └── views.rs     # Native UI components
│
├── frontend-wasm/           # WASM UI (Tauri + Web)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Leptos/Yew/Dioxus components
│       └── client.rs        # Uses ApiPort trait
│
├── frontend-native/         # Native UI (Iced)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Iced components
│       └── client.rs        # Uses ApiPort trait
│
├── app-desktop/             # Tauri binary (WebView)
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/main.rs
│
├── app-native/              # Iced binary (pure native)
│   ├── Cargo.toml
│   └── src/main.rs
│
└── app-web/                 # Axum binary
    ├── Cargo.toml
    └── src/main.rs
```

## SPI Traits (Ports)

### API Port

```rust
// core/src/ports/api.rs
#[async_trait]
pub trait ApiPort: Send + Sync {
    async fn execute<C: Command>(&self, cmd: C) -> Result<C::Output, CoreError>;
    async fn query<Q: Query>(&self, query: Q) -> Result<Q::Output, CoreError>;
}
```

### Storage Port

```rust
// core/src/ports/storage.rs
#[async_trait]
pub trait StoragePort: Send + Sync {
    async fn get<T: Entity>(&self, id: &str) -> Result<T, CoreError>;
    async fn save<T: Entity>(&self, entity: &T) -> Result<(), CoreError>;
    async fn delete<T: Entity>(&self, id: &str) -> Result<(), CoreError>;
}
```

## Provider Implementations

### Tauri Provider

```rust
// providers/tauri-provider/src/adapter.rs
pub struct TauriAdapter {
    app_handle: AppHandle,
}

#[async_trait]
impl ApiPort for TauriAdapter {
    async fn execute<C: Command>(&self, cmd: C) -> Result<C::Output, CoreError> {
        // IPC via Tauri commands
    }
}
```

### Axum Provider

```rust
// providers/axum-provider/src/adapter.rs
pub struct AxumAdapter {
    client: reqwest::Client,
    base_url: String,
}

#[async_trait]
impl ApiPort for AxumAdapter {
    async fn execute<C: Command>(&self, cmd: C) -> Result<C::Output, CoreError> {
        // HTTP calls to Axum server
    }
}
```

### Iced Provider (Native GUI)

```rust
// providers/iced-provider/src/adapter.rs
pub struct IcedAdapter {
    core: Arc<CoreServices>,
}

#[async_trait]
impl ApiPort for IcedAdapter {
    async fn execute<C: Command>(&self, cmd: C) -> Result<C::Output, CoreError> {
        // Direct call to core services (no IPC needed)
        self.core.execute(cmd).await
    }
}
```

```rust
// providers/iced-provider/src/views.rs
use iced::{Element, Task};

pub struct MainView<P: ApiPort> {
    client: AppClient<P>,
    // ... state
}

impl<P: ApiPort> MainView<P> {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::CreateUser(data) => {
                Task::perform(
                    self.client.create_user(data),
                    Message::UserCreated
                )
            }
            // ...
        }
    }

    pub fn view(&self) -> Element<Message> {
        // Pure Rust UI - no HTML/CSS
        column![
            text("Users"),
            button("Add User").on_press(Message::AddUser),
        ].into()
    }
}
```

## Dependency Flow

```
                         ┌─────────────────┐
                         │      core       │
                         │  (ports/traits) │
                         └────────▲────────┘
                                  │ implements
       ┌──────────────┬───────────┼───────────┬──────────────┐
       │              │           │           │              │
┌──────┴─────┐ ┌──────┴─────┐ ┌───┴────┐ ┌────┴─────┐ ┌──────┴─────┐
│   tauri    │ │    axum    │ │  iced  │ │   mock   │ │   test     │
│  provider  │ │  provider  │ │provider│ │ provider │ │  provider  │
└──────┬─────┘ └──────┬─────┘ └───┬────┘ └────┬─────┘ └──────┬─────┘
       │              │           │           │              │
       ▼              ▼           ▼           ▼              ▼
 ┌───────────┐  ┌───────────┐ ┌────────┐ ┌─────────┐  ┌───────────┐
 │app-desktop│  │  app-web  │ │app-    │ │  mocks  │  │   tests   │
 │ (WebView) │  │ (Browser) │ │native  │ │         │  │           │
 └───────────┘  └───────────┘ └────────┘ └─────────┘  └───────────┘
```

## Frontend Client Abstraction

```rust
// frontend/src/client.rs
pub struct AppClient<P: ApiPort> {
    provider: P,
}

impl<P: ApiPort> AppClient<P> {
    pub async fn create_user(&self, data: CreateUser) -> Result<User, CoreError> {
        self.provider.execute(data).await
    }
}

// Usage - same UI code, different provider:
// Desktop: AppClient::new(TauriAdapter::new(handle))
// Web:     AppClient::new(AxumAdapter::new(client, url))
```

## Binary Entry Points

### Desktop (Tauri)

```rust
// app-desktop/src/main.rs
fn main() {
    tauri::Builder::default()
        .manage(AppState::new(CoreServices::new()))
        .invoke_handler(tauri::generate_handler![...])
        .run(tauri::generate_context!())
}
```

### Web (Axum)

```rust
// app-web/src/main.rs
#[tokio::main]
async fn main() {
    let state = AppState::new(CoreServices::new());
    let app = Router::new()
        .merge(api_routes())
        .with_state(state);

    axum::serve(listener, app).await.unwrap();
}
```

### Native Desktop (Iced)

```rust
// app-native/src/main.rs
use iced::Application;

fn main() -> iced::Result {
    let core = Arc::new(CoreServices::new());
    let adapter = IcedAdapter::new(core);
    let client = AppClient::new(adapter);

    iced::application("My App", App::update, App::view)
        .run_with(|| App::new(client))
}
```

## Provider Comparison

| Aspect | Tauri | Axum | Iced |
|--------|-------|------|------|
| **Target** | Desktop | Web | Desktop |
| **UI Rendering** | WebView (HTML/CSS) | Browser | Native (wgpu) |
| **Binary Size** | ~3-10MB | N/A | ~5-15MB |
| **Styling** | CSS/Tailwind | CSS/Tailwind | Rust code |
| **Hot Reload** | Yes (frontend) | Yes | Limited |
| **Web Skills** | Required | Required | Not needed |
| **Performance** | Good | Good | Best |
| **Look & Feel** | Web-like | Web | Native |

## Benefits

| Aspect | Benefit |
|--------|---------|
| **Testability** | Mock provider for unit tests |
| **Portability** | Same core runs on all three platforms |
| **Flexibility** | Add mobile provider later (Tauri Mobile) |
| **Separation** | UI knows nothing about transport |
| **Choice** | WebView (Tauri), Browser (Axum), or Native (Iced) |
| **Shared Logic** | Core business logic written once |

## When to Use Each Provider

| Use Case | Recommended Provider |
|----------|---------------------|
| Web app with desktop version | Tauri + Axum (shared WASM UI) |
| Maximum performance desktop | Iced |
| Embedded/kiosk systems | Iced |
| Rapid prototyping | Tauri (web tooling) |
| Pure web deployment | Axum |
| All three platforms | All providers with shared core |
