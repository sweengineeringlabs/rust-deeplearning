# SWE-Cloud UI Design

Full-stack Rust UI for CloudKit SDK and CloudEmu using the SPI-based architecture.

## Overview

Desktop and web UI for managing local cloud emulation and multi-cloud resources.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SWE-Cloud UI                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   CloudEmu      │  │   CloudKit      │  │   Multi-Cloud   │  │
│  │   Dashboard     │  │   Explorer      │  │   Manager       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Target Users

| User | Use Case |
|------|----------|
| **Developer** | Local development with CloudEmu |
| **DevOps** | Multi-cloud resource management |
| **QA** | Testing against emulated services |

## UI Features

### 1. CloudEmu Dashboard

Local cloud emulator control panel.

```
┌─────────────────────────────────────────────────────────────────┐
│  CloudEmu                                        [Start] [Stop] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Services                          Status        Port           │
│  ├── S3 (Object Storage)           🟢 Running    4566          │
│  ├── DynamoDB (Key-Value)          🟢 Running    4567          │
│  ├── SQS (Message Queue)           🟡 Starting   4568          │
│  ├── SNS (Pub/Sub)                 ⚫ Stopped    -             │
│  └── Lambda (Functions)            ⚫ Stopped    -             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Logs                                              [Clear]  ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  12:03:45 [S3] Bucket 'test-bucket' created                ││
│  │  12:03:46 [S3] PUT object 'data.json' (1.2 KB)             ││
│  │  12:03:47 [DynamoDB] Table 'users' created                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2. Resource Explorer

Browse and manage cloud resources.

```
┌─────────────────────────────────────────────────────────────────┐
│  Resource Explorer                    [CloudEmu ▼] [Refresh]    │
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                              │
│  📁 S3           │  Bucket: my-app-data                        │
│  ├── my-app-data │  ─────────────────────────────────────────  │
│  └── backups     │                                              │
│                  │  Objects (23)              Size    Modified  │
│  📊 DynamoDB     │  ├── config.json          1.2 KB  2 min ago │
│  ├── users       │  ├── users/               -       1 hr ago  │
│  └── sessions    │  │   ├── user-001.json   512 B   1 hr ago  │
│                  │  │   └── user-002.json   489 B   1 hr ago  │
│  📨 SQS          │  └── logs/                -       5 min ago │
│  └── task-queue  │                                              │
│                  │  [Upload] [Create Folder] [Delete]           │
│  📢 SNS          │                                              │
│  └── alerts      │                                              │
│                  │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

### 3. Multi-Cloud Manager

Configure and switch between cloud providers.

```
┌─────────────────────────────────────────────────────────────────┐
│  Cloud Providers                                    [+ Add]     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  🟢 CloudEmu (Local)                          [Active]      ││
│  │  Endpoint: http://localhost:4566                            ││
│  │  Services: S3, DynamoDB, SQS, SNS, Lambda                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ⚪ AWS (Production)                          [Connect]     ││
│  │  Region: us-east-1                                          ││
│  │  Profile: default                                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ⚪ Azure (Staging)                           [Connect]     ││
│  │  Subscription: my-subscription                              ││
│  │  Resource Group: staging-rg                                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Request Inspector

Debug and inspect API calls.

```
┌─────────────────────────────────────────────────────────────────┐
│  Request Inspector                              [Record 🔴]     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time      Service   Operation        Status   Duration         │
│  ───────────────────────────────────────────────────────────── │
│  12:05:01  S3        PutObject        200      45ms            │
│  12:05:02  DynamoDB  GetItem          200      12ms            │
│  12:05:03  SQS       SendMessage      200      23ms            │
│  12:05:04  S3        GetObject        404      8ms    ⚠️       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Request Details                                            ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Service: S3                                                ││
│  │  Operation: GetObject                                       ││
│  │  Bucket: my-app-data                                        ││
│  │  Key: missing-file.txt                                      ││
│  │                                                             ││
│  │  Response: 404 Not Found                                    ││
│  │  { "error": "NoSuchKey", "message": "Object not found" }    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Integration

### Project Structure

```
swe-cloud/
├── crates/
│   ├── cloudemu/                # Existing emulator
│   ├── cloudkit/                # Existing SDK
│   ├── cloudkit-aws/            # Existing AWS provider
│   ├── cloudkit-azure/          # Existing Azure provider
│   ├── cloudkit-gcp/            # Existing GCP provider
│   ├── cloudkit-oracle/         # Existing Oracle provider
│   │
│   │ # NEW: UI crates
│   ├── cloudui-core/            # UI domain logic
│   │   └── src/
│   │       ├── ports/           # SPI traits
│   │       ├── services/        # Business logic
│   │       └── models/          # UI models
│   │
│   ├── cloudui-tauri/           # Tauri provider
│   │   └── src/
│   │       ├── commands.rs      # Tauri IPC commands
│   │       └── adapter.rs       # impl UiPort for Tauri
│   │
│   ├── cloudui-axum/            # Web provider
│   │   └── src/
│   │       ├── routes.rs        # REST API
│   │       └── adapter.rs       # impl UiPort for Axum
│   │
│   └── cloudui-iced/            # Native GUI provider
│       └── src/
│           ├── views/           # Iced views
│           └── adapter.rs       # impl UiPort for Iced
│
├── apps/
│   ├── cloudemu-desktop/        # Tauri app
│   ├── cloudemu-web/            # Axum web server
│   └── cloudemu-native/         # Iced native app
│
└── frontend/
    └── src/                     # Dioxus/Leptos WASM UI
        ├── components/
        │   ├── dashboard.rs
        │   ├── explorer.rs
        │   ├── provider_list.rs
        │   └── inspector.rs
        └── pages/
            ├── home.rs
            ├── resources.rs
            └── settings.rs
```

### SPI Ports for UI

```rust
// cloudui-core/src/ports/mod.rs

/// Emulator control operations
#[async_trait]
pub trait EmulatorPort: Send + Sync {
    async fn start_service(&self, service: ServiceType) -> Result<(), UiError>;
    async fn stop_service(&self, service: ServiceType) -> Result<(), UiError>;
    async fn get_status(&self) -> Result<EmulatorStatus, UiError>;
    async fn get_logs(&self, limit: usize) -> Result<Vec<LogEntry>, UiError>;
}

/// Resource browsing operations
#[async_trait]
pub trait ResourcePort: Send + Sync {
    async fn list_buckets(&self) -> Result<Vec<Bucket>, UiError>;
    async fn list_objects(&self, bucket: &str, prefix: &str) -> Result<Vec<Object>, UiError>;
    async fn get_object(&self, bucket: &str, key: &str) -> Result<ObjectData, UiError>;
    async fn put_object(&self, bucket: &str, key: &str, data: &[u8]) -> Result<(), UiError>;
    async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), UiError>;
}

/// Provider management operations
#[async_trait]
pub trait ProviderPort: Send + Sync {
    async fn list_providers(&self) -> Result<Vec<CloudProvider>, UiError>;
    async fn connect(&self, provider_id: &str) -> Result<(), UiError>;
    async fn disconnect(&self, provider_id: &str) -> Result<(), UiError>;
    async fn add_provider(&self, config: ProviderConfig) -> Result<(), UiError>;
}

/// Request inspection operations
#[async_trait]
pub trait InspectorPort: Send + Sync {
    async fn get_requests(&self, filter: RequestFilter) -> Result<Vec<ApiRequest>, UiError>;
    async fn get_request_detail(&self, id: &str) -> Result<RequestDetail, UiError>;
    async fn start_recording(&self) -> Result<(), UiError>;
    async fn stop_recording(&self) -> Result<(), UiError>;
}
```

### Shared Types with CloudKit

```rust
// cloudui-core/src/models/mod.rs
use cloudkit::prelude::*;  // Reuse CloudKit types

#[derive(Clone, Serialize, Deserialize)]
pub struct EmulatorStatus {
    pub running: bool,
    pub services: Vec<ServiceStatus>,
    pub uptime: Duration,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub service_type: ServiceType,
    pub state: ServiceState,
    pub port: Option<u16>,
    pub request_count: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ServiceState {
    Running,
    Starting,
    Stopping,
    Stopped,
    Error(String),
}
```

### Tauri Integration

```rust
// cloudui-tauri/src/commands.rs
use cloudemu::Emulator;
use cloudui_core::ports::*;

#[tauri::command]
async fn start_service(
    state: State<'_, AppState>,
    service: ServiceType,
) -> Result<(), String> {
    state.emulator
        .start_service(service)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_buckets(
    state: State<'_, AppState>,
) -> Result<Vec<Bucket>, String> {
    state.resources
        .list_buckets()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_emulator_status(
    state: State<'_, AppState>,
) -> Result<EmulatorStatus, String> {
    state.emulator
        .get_status()
        .await
        .map_err(|e| e.to_string())
}
```

### Frontend Component (Dioxus)

```rust
// frontend/src/components/dashboard.rs
use dioxus::prelude::*;
use cloudui_core::models::*;

#[component]
pub fn Dashboard() -> Element {
    let status = use_resource(|| async {
        invoke::<_, EmulatorStatus>("get_emulator_status", ()).await
    });

    rsx! {
        div { class: "dashboard",
            h1 { "CloudEmu" }

            div { class: "controls",
                button { onclick: |_| start_all(), "Start All" }
                button { onclick: |_| stop_all(), "Stop All" }
            }

            match &*status.read() {
                Some(Ok(s)) => rsx! { ServiceList { services: s.services.clone() } },
                Some(Err(e)) => rsx! { p { "Error: {e}" } },
                None => rsx! { p { "Loading..." } }
            }
        }
    }
}

#[component]
fn ServiceList(services: Vec<ServiceStatus>) -> Element {
    rsx! {
        table { class: "service-table",
            thead {
                tr {
                    th { "Service" }
                    th { "Status" }
                    th { "Port" }
                    th { "Actions" }
                }
            }
            tbody {
                for service in services {
                    ServiceRow { service }
                }
            }
        }
    }
}

#[component]
fn ServiceRow(service: ServiceStatus) -> Element {
    let status_class = match service.state {
        ServiceState::Running => "status-running",
        ServiceState::Starting => "status-starting",
        ServiceState::Stopped => "status-stopped",
        ServiceState::Error(_) => "status-error",
        _ => "status-unknown",
    };

    rsx! {
        tr {
            td { "{service.service_type:?}" }
            td { class: status_class, "{service.state:?}" }
            td {
                if let Some(port) = service.port {
                    "{port}"
                } else {
                    "-"
                }
            }
            td {
                match service.state {
                    ServiceState::Running => rsx! {
                        button { onclick: move |_| stop(service.service_type), "Stop" }
                    },
                    ServiceState::Stopped => rsx! {
                        button { onclick: move |_| start(service.service_type), "Start" }
                    },
                    _ => rsx! { span { "..." } }
                }
            }
        }
    }
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  (Dioxus WASM in WebView/Browser, or Iced Native)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
              │   Tauri   │ │  Axum │ │   Iced    │
              │   IPC     │ │  REST │ │  Direct   │
              └─────┬─────┘ └───┬───┘ └─────┬─────┘
                    │           │           │
                    └───────────┼───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    cloudui-core       │
                    │    (Business Logic)   │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        ┌─────▼─────┐     ┌─────▼─────┐    ┌──────▼──────┐
        │ cloudemu  │     │ cloudkit  │    │ cloudkit-*  │
        │ (Local)   │     │ (Core)    │    │ (Providers) │
        └───────────┘     └───────────┘    └─────────────┘
```

## Deployment Targets

| Target | App | UI | Use Case |
|--------|-----|-----|----------|
| **Desktop** | `cloudemu-desktop` | Tauri + Dioxus WASM | Local development |
| **Web** | `cloudemu-web` | Axum + Dioxus WASM | Remote/team access |
| **Native** | `cloudemu-native` | Iced | Performance-critical |
| **CLI** | `cloudemu-cli` | Terminal | Scripts/automation |

## Build Commands

```bash
# Desktop (Tauri)
cd apps/cloudemu-desktop
cargo tauri build

# Web (Axum + WASM)
cd frontend && trunk build --release
cd apps/cloudemu-web && cargo build --release

# Native (Iced)
cd apps/cloudemu-native
cargo build --release

# All platforms
cargo build --workspace --release
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| **UI Framework** | Dioxus (cross-platform) |
| **Desktop Shell** | Tauri 2.0 |
| **Web Server** | Axum |
| **Native GUI** | Iced |
| **Styling** | Tailwind CSS (WASM) / Custom (Iced) |
| **State** | Dioxus Signals |
| **Backend** | CloudEmu + CloudKit |
