# rust-ui Framework

A batteries-included Rust UI framework built on Dioxus. Similar to what Next.js did for React.

## Vision

```
Next.js : React  =  rust-ui : Dioxus
```

| Next.js Provides | rust-ui Provides |
|------------------|------------------|
| File-based routing | Convention-based routing |
| Built-in components | Pre-built UI components |
| API routes | Backend integration |
| Image optimization | Asset optimization |
| SSR/SSG | SSR/SSG support |
| Layouts | Layout system |
| Middleware | Middleware hooks |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         rust-ui                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Components  │ │   Layouts    │ │   Hooks      │            │
│  │  Library     │ │   System     │ │   & Utils    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Router     │ │   Theming    │ │   Forms      │            │
│  │   (enhanced) │ │   Engine     │ │   System     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   SSR/SSG    │ │   API Layer  │ │   CLI        │            │
│  │   Support    │ │   Integration│ │   Tooling    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                         Dioxus                                  │
├─────────────────────────────────────────────────────────────────┤
│              Web (WASM) │ Desktop │ Mobile │ SSR               │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rust-ui/
├── Cargo.toml
├── crates/
│   ├── rust-ui/                 # Core framework
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── prelude.rs
│   │
│   ├── rust-ui-components/      # UI Component library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── button.rs
│   │       ├── input.rs
│   │       ├── select.rs
│   │       ├── modal.rs
│   │       ├── table.rs
│   │       ├── card.rs
│   │       ├── tabs.rs
│   │       ├── toast.rs
│   │       └── ...
│   │
│   ├── rust-ui-layouts/         # Layout system
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── app_shell.rs
│   │       ├── sidebar.rs
│   │       ├── navbar.rs
│   │       ├── grid.rs
│   │       └── stack.rs
│   │
│   ├── rust-ui-forms/           # Form handling
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── form.rs
│   │       ├── validation.rs
│   │       └── fields.rs
│   │
│   ├── rust-ui-hooks/           # Custom hooks
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── use_fetch.rs
│   │       ├── use_storage.rs
│   │       ├── use_debounce.rs
│   │       ├── use_clipboard.rs
│   │       └── use_media_query.rs
│   │
│   ├── rust-ui-theme/           # Theming engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── provider.rs
│   │       ├── tokens.rs
│   │       └── presets/
│   │           ├── dark.rs
│   │           └── light.rs
│   │
│   ├── rust-ui-icons/           # Icon library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── icons/
│   │
│   └── rust-ui-cli/             # CLI tooling
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── new.rs
│           ├── dev.rs
│           └── build.rs
│
├── examples/
│   ├── dashboard/
│   ├── ecommerce/
│   └── admin-panel/
│
└── docs/
```

## Component Library

### Button

```rust
use rust_ui::prelude::*;

// Variants
rsx! {
    Button { "Default" }
    Button { variant: Variant::Primary, "Primary" }
    Button { variant: Variant::Secondary, "Secondary" }
    Button { variant: Variant::Danger, "Delete" }
    Button { variant: Variant::Ghost, "Ghost" }
}

// Sizes
rsx! {
    Button { size: Size::Sm, "Small" }
    Button { size: Size::Md, "Medium" }
    Button { size: Size::Lg, "Large" }
}

// With icons
rsx! {
    Button {
        icon: Icon::Plus,
        "Add Item"
    }
    Button {
        icon: Icon::Download,
        icon_position: IconPosition::Right,
        "Export"
    }
}

// Loading state
rsx! {
    Button { loading: true, "Saving..." }
}
```

### Input

```rust
rsx! {
    Input {
        label: "Email",
        placeholder: "you@example.com",
        value: email,
        on_change: move |v| email.set(v),
    }

    Input {
        label: "Password",
        input_type: InputType::Password,
        value: password,
        on_change: move |v| password.set(v),
        error: "Password must be at least 8 characters",
    }

    Input {
        label: "Search",
        icon: Icon::Search,
        clearable: true,
    }
}
```

### Select

```rust
rsx! {
    Select {
        label: "Region",
        value: region,
        on_change: move |v| region.set(v),
        options: vec![
            ("us-east-1", "US East"),
            ("us-west-2", "US West"),
            ("eu-west-1", "EU Ireland"),
        ],
    }

    Select {
        label: "Services",
        multiple: true,
        searchable: true,
        options: services,
    }
}
```

### Table

```rust
rsx! {
    Table {
        columns: vec![
            Column::new("name", "Name").sortable(),
            Column::new("status", "Status"),
            Column::new("port", "Port"),
            Column::new("actions", "Actions"),
        ],
        data: services,
        row_render: move |row| rsx! {
            Td { "{row.name}" }
            Td { Badge { variant: row.status_variant(), "{row.status}" } }
            Td { "{row.port}" }
            Td {
                IconButton { icon: Icon::Edit }
                IconButton { icon: Icon::Trash, variant: Variant::Danger }
            }
        },
        sortable: true,
        paginated: true,
        page_size: 10,
    }
}
```

### Modal

```rust
rsx! {
    Button { onclick: move |_| show_modal.set(true), "Open Modal" }

    Modal {
        open: show_modal,
        on_close: move |_| show_modal.set(false),
        title: "Confirm Action",

        p { "Are you sure you want to delete this item?" }

        ModalFooter {
            Button { variant: Variant::Ghost, onclick: close, "Cancel" }
            Button { variant: Variant::Danger, onclick: delete, "Delete" }
        }
    }
}
```

### Toast Notifications

```rust
rsx! {
    ToastProvider {
        App {}
    }
}

// In component
fn MyComponent() -> Element {
    let toast = use_toast();

    let save = move |_| {
        toast.success("Item saved successfully");
        // or
        toast.error("Failed to save item");
        toast.warning("Session expiring soon");
        toast.info("New update available");
    };

    rsx! { Button { onclick: save, "Save" } }
}
```

### Card

```rust
rsx! {
    Card {
        CardHeader {
            CardTitle { "Service Status" }
            CardDescription { "Current status of all services" }
        }
        CardContent {
            ServiceList { services: services }
        }
        CardFooter {
            Button { "Refresh" }
        }
    }
}
```

### Tabs

```rust
rsx! {
    Tabs { default_value: "overview",
        TabList {
            Tab { value: "overview", "Overview" }
            Tab { value: "logs", "Logs" }
            Tab { value: "settings", "Settings" }
        }
        TabPanel { value: "overview", OverviewContent {} }
        TabPanel { value: "logs", LogsContent {} }
        TabPanel { value: "settings", SettingsContent {} }
    }
}
```

## Layout System

### AppShell

```rust
rsx! {
    AppShell {
        navbar: rsx! {
            Navbar {
                Logo { "MyApp" }
                NavLinks {
                    NavLink { to: "/", "Home" }
                    NavLink { to: "/dashboard", "Dashboard" }
                }
                UserMenu { user: current_user }
            }
        },
        sidebar: rsx! {
            Sidebar {
                SidebarSection { title: "Main",
                    SidebarLink { icon: Icon::Home, to: "/", "Dashboard" }
                    SidebarLink { icon: Icon::Folder, to: "/explorer", "Explorer" }
                }
                SidebarSection { title: "Settings",
                    SidebarLink { icon: Icon::Settings, to: "/settings", "Settings" }
                }
            }
        },

        // Main content
        Outlet {}
    }
}
```

### Grid & Stack

```rust
rsx! {
    // Responsive grid
    Grid { cols: 3, gap: 4,
        Card { "Card 1" }
        Card { "Card 2" }
        Card { "Card 3" }
    }

    // Responsive: 1 col on mobile, 2 on tablet, 3 on desktop
    Grid { cols: (1, 2, 3), gap: 4,
        // ...
    }

    // Vertical stack
    VStack { gap: 4, align: Align::Start,
        Heading { "Title" }
        Text { "Description" }
        Button { "Action" }
    }

    // Horizontal stack
    HStack { gap: 2, justify: Justify::SpaceBetween,
        Logo {}
        NavLinks {}
        UserMenu {}
    }
}
```

## Theming

### Theme Provider

```rust
use rust_ui::theme::*;

fn main() {
    dioxus::launch(|| rsx! {
        ThemeProvider { theme: Theme::dark(),
            App {}
        }
    });
}
```

### Custom Theme

```rust
let custom_theme = Theme::builder()
    .colors(Colors {
        primary: "#3b82f6",
        secondary: "#64748b",
        success: "#22c55e",
        warning: "#f59e0b",
        danger: "#ef4444",
        background: "#0f172a",
        surface: "#1e293b",
        text: "#f8fafc",
    })
    .radius(Radius::Md)
    .font_family("Inter, sans-serif")
    .build();

rsx! {
    ThemeProvider { theme: custom_theme,
        App {}
    }
}
```

### Using Theme

```rust
fn MyComponent() -> Element {
    let theme = use_theme();

    rsx! {
        div {
            style: "background: {theme.colors.surface}; color: {theme.colors.text}",
            "Themed content"
        }
    }
}
```

## Hooks

### use_fetch

```rust
fn UserList() -> Element {
    let users = use_fetch::<Vec<User>>("/api/users");

    match &*users.read() {
        FetchState::Loading => rsx! { Spinner {} },
        FetchState::Error(e) => rsx! { Alert { variant: Variant::Danger, "{e}" } },
        FetchState::Success(data) => rsx! {
            for user in data {
                UserCard { user: user.clone() }
            }
        },
    }
}
```

### use_form

```rust
fn LoginForm() -> Element {
    let form = use_form(LoginData::default())
        .validate(|data| {
            let mut errors = Errors::new();
            if data.email.is_empty() {
                errors.add("email", "Email is required");
            }
            if data.password.len() < 8 {
                errors.add("password", "Password must be at least 8 characters");
            }
            errors
        });

    let submit = move |_| {
        if form.validate() {
            // Submit form
        }
    };

    rsx! {
        Form { on_submit: submit,
            FormField {
                label: "Email",
                error: form.error("email"),
                Input {
                    value: form.value("email"),
                    on_change: form.set("email"),
                }
            }
            FormField {
                label: "Password",
                error: form.error("password"),
                Input {
                    input_type: InputType::Password,
                    value: form.value("password"),
                    on_change: form.set("password"),
                }
            }
            Button { r#type: "submit", "Login" }
        }
    }
}
```

### use_storage

```rust
fn Settings() -> Element {
    // Persists to localStorage
    let theme = use_local_storage("theme", "dark");
    let token = use_session_storage("auth_token", "");

    rsx! {
        Select {
            value: theme,
            on_change: move |v| theme.set(v),
            options: vec![("dark", "Dark"), ("light", "Light")],
        }
    }
}
```

### use_media_query

```rust
fn ResponsiveComponent() -> Element {
    let is_mobile = use_media_query("(max-width: 768px)");
    let is_dark = use_media_query("(prefers-color-scheme: dark)");

    if *is_mobile.read() {
        rsx! { MobileLayout {} }
    } else {
        rsx! { DesktopLayout {} }
    }
}
```

## CLI

```bash
# Create new project
rust-ui new my-app
rust-ui new my-app --template dashboard
rust-ui new my-app --template admin

# Development
rust-ui dev                    # Start dev server
rust-ui dev --port 3000        # Custom port

# Build
rust-ui build                  # Production build
rust-ui build --platform web   # Web only
rust-ui build --platform desktop

# Generate
rust-ui generate component Button
rust-ui generate page Dashboard
rust-ui generate layout AdminLayout
```

## Templates

### Dashboard Template

```bash
rust-ui new my-dashboard --template dashboard
```

Includes:
- AppShell with sidebar
- Dashboard page with stats cards
- Data table with pagination
- Charts integration
- Settings page
- Dark/light theme toggle

### Admin Template

```bash
rust-ui new my-admin --template admin
```

Includes:
- Everything in dashboard
- User management
- Role-based access
- CRUD generators
- Form builders

## Comparison

| Feature | rust-ui | Next.js | ShadCN |
|---------|---------|---------|--------|
| Language | Rust | JS/TS | JS/TS |
| Components | Built-in | None | Copy-paste |
| Theming | Built-in | None | Built-in |
| Forms | Built-in | None | None |
| SSR | Built-in | Built-in | N/A |
| Desktop | Built-in | Electron | None |
| Type Safety | Compile-time | Runtime | Runtime |

## Roadmap

### Phase 1: Core Components
- [ ] Button, Input, Select, Checkbox, Radio
- [ ] Card, Modal, Drawer, Popover
- [ ] Table, List, Pagination
- [ ] Tabs, Accordion, Menu

### Phase 2: Layout & Navigation
- [ ] AppShell, Sidebar, Navbar
- [ ] Grid, Stack, Container
- [ ] Breadcrumbs, Stepper

### Phase 3: Data Display
- [ ] Charts (via integration)
- [ ] Data tables with sorting/filtering
- [ ] Tree view, Timeline

### Phase 4: Advanced
- [ ] Form builder
- [ ] CLI scaffolding
- [ ] SSR optimization
- [ ] Component playground
