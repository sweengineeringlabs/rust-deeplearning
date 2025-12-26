# rust-ui Icons

SVG icon system for rust-ui framework.

## Architecture

```
rust-ui-icons/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── icon.rs           # Base Icon component
    ├── sets/
    │   ├── mod.rs
    │   ├── lucide.rs     # Lucide icons (default)
    │   ├── heroicons.rs  # Heroicons
    │   ├── phosphor.rs   # Phosphor icons
    │   └── custom.rs     # User-defined icons
    └── generated/        # Auto-generated from SVG files
```

## Usage

### Basic

```rust
use rust_ui::prelude::*;
use rust_ui_icons::*;

rsx! {
    Icon { name: "home" }
    Icon { name: "settings" }
    Icon { name: "user" }
}
```

### With Enum (Type-safe)

```rust
use rust_ui_icons::LucideIcon;

rsx! {
    Icon { icon: LucideIcon::Home }
    Icon { icon: LucideIcon::Settings }
    Icon { icon: LucideIcon::User }
}
```

### Sizes

```rust
rsx! {
    Icon { icon: LucideIcon::Home, size: 16 }  // 16px
    Icon { icon: LucideIcon::Home, size: 24 }  // 24px (default)
    Icon { icon: LucideIcon::Home, size: 32 }  // 32px

    // Or use Size enum
    Icon { icon: LucideIcon::Home, size: Size::Sm }  // 16px
    Icon { icon: LucideIcon::Home, size: Size::Md }  // 24px
    Icon { icon: LucideIcon::Home, size: Size::Lg }  // 32px
    Icon { icon: LucideIcon::Home, size: Size::Xl }  // 48px
}
```

### Colors

```rust
rsx! {
    // Inherit from parent (default)
    Icon { icon: LucideIcon::Check }

    // Custom color
    Icon { icon: LucideIcon::Check, color: "green" }
    Icon { icon: LucideIcon::X, color: "#ef4444" }

    // Theme colors
    Icon { icon: LucideIcon::Check, color: Color::Success }
    Icon { icon: LucideIcon::AlertTriangle, color: Color::Warning }
    Icon { icon: LucideIcon::X, color: Color::Danger }
}
```

### Stroke Width

```rust
rsx! {
    Icon { icon: LucideIcon::Home, stroke_width: 1.0 }  // Thin
    Icon { icon: LucideIcon::Home, stroke_width: 2.0 }  // Default
    Icon { icon: LucideIcon::Home, stroke_width: 3.0 }  // Bold
}
```

### With Components

```rust
rsx! {
    // Button with icon
    Button {
        icon: LucideIcon::Plus,
        "Add Item"
    }

    // Icon-only button
    IconButton { icon: LucideIcon::Settings }
    IconButton { icon: LucideIcon::Trash, variant: Variant::Danger }

    // Input with icon
    Input {
        icon: LucideIcon::Search,
        placeholder: "Search...",
    }

    // Menu items
    MenuItem {
        icon: LucideIcon::User,
        "Profile"
    }

    // Sidebar links
    SidebarLink {
        icon: LucideIcon::Home,
        to: "/",
        "Dashboard"
    }
}
```

## Available Icon Sets

### Lucide (Default) - 1400+ icons

```rust
use rust_ui_icons::lucide::*;

rsx! {
    // Navigation
    Icon { icon: Home }
    Icon { icon: Menu }
    Icon { icon: ChevronLeft }
    Icon { icon: ChevronRight }

    // Actions
    Icon { icon: Plus }
    Icon { icon: Minus }
    Icon { icon: Edit }
    Icon { icon: Trash }
    Icon { icon: Download }
    Icon { icon: Upload }
    Icon { icon: Copy }
    Icon { icon: Check }
    Icon { icon: X }

    // Status
    Icon { icon: AlertCircle }
    Icon { icon: AlertTriangle }
    Icon { icon: CheckCircle }
    Icon { icon: XCircle }
    Icon { icon: Info }

    // Objects
    Icon { icon: File }
    Icon { icon: Folder }
    Icon { icon: Image }
    Icon { icon: Database }
    Icon { icon: Server }
    Icon { icon: Cloud }

    // Social
    Icon { icon: Github }
    Icon { icon: Twitter }
    Icon { icon: Linkedin }
}
```

### Heroicons - 300+ icons

```rust
use rust_ui_icons::heroicons::*;

rsx! {
    Icon { icon: heroicons::Home }
    Icon { icon: heroicons::Cog }
}
```

### Phosphor - 1200+ icons

```rust
use rust_ui_icons::phosphor::*;

rsx! {
    Icon { icon: phosphor::House }
    Icon { icon: phosphor::Gear }
}
```

## Icon Component Implementation

```rust
// src/icon.rs
use dioxus::prelude::*;

#[derive(Clone, PartialEq)]
pub enum Size {
    Xs,  // 12px
    Sm,  // 16px
    Md,  // 24px
    Lg,  // 32px
    Xl,  // 48px
    Custom(u32),
}

impl Size {
    pub fn to_px(&self) -> u32 {
        match self {
            Size::Xs => 12,
            Size::Sm => 16,
            Size::Md => 24,
            Size::Lg => 32,
            Size::Xl => 48,
            Size::Custom(px) => *px,
        }
    }
}

#[derive(Props, Clone, PartialEq)]
pub struct IconProps {
    /// Icon to render (enum or string)
    #[props(into)]
    pub icon: IconType,

    /// Size in pixels or Size enum
    #[props(default = Size::Md)]
    pub size: Size,

    /// Color (CSS color or Color enum)
    #[props(default)]
    pub color: Option<String>,

    /// Stroke width for outline icons
    #[props(default = 2.0)]
    pub stroke_width: f32,

    /// Additional CSS class
    #[props(default)]
    pub class: Option<String>,

    /// Accessibility label
    #[props(default)]
    pub aria_label: Option<String>,
}

#[component]
pub fn Icon(props: IconProps) -> Element {
    let size_px = props.size.to_px();
    let svg_content = props.icon.get_svg();

    let style = format!(
        "width: {}px; height: {}px; {}",
        size_px,
        size_px,
        props.color.as_ref().map(|c| format!("color: {}", c)).unwrap_or_default()
    );

    rsx! {
        span {
            class: "rust-ui-icon {props.class.unwrap_or_default()}",
            style: "{style}",
            aria_label: props.aria_label,
            dangerous_inner_html: "{svg_content}"
        }
    }
}
```

## Icon Type Trait

```rust
// src/lib.rs
pub trait IconType: Clone + PartialEq {
    fn get_svg(&self) -> &'static str;
    fn name(&self) -> &'static str;
}

// Auto-implemented for each icon set
impl IconType for LucideIcon {
    fn get_svg(&self) -> &'static str {
        match self {
            LucideIcon::Home => include_str!("icons/lucide/home.svg"),
            LucideIcon::Settings => include_str!("icons/lucide/settings.svg"),
            // ... generated for all icons
        }
    }

    fn name(&self) -> &'static str {
        match self {
            LucideIcon::Home => "home",
            LucideIcon::Settings => "settings",
            // ...
        }
    }
}
```

## Generated Icon Enum

```rust
// src/sets/lucide.rs (auto-generated)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum LucideIcon {
    // A
    Activity,
    Airplay,
    AlertCircle,
    AlertTriangle,
    AlignCenter,
    AlignJustify,
    AlignLeft,
    AlignRight,
    Anchor,
    Archive,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    // B
    Bookmark,
    Box,
    Briefcase,
    // C
    Calendar,
    Camera,
    Check,
    CheckCircle,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    ChevronUp,
    Circle,
    Clipboard,
    Clock,
    Cloud,
    Code,
    Copy,
    // D
    Database,
    Delete,
    Download,
    // E
    Edit,
    ExternalLink,
    Eye,
    EyeOff,
    // F
    File,
    FileText,
    Filter,
    Flag,
    Folder,
    // G
    Gift,
    Github,
    Globe,
    Grid,
    // H
    HardDrive,
    Hash,
    Heart,
    Home,
    // I
    Image,
    Inbox,
    Info,
    // K
    Key,
    // L
    Layers,
    Layout,
    Link,
    List,
    Lock,
    LogIn,
    LogOut,
    // M
    Mail,
    Map,
    Menu,
    MessageCircle,
    Minus,
    Monitor,
    Moon,
    MoreHorizontal,
    MoreVertical,
    // N
    Navigation,
    // O
    Octagon,
    // P
    Package,
    Paperclip,
    Pause,
    PenTool,
    Phone,
    Play,
    Plus,
    Power,
    Printer,
    // R
    Radio,
    RefreshCw,
    Repeat,
    RotateCw,
    // S
    Save,
    Search,
    Send,
    Server,
    Settings,
    Share,
    Shield,
    ShoppingCart,
    Sidebar,
    Slash,
    Sliders,
    Smartphone,
    Star,
    Sun,
    // T
    Table,
    Tag,
    Terminal,
    Trash,
    TrendingUp,
    Triangle,
    // U
    Unlock,
    Upload,
    User,
    Users,
    // V
    Video,
    // W
    Wifi,
    // X
    X,
    XCircle,
    // Z
    ZoomIn,
    ZoomOut,
    // ... 1400+ total
}
```

## Build Script (Generate from SVG)

```rust
// build.rs
use std::fs;
use std::path::Path;

fn main() {
    let icons_dir = Path::new("icons/lucide");
    let output = Path::new("src/generated/lucide.rs");

    let mut enum_variants = Vec::new();
    let mut match_arms = Vec::new();

    for entry in fs::read_dir(icons_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().map(|e| e == "svg").unwrap_or(false) {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let variant = to_pascal_case(name);

            enum_variants.push(variant.clone());
            match_arms.push(format!(
                "LucideIcon::{} => include_str!(\"../icons/lucide/{}.svg\")",
                variant, name
            ));
        }
    }

    // Generate Rust code
    let code = format!(
        r#"
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum LucideIcon {{
    {}
}}

impl IconType for LucideIcon {{
    fn get_svg(&self) -> &'static str {{
        match self {{
            {}
        }}
    }}
}}
"#,
        enum_variants.join(",\n    "),
        match_arms.join(",\n            ")
    );

    fs::write(output, code).unwrap();
}

fn to_pascal_case(s: &str) -> String {
    s.split('-')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}
```

## CSS

```css
/* Included in rust-ui base styles */
.rust-ui-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.rust-ui-icon svg {
    width: 100%;
    height: 100%;
    stroke: currentColor;
    fill: none;
}

/* Filled variant */
.rust-ui-icon.filled svg {
    fill: currentColor;
    stroke: none;
}
```

## CloudEmu Icons Example

```rust
use rust_ui_icons::lucide::*;

fn ServiceList(services: Vec<Service>) -> Element {
    rsx! {
        for service in services {
            div { class: "service-row",
                Icon {
                    icon: service_icon(&service.name),
                    size: Size::Md,
                    color: status_color(&service.status),
                }
                span { "{service.name}" }
            }
        }
    }
}

fn service_icon(name: &str) -> LucideIcon {
    match name {
        "S3" => LucideIcon::Database,
        "DynamoDB" => LucideIcon::Table,
        "SQS" => LucideIcon::MessageCircle,
        "SNS" => LucideIcon::Radio,
        "Lambda" => LucideIcon::Zap,
        _ => LucideIcon::Cloud,
    }
}

fn status_color(status: &ServiceStatus) -> &'static str {
    match status {
        ServiceStatus::Running => "#22c55e",
        ServiceStatus::Starting => "#f59e0b",
        ServiceStatus::Stopped => "#64748b",
        ServiceStatus::Error(_) => "#ef4444",
    }
}
```
