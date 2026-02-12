//! Icon component with built-in SVG icons

use dioxus::prelude::*;
use crate::api::types::Size;

/// Built-in icon names
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IconName {
    // Actions
    Check,
    X,
    Plus,
    Minus,
    Edit,
    Trash,
    Copy,
    Save,
    Download,
    Upload,
    Print,
    Share,
    Undo,
    Redo,

    // Navigation
    ChevronLeft,
    ChevronRight,
    ChevronUp,
    ChevronDown,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Menu,
    Home,
    MoreHorizontal,
    MoreVertical,

    // Status
    Info,
    Warning,
    Error,
    Success,
    Question,

    // Objects
    User,
    Users,
    Settings,
    Search,
    Filter,
    Calendar,
    Clock,
    Mail,
    Phone,
    Link,
    File,
    FileText,
    Folder,
    FolderOpen,
    Image,
    Clipboard,
    ClipboardCheck,
    Notepad,
    Book,
    BookOpen,
    Bookmark,
    Tag,
    Tags,

    // Media
    Play,
    Pause,
    Stop,
    SkipBack,
    SkipForward,
    Volume,
    VolumeOff,
    Mic,
    MicOff,
    Camera,
    Video,

    // Communication
    MessageSquare,
    MessageCircle,
    Send,
    Inbox,

    // Misc
    Eye,
    EyeOff,
    Lock,
    Unlock,
    Key,
    Star,
    StarFilled,
    Heart,
    HeartFilled,
    Bell,
    BellOff,
    Refresh,
    RotateCw,
    RotateCcw,
    ExternalLink,
    Code,
    Terminal,
    Database,
    Cloud,
    CloudUpload,
    CloudDownload,
    Server,
    Wifi,
    WifiOff,
    Battery,
    Power,
    Zap,
    Activity,
    PieChart,
    BarChart,
    TrendingUp,
    TrendingDown,
    Globe,
    Map,
    MapPin,
    Navigation,
    Compass,
    Flag,
    Award,
    Gift,
    ShoppingCart,
    ShoppingBag,
    CreditCard,
    DollarSign,
    Percent,
    Package,
    Box,
    Layers,
    Layout,
    Grid,
    List,
    Columns,
    Sidebar,
    Maximize,
    Minimize,
    Move,
    Crosshair,
    Target,
    Sliders,
    Tool,
    Wrench,
    Hammer,
    Cpu,
    HardDrive,
    Monitor,
    Smartphone,
    Tablet,
    Watch,
    Printer,
    Headphones,
    Speaker,
    Radio,
    Bluetooth,
    Cast,
    Airplay,
    Sun,
    Moon,
    Sunrise,
    Sunset,
    CloudRain,
    Umbrella,
    Thermometer,
    Droplet,
    Wind,
}

impl IconName {
    /// Get SVG path data for the icon
    pub fn path(&self) -> &'static str {
        match self {
            // Actions
            IconName::Check => "M20 6L9 17l-5-5",
            IconName::X => "M18 6L6 18M6 6l12 12",
            IconName::Plus => "M12 5v14M5 12h14",
            IconName::Minus => "M5 12h14",
            IconName::Edit => "M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z",
            IconName::Trash => "M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2",
            IconName::Copy => "M20 9h-9a2 2 0 00-2 2v9a2 2 0 002 2h9a2 2 0 002-2v-9a2 2 0 00-2-2zM5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1",
            IconName::Save => "M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2zM17 21v-8H7v8M7 3v5h8",
            IconName::Download => "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3",
            IconName::Upload => "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",
            IconName::Print => "M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2M6 14h12v8H6z",
            IconName::Share => "M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8M16 6l-4-4-4 4M12 2v13",
            IconName::Undo => "M3 7v6h6M21 17a9 9 0 00-9-9 9 9 0 00-6.36 2.64L3 13",
            IconName::Redo => "M21 7v6h-6M3 17a9 9 0 019-9 9 9 0 016.36 2.64L21 13",

            // Navigation
            IconName::ChevronLeft => "M15 18l-6-6 6-6",
            IconName::ChevronRight => "M9 18l6-6-6-6",
            IconName::ChevronUp => "M18 15l-6-6-6 6",
            IconName::ChevronDown => "M6 9l6 6 6-6",
            IconName::ArrowLeft => "M19 12H5M12 19l-7-7 7-7",
            IconName::ArrowRight => "M5 12h14M12 5l7 7-7 7",
            IconName::ArrowUp => "M12 19V5M5 12l7-7 7 7",
            IconName::ArrowDown => "M12 5v14M19 12l-7 7-7-7",
            IconName::Menu => "M3 12h18M3 6h18M3 18h18",
            IconName::Home => "M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z",
            IconName::MoreHorizontal => "M12 12h.01M19 12h.01M5 12h.01",
            IconName::MoreVertical => "M12 12h.01M12 5h.01M12 19h.01",

            // Status
            IconName::Info => "M12 16v-4M12 8h.01M22 12a10 10 0 11-20 0 10 10 0 0120 0z",
            IconName::Warning => "M12 9v2M12 15h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z",
            IconName::Error => "M12 8v4M12 16h.01M22 12a10 10 0 11-20 0 10 10 0 0120 0z",
            IconName::Success => "M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3",
            IconName::Question => "M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3M12 17h.01M22 12a10 10 0 11-20 0 10 10 0 0120 0z",

            // Objects
            IconName::User => "M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z",
            IconName::Users => "M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75M9 11a4 4 0 100-8 4 4 0 000 8z",
            IconName::Settings => "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z",
            IconName::Search => "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z",
            IconName::Filter => "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
            IconName::Calendar => "M19 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2zM16 2v4M8 2v4M3 10h18",
            IconName::Clock => "M12 22a10 10 0 100-20 10 10 0 000 20zM12 6v6l4 2",
            IconName::Mail => "M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2zM22 6l-10 7L2 6",
            IconName::Phone => "M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z",
            IconName::Link => "M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71",
            IconName::File => "M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9zM13 2v7h7",
            IconName::FileText => "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M16 13H8M16 17H8M10 9H8",
            IconName::Folder => "M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z",
            IconName::FolderOpen => "M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2v1M2 10l2.45 7.35A2 2 0 006.35 19h11.3a2 2 0 001.9-1.35L22 10H2z",
            IconName::Image => "M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zM8.5 10a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM21 15l-5-5L5 21",
            IconName::Clipboard => "M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2M9 2h6a1 1 0 011 1v2a1 1 0 01-1 1H9a1 1 0 01-1-1V3a1 1 0 011-1z",
            IconName::ClipboardCheck => "M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2M9 2h6a1 1 0 011 1v2a1 1 0 01-1 1H9a1 1 0 01-1-1V3a1 1 0 011-1zM9 14l2 2 4-4",
            IconName::Notepad => "M14.5 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V7.5L14.5 2zM14 2v6h6M16 13H8M16 17H8M10 9H8",
            IconName::Book => "M4 19.5A2.5 2.5 0 016.5 17H20M4 19.5A2.5 2.5 0 016.5 22H20V2H6.5A2.5 2.5 0 004 4.5v15z",
            IconName::BookOpen => "M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2zM22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z",
            IconName::Bookmark => "M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z",
            IconName::Tag => "M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82zM7 7h.01",
            IconName::Tags => "M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82zM7 7h.01",

            // Media
            IconName::Play => "M5 3l14 9-14 9V3z",
            IconName::Pause => "M6 4h4v16H6zM14 4h4v16h-4z",
            IconName::Stop => "M6 4h12v16H6z",
            IconName::SkipBack => "M19 20L9 12l10-8v16zM5 19V5",
            IconName::SkipForward => "M5 4l10 8-10 8V4zM19 5v14",
            IconName::Volume => "M11 5L6 9H2v6h4l5 4V5zM19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07",
            IconName::VolumeOff => "M11 5L6 9H2v6h4l5 4V5zM23 9l-6 6M17 9l6 6",
            IconName::Mic => "M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3zM19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8",
            IconName::MicOff => "M1 1l22 22M9 9v3a3 3 0 005.12 2.12M15 9.34V4a3 3 0 00-5.94-.6M17 16.95A7 7 0 015 12v-2m14 0v2a7 7 0 01-.11 1.23M12 19v4M8 23h8",
            IconName::Camera => "M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2zM12 17a4 4 0 100-8 4 4 0 000 8z",
            IconName::Video => "M23 7l-7 5 7 5V7zM14 5H3a2 2 0 00-2 2v10a2 2 0 002 2h11a2 2 0 002-2V7a2 2 0 00-2-2z",

            // Communication
            IconName::MessageSquare => "M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z",
            IconName::MessageCircle => "M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z",
            IconName::Send => "M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z",
            IconName::Inbox => "M22 12h-6l-2 3h-4l-2-3H2M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z",

            // Misc
            IconName::Eye => "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 15a3 3 0 100-6 3 3 0 000 6z",
            IconName::EyeOff => "M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24M1 1l22 22",
            IconName::Lock => "M19 11H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2zM7 11V7a5 5 0 0110 0v4",
            IconName::Unlock => "M19 11H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2zM7 11V7a5 5 0 019.9-1",
            IconName::Key => "M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4",
            IconName::Star => "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
            IconName::StarFilled => "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
            IconName::Heart => "M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z",
            IconName::HeartFilled => "M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z",
            IconName::Bell => "M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0",
            IconName::BellOff => "M13.73 21a2 2 0 01-3.46 0M18.63 13A17.89 17.89 0 0118 8M6.26 6.26A5.86 5.86 0 006 8c0 7-3 9-3 9h14M18 8a6 6 0 00-9.33-5M1 1l22 22",
            IconName::Refresh => "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15",
            IconName::RotateCw => "M23 4v6h-6M21 12a9 9 0 11-9-9 9.75 9.75 0 016.74 2.74L23 10",
            IconName::RotateCcw => "M1 4v6h6M3 12a9 9 0 019-9 9.75 9.75 0 016.74 2.74L23 10",
            IconName::ExternalLink => "M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3",
            IconName::Code => "M16 18l6-6-6-6M8 6l-6 6 6 6",
            IconName::Terminal => "M4 17l6-6-6-6M12 19h8",
            IconName::Database => "M12 2C6.48 2 2 4.69 2 8s4.48 6 10 6 10-2.69 10-6-4.48-6-10-6zM2 8v8c0 3.31 4.48 6 10 6s10-2.69 10-6V8",
            IconName::Cloud => "M18 10h-1.26A8 8 0 109 20h9a5 5 0 000-10z",
            IconName::CloudUpload => "M16.88 3.549A8 8 0 009 12H6a5 5 0 000 10h9a7 7 0 007-7c0-1.3-.3-2.6-.92-3.87M12 10V2M16 6l-4-4-4 4",
            IconName::CloudDownload => "M8 17l4 4 4-4M12 12v9M20.88 18.09A5 5 0 0018 9h-1.26A8 8 0 103 16.29",
            IconName::Server => "M2 4h20v6H2zM2 14h20v6H2zM6 7h.01M6 17h.01",
            IconName::Wifi => "M5 12.55a11 11 0 0114.08 0M1.42 9a16 16 0 0121.16 0M8.53 16.11a6 6 0 016.95 0M12 20h.01",
            IconName::WifiOff => "M1 1l22 22M16.72 11.06A10.94 10.94 0 0119 12.55M5 12.55a10.94 10.94 0 015.17-2.39M10.71 5.05A16 16 0 0122.58 9M1.42 9a15.91 15.91 0 014.7-2.88M8.53 16.11a6 6 0 016.95 0M12 20h.01",
            IconName::Battery => "M23 11h-2V6H1v12h20v-5h2zM6 11v2",
            IconName::Power => "M18.36 6.64a9 9 0 11-12.73 0M12 2v10",
            IconName::Zap => "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
            IconName::Activity => "M22 12h-4l-3 9L9 3l-3 9H2",
            IconName::PieChart => "M21.21 15.89A10 10 0 118 2.83M22 12A10 10 0 0012 2v10z",
            IconName::BarChart => "M12 20V10M18 20V4M6 20v-4",
            IconName::TrendingUp => "M23 6l-9.5 9.5-5-5L1 18M17 6h6v6",
            IconName::TrendingDown => "M23 18l-9.5-9.5-5 5L1 6M17 18h6v-6",
            IconName::Globe => "M12 22a10 10 0 100-20 10 10 0 000 20zM2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z",
            IconName::Map => "M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4zM8 2v16M16 6v16",
            IconName::MapPin => "M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0zM12 13a3 3 0 100-6 3 3 0 000 6z",
            IconName::Navigation => "M3 11l19-9-9 19-2-8-8-2z",
            IconName::Compass => "M12 22a10 10 0 100-20 10 10 0 000 20zM16.24 7.76l-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z",
            IconName::Flag => "M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7",
            IconName::Award => "M12 15a7 7 0 100-14 7 7 0 000 14zM8.21 13.89L7 23l5-3 5 3-1.21-9.12",
            IconName::Gift => "M20 12v10H4V12M2 7h20v5H2zM12 22V7M12 7H7.5a2.5 2.5 0 010-5C11 2 12 7 12 7zM12 7h4.5a2.5 2.5 0 000-5C13 2 12 7 12 7z",
            IconName::ShoppingCart => "M9 22a1 1 0 100-2 1 1 0 000 2zM20 22a1 1 0 100-2 1 1 0 000 2zM1 1h4l2.68 13.39a2 2 0 002 1.61h9.72a2 2 0 002-1.61L23 6H6",
            IconName::ShoppingBag => "M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4zM3 6h18M16 10a4 4 0 01-8 0",
            IconName::CreditCard => "M21 4H3a2 2 0 00-2 2v12a2 2 0 002 2h18a2 2 0 002-2V6a2 2 0 00-2-2zM1 10h22",
            IconName::DollarSign => "M12 1v22M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6",
            IconName::Percent => "M19 5L5 19M6.5 9a2.5 2.5 0 100-5 2.5 2.5 0 000 5zM17.5 20a2.5 2.5 0 100-5 2.5 2.5 0 000 5z",
            IconName::Package => "M16.5 9.4l-9-5.19M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16zM3.27 6.96L12 12.01l8.73-5.05M12 22.08V12",
            IconName::Box => "M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z",
            IconName::Layers => "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
            IconName::Layout => "M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zM3 9h18M9 21V9",
            IconName::Grid => "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z",
            IconName::List => "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
            IconName::Columns => "M12 3h7a2 2 0 012 2v14a2 2 0 01-2 2h-7m0-18H5a2 2 0 00-2 2v14a2 2 0 002 2h7m0-18v18",
            IconName::Sidebar => "M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zM9 3v18",
            IconName::Maximize => "M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3",
            IconName::Minimize => "M8 3v3a2 2 0 01-2 2H3m18 0h-3a2 2 0 01-2-2V3m0 18v-3a2 2 0 012-2h3M3 16h3a2 2 0 012 2v3",
            IconName::Move => "M5 9l-3 3 3 3M9 5l3-3 3 3M15 19l-3 3-3-3M19 9l3 3-3 3M2 12h20M12 2v20",
            IconName::Crosshair => "M12 22a10 10 0 100-20 10 10 0 000 20zM22 12h-4M6 12H2M12 6V2M12 22v-4",
            IconName::Target => "M12 22a10 10 0 100-20 10 10 0 000 20zM12 18a6 6 0 100-12 6 6 0 000 12zM12 14a2 2 0 100-4 2 2 0 000 4z",
            IconName::Sliders => "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
            IconName::Tool => "M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z",
            IconName::Wrench => "M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z",
            IconName::Hammer => "M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z",
            IconName::Cpu => "M18 4H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2zM9 9h6v6H9zM9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3",
            IconName::HardDrive => "M22 12H2M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11zM6 16h.01",
            IconName::Monitor => "M20 3H4a2 2 0 00-2 2v10a2 2 0 002 2h16a2 2 0 002-2V5a2 2 0 00-2-2zM8 21h8M12 17v4",
            IconName::Smartphone => "M17 2H7a2 2 0 00-2 2v16a2 2 0 002 2h10a2 2 0 002-2V4a2 2 0 00-2-2zM12 18h.01",
            IconName::Tablet => "M18 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V4a2 2 0 00-2-2zM12 18h.01",
            IconName::Watch => "M12 19a7 7 0 100-14 7 7 0 000 14zM12 19v3M12 5V2M16.13 6.87l2.12-2.12M5.75 4.75l2.12 2.12M17.24 17.24l2.12 2.12M4.64 19.36l2.12-2.12",
            IconName::Printer => "M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2M6 14h12v8H6z",
            IconName::Headphones => "M3 18v-6a9 9 0 0118 0v6M21 19a2 2 0 01-2 2h-1a2 2 0 01-2-2v-3a2 2 0 012-2h3zM3 19a2 2 0 002 2h1a2 2 0 002-2v-3a2 2 0 00-2-2H3z",
            IconName::Speaker => "M19 7H5a2 2 0 00-2 2v6a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2zM12 14a2 2 0 100-4 2 2 0 000 4z",
            IconName::Radio => "M12 14a2 2 0 100-4 2 2 0 000 4zM16.24 7.76a6 6 0 010 8.49m-8.48-.01a6 6 0 010-8.49m11.31-2.82a10 10 0 010 14.14m-14.14 0a10 10 0 010-14.14",
            IconName::Bluetooth => "M6.5 6.5l11 11L12 23V1l5.5 5.5-11 11",
            IconName::Cast => "M2 16.1A5 5 0 015.9 20M2 12.05A9 9 0 019.95 20M2 8V6a2 2 0 012-2h16a2 2 0 012 2v12a2 2 0 01-2 2h-6M2 20h.01",
            IconName::Airplay => "M5 17H4a2 2 0 01-2-2V5a2 2 0 012-2h16a2 2 0 012 2v10a2 2 0 01-2 2h-1M12 15l5 6H7l5-6z",
            IconName::Sun => "M12 17a5 5 0 100-10 5 5 0 000 10zM12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42",
            IconName::Moon => "M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z",
            IconName::Sunrise => "M17 18a5 5 0 00-10 0M12 2v7M4.22 10.22l1.42 1.42M1 18h2M21 18h2M18.36 11.64l1.42-1.42M23 22H1M8 6l4-4 4 4",
            IconName::Sunset => "M17 18a5 5 0 00-10 0M12 9v7M4.22 10.22l1.42 1.42M1 18h2M21 18h2M18.36 11.64l1.42-1.42M23 22H1M16 5l-4 4-4-4",
            IconName::CloudRain => "M16 13v8M8 13v8M12 15v8M20 16.58A5 5 0 0018 7h-1.26A8 8 0 104 15.25",
            IconName::Umbrella => "M23 12a11.05 11.05 0 00-22 0zM12 12v9a3 3 0 006 0",
            IconName::Thermometer => "M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z",
            IconName::Droplet => "M12 2.69l5.66 5.66a8 8 0 11-11.31 0z",
            IconName::Wind => "M9.59 4.59A2 2 0 1111 8H2m10.59 11.41A2 2 0 1014 16H2m15.73-8.27A2.5 2.5 0 1119.5 12H2",
        }
    }

    /// Icon size in pixels
    pub fn default_size(&self) -> u32 {
        24
    }
}

/// Icon props
#[derive(Props, Clone, PartialEq)]
pub struct IconProps {
    /// Icon name
    pub name: IconName,

    /// Size (overrides icon size)
    #[props(default)]
    pub size: Option<Size>,

    /// Custom size in pixels
    #[props(default)]
    pub px: Option<u32>,

    /// Color (CSS color value)
    #[props(default)]
    pub color: Option<String>,

    /// Stroke width
    #[props(default = 2)]
    pub stroke_width: u32,

    /// Additional class
    #[props(default)]
    pub class: Option<String>,
}

/// Icon component
#[component]
pub fn Icon(props: IconProps) -> Element {
    let size = props.px.unwrap_or_else(|| {
        match props.size {
            Some(Size::Xs) => 12,
            Some(Size::Sm) => 16,
            Some(Size::Md) | None => 20,
            Some(Size::Lg) => 24,
            Some(Size::Xl) => 32,
        }
    });

    let class = format!(
        "rust-ui-icon {}",
        props.class.as_deref().unwrap_or(""),
    );

    rsx! {
        svg {
            class: "{class}",
            width: "{size}",
            height: "{size}",
            view_box: "0 0 24 24",
            fill: "none",
            stroke: props.color.as_deref().unwrap_or("currentColor"),
            stroke_width: "{props.stroke_width}",
            stroke_linecap: "round",
            stroke_linejoin: "round",

            path { d: "{props.name.path()}" }
        }
    }
}
