# rust-ui Use Cases

This document outlines what types of applications rust-ui is designed for, and when you should use alternative frameworks.

---

## What rust-ui Excels At

rust-ui (Dioxus-based) is ideal for building **UI-driven applications**:

### Perfect Use Cases ✅

| Type | Examples | Why It Works |
|------|----------|--------------|
| **Gamified Learning** | Duolingo, Codecademy, Kahoot | UI components, progress tracking, animations |
| **Quiz Applications** | Trivia games, assessments | Cards, buttons, scoring |
| **Board Games** | Chess, Sudoku, Tic-Tac-Toe | Turn-based, grid layouts |
| **Card Games** | Memory matching, flashcards | CSS animations, state |
| **Interactive Stories** | Choose your adventure | Modal dialogs, branching |
| **Dashboards** | Admin panels, analytics | Tables, charts, forms |
| **Business Apps** | CRM, ERP, inventory | CRUD operations |
| **Developer Tools** | IDEs, debuggers, monitors | Layouts, tabs, panels |

### Available Components

```rust
// Progress and achievements
Progress { value: 750.0, max: 1000.0, show_label: true }
Badge { variant: Variant::Success, "Level 5" }
Avatar { src: user.avatar, status: AvatarStatus::Online }

// Interactive elements  
Button { variant: Variant::Primary, onclick: handle_answer, "Submit Answer" }
Card { title: "Challenge 1", ... }
Modal { open: show_celebration, "🎉 Level Up!" }

// Feedback
Toast { message: "+50 XP!", variant: Variant::Success }
Alert { variant: Variant::Info, "New achievement unlocked!" }

// Navigation
Tabs { tabs: lesson_tabs, active: current_lesson }
Dropdown { items: menu_items, on_select: handle_nav }
```

---

## What rust-ui Is NOT For

rust-ui is **not a game engine**. It cannot handle real-time graphics-intensive applications.

### Not Suitable ❌

| Type | Examples | Why Not |
|------|----------|---------|
| **Real-time 2D Games** | Platformers, shooters | No game loop, no sprite rendering |
| **3D Games** | FPS, racing, RPG | No 3D rendering pipeline |
| **Physics Games** | Angry Birds, pool | No physics engine |
| **Fast Animation** | 60fps gameplay | HTML/CSS is too slow |
| **Pixel Art Games** | Retro-style games | No sprite sheets, no pixel-perfect rendering |

---

## Rust Game Engines

For actual games, use a proper game engine:

| Engine | Best For | GitHub |
|--------|----------|--------|
| **Bevy** | 2D/3D games, ECS architecture | [bevyengine/bevy](https://github.com/bevyengine/bevy) |
| **macroquad** | Simple 2D games, quick prototypes | [not-fl3/macroquad](https://github.com/not-fl3/macroquad) |
| **ggez** | 2D game framework | [ggez/ggez](https://github.com/ggez/ggez) |
| **Fyrox** | Full 3D game engine | [FyroxEngine/Fyrox](https://github.com/FyroxEngine/Fyrox) |
| **Piston** | Modular game engine | [PistonDevelopers/piston](https://github.com/PistonDevelopers/piston) |

### Game Engine Example (Bevy)

```rust
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, player_movement)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(SpriteBundle { ... }); // Real sprites!
}

fn player_movement(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Transform, With<Player>>,
) {
    // 60fps game loop with real-time input
}
```

---

## Decision Tree

```
What are you building?
│
├── Interactive UI app with gamification?
│   └── ✅ Use rust-ui (Dioxus)
│       ├── Quizzes, learning platforms
│       ├── Turn-based games (chess, cards)
│       ├── Dashboards with achievements
│       └── Progress tracking apps
│
├── Real-time 2D game?
│   └── ✅ Use Bevy or macroquad
│       ├── Platformers
│       ├── Top-down shooters
│       └── Arcade games
│
├── 3D game?
│   └── ✅ Use Bevy or Fyrox
│       ├── First-person games
│       ├── Racing games
│       └── 3D worlds
│
└── Hybrid (UI + Game)?
    └── ✅ rust-ui for menus, Bevy for gameplay
        ├── Game with complex UI overlays
        ├── Strategy games with map + panels
        └── RPGs with dialog systems
```

---

## Gamified App Example

A Duolingo-style learning app with rust-ui:

```rust
use components::prelude::*;

#[derive(Clone)]
struct User {
    level: u32,
    xp: u32,
    streak: u32,
    achievements: Vec<String>,
}

fn LearningApp() -> Element {
    let mut user = use_signal(|| User { level: 1, xp: 0, streak: 5, achievements: vec![] });
    let mut current_lesson = use_signal(|| 0);
    
    rsx! {
        div { class: "app",
            // Header with user stats
            header { class: "user-stats",
                Avatar { fallback: "JD" }
                
                div { class: "level",
                    Badge { variant: Variant::Primary, "Level {user.read().level}" }
                }
                
                div { class: "xp-bar",
                    Progress { 
                        value: user.read().xp as f32,
                        max: 100.0,
                        variant: Variant::Success,
                        show_label: true,
                    }
                }
                
                div { class: "streak",
                    Icon { name: IconName::Zap, color: "orange" }
                    span { "{user.read().streak} day streak" }
                }
            }
            
            // Lesson content
            main {
                Card {
                    title: "Lesson 1: Variables",
                    children: rsx! {
                        p { "In Rust, variables are immutable by default." }
                        pre { code { "let x = 5; // immutable" } }
                        pre { code { "let mut y = 10; // mutable" } }
                    }
                }
                
                // Quiz
                Card {
                    title: "Quick Quiz",
                    children: rsx! {
                        p { "What keyword makes a variable mutable?" }
                        
                        Button { 
                            variant: Variant::Secondary,
                            onclick: move |_| { /* wrong */ },
                            "let"
                        }
                        Button { 
                            variant: Variant::Secondary,
                            onclick: move |_| {
                                user.write().xp += 50;
                            },
                            "mut"
                        }
                        Button { 
                            variant: Variant::Secondary,
                            onclick: move |_| { /* wrong */ },
                            "var"
                        }
                    }
                }
            }
            
            // Achievement toast
            ToastContainer {
                toasts: if user.read().xp >= 50 {
                    vec![ToastData::success("🎉 +50 XP!")]
                } else {
                    vec![]
                },
                position: ToastPosition::TopRight,
            }
        }
    }
}
```

---

## Summary

| Application Type | Framework |
|-----------------|-----------|
| UI-driven gamified apps | **rust-ui** ✅ |
| Turn-based games | **rust-ui** ✅ |
| Real-time 2D games | **Bevy / macroquad** |
| 3D games | **Bevy / Fyrox** |
| Dashboards & business apps | **rust-ui** ✅ |
