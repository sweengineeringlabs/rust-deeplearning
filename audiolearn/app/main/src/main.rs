//! AudioLearn entry point

use audiolearn_app::AudioLearnApp;

#[cfg(feature = "desktop")]
fn main() {
    use dioxus::desktop::{Config, WindowBuilder};

    let config = Config::new()
        .with_window(
            WindowBuilder::new()
                .with_title("AudioLearn")
                .with_inner_size(dioxus::desktop::LogicalSize::new(1200.0, 800.0))
                .with_min_inner_size(dioxus::desktop::LogicalSize::new(800.0, 600.0))
                .with_always_on_top(false)
                .with_decorations(true)
                .with_resizable(true)
        );

    dioxus::LaunchBuilder::desktop()
        .with_cfg(config)
        .launch(AudioLearnApp);
}

#[cfg(all(feature = "web", not(feature = "desktop")))]
fn main() {
    dioxus::launch(AudioLearnApp);
}
