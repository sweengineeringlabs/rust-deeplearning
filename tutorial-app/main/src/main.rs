//! Tutorial App entry point

use dioxus::prelude::*;
use tutorial_app::TutorialApp;

fn main() {
    // Launch as web or desktop depending on feature
    #[cfg(feature = "web")]
    dioxus::launch(TutorialApp);
    
    #[cfg(feature = "desktop")]
    dioxus::launch(TutorialApp);
}
