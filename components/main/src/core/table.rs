//! Table component

use dioxus::prelude::*;

/// Table column definition
#[derive(Clone, PartialEq)]
pub struct Column<T: Clone + PartialEq + 'static> {
    pub key: String,
    pub header: String,
    pub render: fn(&T) -> Element,
    pub sortable: bool,
    pub width: Option<String>,
}

impl<T: Clone + PartialEq + 'static> Column<T> {
    pub fn new(key: impl Into<String>, header: impl Into<String>, render: fn(&T) -> Element) -> Self {
        Self {
            key: key.into(),
            header: header.into(),
            render,
            sortable: false,
            width: None,
        }
    }

    pub fn sortable(mut self) -> Self {
        self.sortable = true;
        self
    }

    pub fn width(mut self, width: impl Into<String>) -> Self {
        self.width = Some(width.into());
        self
    }
}

/// Sort direction
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    #[default]
    None,
    Asc,
    Desc,
}

/// Table props
#[derive(Props, Clone, PartialEq)]
pub struct TableProps<T: Clone + PartialEq + 'static> {
    /// Data rows
    pub data: Vec<T>,

    /// Column definitions
    pub columns: Vec<Column<T>>,

    /// Striped rows
    #[props(default = true)]
    pub striped: bool,

    /// Hoverable rows
    #[props(default = true)]
    pub hoverable: bool,

    /// Bordered
    #[props(default = false)]
    pub bordered: bool,

    /// Compact mode
    #[props(default = false)]
    pub compact: bool,

    /// Row click handler
    #[props(default)]
    pub on_row_click: Option<EventHandler<T>>,

    /// Loading state
    #[props(default = false)]
    pub loading: bool,

    /// Empty message
    #[props(default = "No data")]
    pub empty_message: &'static str,
}

/// Table component
#[component]
pub fn Table<T: Clone + PartialEq + 'static>(props: TableProps<T>) -> Element {
    let class = format!(
        "rust-ui-table {} {} {} {}",
        if props.striped { "striped" } else { "" },
        if props.hoverable { "hoverable" } else { "" },
        if props.bordered { "bordered" } else { "" },
        if props.compact { "compact" } else { "" },
    );

    rsx! {
        div { class: "rust-ui-table-container",
            if props.loading {
                div { class: "table-loading",
                    div { class: "spinner" }
                }
            }

            table { class: "{class}",
                thead {
                    tr {
                        for col in props.columns.iter() {
                            th {
                                style: col.width.as_ref().map(|w| format!("width: {}", w)),
                                class: if col.sortable { "sortable" } else { "" },
                                "{col.header}"
                            }
                        }
                    }
                }
                tbody {
                    if props.data.is_empty() {
                        tr { class: "empty-row",
                            td {
                                colspan: "{props.columns.len()}",
                                "{props.empty_message}"
                            }
                        }
                    } else {
                        for row in props.data.iter() {
                            tr {
                                onclick: {
                                    let row = row.clone();
                                    move |_| {
                                        if let Some(handler) = &props.on_row_click {
                                            handler.call(row.clone());
                                        }
                                    }
                                },
                                for col in props.columns.iter() {
                                    td { {(col.render)(row)} }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
