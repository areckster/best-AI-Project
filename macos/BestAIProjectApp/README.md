# BestAIProjectApp (macOS)

A native SwiftUI macOS client for the FastAPI backend in this repository. The project is structured as a Swift Package so you can open the folder directly in Xcode, run the desktop app, and iterate with the SwiftUI previews.

## Features

- Real-time chat streaming via the `/api/chat/stream` endpoint (Server-Sent Events)
- Backend health indicator with manual refresh
- Model browser with the ability to switch the active Ollama tag
- Editable system & developer prompts and advanced generation settings
- Optional auto-launch of the bundled Python backend so everything runs from the desktop app

## Opening in Xcode

1. Install Xcode 15 or newer on macOS 13 Ventura (or later).
2. From the Xcode welcome window choose **Open a project or file…** and select the `macos/BestAIProjectApp` directory.
3. Xcode treats Swift packages as first-class projects; pick the `BestAIProjectApp` scheme and run it on "My Mac".

## Local backend workflow

The SwiftUI app assumes that the FastAPI server in this repository is available on `http://127.0.0.1:8000`.

- If you are already running `python server.py` (or `./start.sh`) separately you can hit **Connect** inside the macOS app to verify the connection.
- To launch the backend directly from the macOS app, point the **Python Runtime** setting at your preferred interpreter (`/usr/bin/python3`, a virtual environment, etc.) and enable **Launch Python backend on start**. The app will spawn the server in-process and show its lifecycle status at the bottom of the sidebar.

## Directory layout

```
macos/BestAIProjectApp
├── Package.swift                # Swift Package manifest (open in Xcode)
├── README.md
└── Sources
    └── BestAIProjectApp
        ├── BestAIProjectApp.swift    # @main entry point
        ├── AppState.swift            # ObservableObject orchestrating chat state
        ├── Models.swift              # Shared DTOs & JSON helpers
        ├── Services
        │   ├── ChatService.swift     # HTTP/SSE client for the FastAPI backend
        │   └── PythonServerManager.swift  # Launches/stops `python server.py`
        ├── Support
        │   └── MarkdownRenderer.swift    # Lightweight Markdown → AttributedString bridge
        └── Views
            ├── ContentView.swift
            ├── ChatComposerView.swift
            ├── ChatMessageListView.swift
            ├── SettingsSidebarView.swift
            └── StatusFooterView.swift
```

You can freely extend the UI or add new Swift packages inside the same folder; Xcode will detect new files automatically.

## Tests

The current project focuses on the UI layer. If you add model or service code that benefits from unit tests, create a `Tests/` directory next to `Sources/` and Xcode will generate a matching test target automatically.
