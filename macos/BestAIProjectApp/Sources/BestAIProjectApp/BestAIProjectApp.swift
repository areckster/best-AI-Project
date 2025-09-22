import SwiftUI

@main
struct BestAIProjectApp: App {
    private let appState: AppState

    init() {
        let rootURL = BestAIProjectApp.locateProjectRoot()
        self.appState = AppState(projectRoot: rootURL)
    }

    var body: some Scene {
        WindowGroup {
            ContentView(appState: appState)
        }
        .windowStyle(.automatic)
    }

    private static func locateProjectRoot() -> URL {
        let fileManager = FileManager.default
        let candidates: [URL] = [
            URL(fileURLWithPath: fileManager.currentDirectoryPath),
            Bundle.main.bundleURL,
            Bundle.main.bundleURL.deletingLastPathComponent(),
            Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        ]

        for candidate in candidates {
            let path = candidate.appendingPathComponent("server.py").path
            if fileManager.fileExists(atPath: path) {
                return candidate
            }
        }

        // Fallback to current directory
        return URL(fileURLWithPath: fileManager.currentDirectoryPath)
    }
}
