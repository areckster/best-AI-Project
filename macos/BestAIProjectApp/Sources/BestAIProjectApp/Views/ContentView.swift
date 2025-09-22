import SwiftUI

struct ContentView: View {
    @StateObject private var appState: AppState

    init(appState: AppState) {
        _appState = StateObject(wrappedValue: appState)
    }

    var body: some View {
        NavigationSplitView {
            SettingsSidebarView(appState: appState)
                .frame(minWidth: 320)
        } detail: {
            VStack(spacing: 0) {
                ChatMessageListView(messages: appState.messages)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                ChatComposerView(
                    text: $appState.composerText,
                    isStreaming: appState.isStreaming,
                    sendAction: appState.send,
                    cancelAction: appState.cancelStreaming
                )
                StatusFooterView(appState: appState)
            }
            .toolbar {
                ToolbarItemGroup(placement: .automatic) {
                    Button("Check health") {
                        Task { await appState.refreshHealth() }
                    }
                    Button("Refresh models") {
                        Task { await appState.refreshModels() }
                    }
                }
            }
        }
        .onAppear { appState.onAppear() }
        .alert(isPresented: Binding<Bool>(
            get: { appState.lastError != nil },
            set: { if !$0 { appState.lastError = nil } }
        )) {
            Alert(
                title: Text("Error"),
                message: Text(appState.lastError ?? "Unknown error"),
                dismissButton: .default(Text("OK"))
            )
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(appState: AppState(projectRoot: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)))
    }
}
