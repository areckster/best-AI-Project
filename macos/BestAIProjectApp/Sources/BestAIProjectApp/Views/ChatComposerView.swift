import SwiftUI

struct ChatComposerView: View {
    @Binding var text: String
    var isStreaming: Bool
    var sendAction: () -> Void
    var cancelAction: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 12) {
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                TextEditor(text: $text)
                    .focused($isFocused)
                    .frame(minHeight: 80, maxHeight: 160)
                    .overlay(alignment: .topLeading) {
                        if text.isEmpty {
                            Text("Type your message…")
                                .foregroundStyle(.secondary)
                                .padding(.top, 8)
                                .padding(.leading, 6)
                        }
                    }
                    .onSubmit(send)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                VStack(spacing: 8) {
                    Button(action: send) {
                        Label(isStreaming ? "Stop" : "Send", systemImage: isStreaming ? "stop.fill" : "paperplane.fill")
                            .labelStyle(.titleAndIcon)
                    }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .buttonStyle(.borderedProminent)
                    .disabled(!isStreaming && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if isStreaming {
                        Button("Cancel", action: cancelAction)
                            .buttonStyle(.bordered)
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 20)
    }

    private func send() {
        if isStreaming {
            cancelAction()
        } else {
            sendAction()
        }
    }
}
