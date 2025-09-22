import SwiftUI

struct ChatMessageListView: View {
    let messages: [ChatMessage]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(messages) { message in
                        ChatMessageBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding(.vertical, 16)
                .padding(.horizontal, 20)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .onChange(of: messages.last?.id) { id in
                guard let id else { return }
                withAnimation(.easeInOut(duration: 0.3)) {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
        }
    }
}

private struct ChatMessageBubble: View {
    let message: ChatMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(message.role.label.uppercased())
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(roleColor)
                Spacer()
                Text(message.formattedTimestamp)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            MarkdownRenderer.text(from: message.content.isEmpty ? "…" : message.content)
                .font(.body)
                .textSelection(.enabled)
            if !message.metadata.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(message.metadata.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        Text("\(key): \(value)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(8)
                .background(Color(nsColor: .quaternaryLabelColor).opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            if message.isStreaming {
                ProgressView()
                    .scaleEffect(0.6, anchor: .leading)
                    .padding(.top, 4)
            }
        }
        .padding(16)
        .background(bubbleBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var bubbleBackground: some View {
        switch message.role {
        case .user:
            return Color.accentColor.opacity(0.12)
        case .assistant:
            return Color.blue.opacity(0.1)
        case .system:
            return Color.gray.opacity(0.08)
        }
    }

    private var roleColor: Color {
        switch message.role {
        case .user:
            return .accentColor
        case .assistant:
            return .blue
        case .system:
            return .gray
        }
    }
}
