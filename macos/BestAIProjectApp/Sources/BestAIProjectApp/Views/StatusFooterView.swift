import SwiftUI

struct StatusFooterView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        HStack(spacing: 16) {
            if let status = appState.healthStatus {
                Label(
                    status.ok ? "Backend OK" : "Backend unreachable",
                    systemImage: status.ok ? "checkmark.circle" : "exclamationmark.triangle"
                )
                .foregroundStyle(status.ok ? Color.green : Color.orange)
                if let model = status.model {
                    Text("Model: \(model)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Label("Health unknown", systemImage: "questionmark.diamond")
                    .foregroundStyle(.secondary)
            }

            Divider()
                .frame(height: 20)

            Label(appState.pythonManager.isRunning ? "Backend running" : "Backend stopped", systemImage: "terminal")
                .foregroundStyle(appState.pythonManager.isRunning ? Color.green : Color.secondary)

            if let error = appState.lastError {
                Divider().frame(height: 20)
                Label(error, systemImage: "bolt.trianglebadge.exclamationmark")
                    .foregroundStyle(Color.red)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.95))
    }
}
