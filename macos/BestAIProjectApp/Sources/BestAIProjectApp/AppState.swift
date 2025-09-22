import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var composerText: String = ""
    @Published var isStreaming = false
    @Published var lastError: String?
    @Published var healthStatus: HealthStatus?
    @Published var models: [ModelSummary] = []
    @Published var selectedModel: String = ""
    @Published var settings = ChatSettings()
    @Published var systemPrompt: String = ""
    @Published var developerPrompt: String = ""
    @Published var toolsEnabled: Bool = true
    @Published var backendConfiguration: BackendConfiguration = .default {
        didSet { persistConfiguration() }
    }

    @Published private(set) var pythonManager: PythonServerManager

    private let chatService: ChatService
    private var streamTask: Task<Void, Never>?
    private let configurationURL: URL

    init(projectRoot: URL) {
        self.configurationURL = projectRoot.appendingPathComponent(".bestaiapp.json")
        self.backendConfiguration = AppState.loadConfiguration(from: configurationURL) ?? .default
        self.chatService = ChatService(baseURL: backendConfiguration.baseURL)
        self.pythonManager = PythonServerManager(projectRoot: projectRoot)
    }

    func onAppear() {
        if backendConfiguration.autoLaunchBackend,
           let python = backendConfiguration.pythonInterpreter {
            Task { await pythonManager.start(pythonInterpreter: python) }
        }
        Task { await refreshHealth() }
        Task { await refreshModels() }
    }

    func send() {
        let prompt = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }

        composerText = ""
        lastError = nil

        let userMessage = ChatMessage(role: .user, content: prompt)
        let historyBeforeSend = messages
        let payloadHistory = (historyBeforeSend + [userMessage]).map { $0.payload }

        messages.append(userMessage)
        let placeholder = ChatMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(placeholder)
        isStreaming = true

        streamTask?.cancel()
        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = await chatService.stream(
                    history: payloadHistory,
                    settings: settings,
                    systemPrompt: systemPrompt,
                    developerPrompt: developerPrompt,
                    toolsEnabled: toolsEnabled
                )

                for try await event in stream {
                    guard !Task.isCancelled else { break }
                    await MainActor.run {
                        self.apply(event: event, to: placeholder.id)
                    }
                }

                await MainActor.run {
                    self.finishStreaming(messageID: placeholder.id)
                }
            } catch {
                await MainActor.run {
                    self.failStreaming(error: error, messageID: placeholder.id)
                }
            }
        }
    }

    func cancelStreaming() {
        streamTask?.cancel()
        streamTask = nil
        isStreaming = false
        if let index = messages.lastIndex(where: { $0.isStreaming }) {
            messages[index].isStreaming = false
            messages[index].content += "\n\n_Cancelled by user._"
        }
    }

    private func apply(event: ChatStreamEvent, to messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        switch event.type {
        case .delta:
            if let delta = event.delta {
                messages[index].content.append(delta)
            }
        case .done:
            messages[index].isStreaming = false
            isStreaming = false
        case .error:
            let errorText = event.errorDescription ?? "Unknown error"
            messages[index].content = "**Error:** \(errorText)"
            messages[index].isStreaming = false
            isStreaming = false
            lastError = errorText
        case .toolCalls:
            if let existing = messages[index].metadata["tool_calls"] {
                messages[index].metadata["tool_calls"] = existing + "\n" + (event.raw.description)
            } else {
                messages[index].metadata["tool_calls"] = event.raw.description
            }
        case .toolResult:
            if let output = event.raw["output"]?.stringValue {
                messages[index].metadata["tool_result"] = output
            }
        case .gateWarning:
            let warning = event.message ?? "The model may not be allowed to answer."
            messages[index].metadata["warning"] = warning
        case .thinking, .message, .usage, .unknown:
            break
        }
    }

    private func finishStreaming(messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[index].isStreaming = false
        if messages[index].content.isEmpty {
            messages[index].content = "_(No response)_"
        }
        isStreaming = false
    }

    private func failStreaming(error: Error, messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        let message = error.localizedDescription
        messages[index].content = "**Error:** \(message)"
        messages[index].isStreaming = false
        lastError = message
        isStreaming = false
    }

    func refreshHealth() async {
        do {
            let status = try await chatService.health()
            await MainActor.run {
                self.healthStatus = status
                if let model = status.model, !model.isEmpty {
                    self.selectedModel = model
                }
            }
        } catch {
            await MainActor.run {
                self.healthStatus = HealthStatus(ok: false, ollama: backendConfiguration.baseURL.absoluteString, model: nil)
                self.lastError = error.localizedDescription
            }
        }
    }

    func refreshModels() async {
        do {
            let items = try await chatService.fetchModels()
            await MainActor.run {
                self.models = items.sorted { $0.name < $1.name }
            }
        } catch {
            await MainActor.run {
                self.lastError = error.localizedDescription
            }
        }
    }

    func applyModel(tag: String) async {
        do {
            let applied = try await chatService.setModel(tag: tag)
            await MainActor.run {
                self.selectedModel = applied
                Task { await self.refreshHealth() }
            }
        } catch {
            await MainActor.run {
                self.lastError = error.localizedDescription
            }
        }
    }

    func updateBackendURL(_ url: URL) {
        backendConfiguration.baseURL = url
        Task { await chatService.update(baseURL: url) }
        Task { await refreshHealth() }
    }

    func toggleAutoLaunch(_ enabled: Bool) {
        backendConfiguration.autoLaunchBackend = enabled
    }

    func setPythonInterpreter(_ url: URL?) {
        backendConfiguration.pythonInterpreter = url
    }

    func startPythonBackend() {
        guard let python = backendConfiguration.pythonInterpreter else {
            lastError = "Select a Python interpreter before launching the backend."
            return
        }
        Task { await pythonManager.start(pythonInterpreter: python) }
    }

    func stopPythonBackend() {
        pythonManager.stop()
    }

    private func persistConfiguration() {
        do {
            let data = try JSONEncoder().encode(backendConfiguration)
            try data.write(to: configurationURL, options: .atomic)
        } catch {
            lastError = "Failed to persist configuration: \(error.localizedDescription)"
        }
    }

    private static func loadConfiguration(from url: URL) -> BackendConfiguration? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(BackendConfiguration.self, from: data)
    }
}
