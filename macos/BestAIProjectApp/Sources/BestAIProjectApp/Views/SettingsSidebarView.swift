import SwiftUI
import UniformTypeIdentifiers

struct SettingsSidebarView: View {
    @ObservedObject var appState: AppState
    @State private var backendURLString: String
    @State private var pythonPath: String
    @State private var showingPythonImporter = false

    init(appState: AppState) {
        self.appState = appState
        _backendURLString = State(initialValue: appState.backendConfiguration.baseURL.absoluteString)
        _pythonPath = State(initialValue: appState.backendConfiguration.pythonInterpreter?.path ?? "")
    }

    private var numPredictBinding: Binding<String> {
        Binding<String>(
            get: { appState.settings.numPredict.map(String.init) ?? "" },
            set: { newValue in
                let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                appState.settings.numPredict = Int(trimmed)
            }
        )
    }

    private var seedBinding: Binding<String> {
        Binding<String>(
            get: { appState.settings.seed.map(String.init) ?? "" },
            set: { newValue in
                let trimmed = newValue.trimmingCharacters(in: .whitespaces)
                appState.settings.seed = Int(trimmed)
            }
        )
    }

    var body: some View {
        Form {
            Section("Backend") {
                TextField("Base URL", text: $backendURLString)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(applyBackendURL)
                HStack {
                    Button("Apply URL", action: applyBackendURL)
                    Button("Refresh", action: { Task { await appState.refreshHealth() } })
                }
                if let status = appState.healthStatus {
                    VStack(alignment: .leading, spacing: 4) {
                        Label(status.ok ? "Backend reachable" : "Backend unavailable", systemImage: status.ok ? "checkmark.seal" : "exclamationmark.triangle")
                            .symbolVariant(.fill)
                            .foregroundStyle(status.ok ? Color.green : Color.orange)
                        if let model = status.model {
                            Text("Active model: \(model)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if let host = status.ollama {
                            Text("Ollama host: \(host)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Section("Models") {
                if appState.models.isEmpty {
                    Text("No models returned yet.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Select model", selection: $appState.selectedModel) {
                        ForEach(appState.models) { model in
                            Text(model.name).tag(model.name)
                        }
                    }
                    .onChange(of: appState.selectedModel) { tag in
                        guard !tag.isEmpty else { return }
                        Task { await appState.applyModel(tag: tag) }
                    }
                }
                Button("Refresh models") {
                    Task { await appState.refreshModels() }
                }
            }

            Section("Prompts") {
                TextField("System prompt", text: $appState.systemPrompt, axis: .vertical)
                    .lineLimit(3, reservesSpace: true)
                TextField("Developer prompt", text: $appState.developerPrompt, axis: .vertical)
                    .lineLimit(4, reservesSpace: true)
            }

            Section("Generation settings") {
                Toggle("Enable tool calls", isOn: $appState.toolsEnabled)
                Toggle("Dynamic context", isOn: $appState.settings.dynamicContext)
                Stepper(value: $appState.settings.maxContext, in: 4096...80_000, step: 1024) {
                    Text("Max context: \(appState.settings.maxContext)")
                }
                Stepper(value: $appState.settings.numContext, in: 4096...80_000, step: 512) {
                    Text("Working context: \(appState.settings.numContext)")
                }
                VStack(alignment: .leading) {
                    Text("Temperature: \(String(format: "%.2f", appState.settings.temperature))")
                    Slider(value: $appState.settings.temperature, in: 0...1, step: 0.05)
                }
                VStack(alignment: .leading) {
                    Text("Top P: \(String(format: "%.2f", appState.settings.topP))")
                    Slider(value: $appState.settings.topP, in: 0...1, step: 0.05)
                }
                Stepper(value: $appState.settings.topK, in: 10...200, step: 5) {
                    Text("Top K: \(appState.settings.topK)")
                }
                TextField("Max tokens (leave blank for default)", text: numPredictBinding)
                    .textFieldStyle(.roundedBorder)
                TextField("Seed (optional)", text: seedBinding)
                    .textFieldStyle(.roundedBorder)
            }

            Section("Python backend") {
                Toggle("Launch Python backend on start", isOn: $appState.backendConfiguration.autoLaunchBackend)
                HStack {
                    TextField("Interpreter path", text: $pythonPath)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(applyPythonPath)
                    Button("Browse…") { showingPythonImporter = true }
                }
                HStack {
                    Button(appState.pythonManager.isRunning ? "Restart" : "Launch") {
                        if appState.pythonManager.isRunning {
                            appState.stopPythonBackend()
                            appState.startPythonBackend()
                        } else {
                            applyPythonPath()
                            appState.startPythonBackend()
                        }
                    }
                    Button("Stop") {
                        appState.stopPythonBackend()
                    }
                    .disabled(!appState.pythonManager.isRunning)
                }
                if let error = appState.pythonManager.lastError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                if !appState.pythonManager.logs.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Recent logs")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ScrollView {
                            VStack(alignment: .leading, spacing: 2) {
                                ForEach(appState.pythonManager.logs.suffix(8)) { line in
                                    Text("\(line.timestamp.formatted(date: .omitted, time: .shortened))  •  \(line.message)")
                                        .font(.caption2)
                                        .foregroundStyle(line.isError ? Color.red : Color.secondary)
                                }
                            }
                        }
                        .frame(maxHeight: 120)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear {
            backendURLString = appState.backendConfiguration.baseURL.absoluteString
            pythonPath = appState.backendConfiguration.pythonInterpreter?.path ?? ""
        }
        .onReceive(appState.$backendConfiguration) { config in
            backendURLString = config.baseURL.absoluteString
            pythonPath = config.pythonInterpreter?.path ?? ""
        }
        .fileImporter(isPresented: $showingPythonImporter, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let first = urls.first else { return }
                pythonPath = first.path
                appState.setPythonInterpreter(first)
            case .failure(let error):
                appState.lastError = error.localizedDescription
            }
        }
    }

    private func applyBackendURL() {
        guard let url = URL(string: backendURLString.trimmingCharacters(in: .whitespaces)) else { return }
        appState.updateBackendURL(url)
    }

    private func applyPythonPath() {
        let trimmed = pythonPath.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else {
            appState.setPythonInterpreter(nil)
            return
        }
        let url = URL(fileURLWithPath: trimmed)
        appState.setPythonInterpreter(url)
    }
}
