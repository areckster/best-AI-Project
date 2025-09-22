import Foundation

private struct ChatStreamRequest: Encodable {
    var messages: [MessagePayload]
    var settings: ChatSettings
    var system: String
    var developer: String
    var tools: Bool
}

actor ChatService {
    private var baseURL: URL
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private let session: URLSession

    init(baseURL: URL) {
        self.baseURL = baseURL
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.sortedKeys]
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60 * 5
        configuration.timeoutIntervalForResource = 60 * 10
        configuration.httpAdditionalHeaders = ["Accept": "text/event-stream"]
        self.session = URLSession(configuration: configuration)
    }

    func update(baseURL: URL) {
        self.baseURL = baseURL
    }

    func health() async throws -> HealthStatus {
        let url = baseURL.appendingPathComponent("api/health")
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
            throw URLError(.badServerResponse)
        }
        return try decoder.decode(HealthStatus.self, from: data)
    }

    func fetchModels() async throws -> [ModelSummary] {
        let url = baseURL.appendingPathComponent("api/models")
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
            throw URLError(.badServerResponse)
        }
        let payload = try decoder.decode(ModelsResponse.self, from: data)
        return payload.models
    }

    func setModel(tag: String) async throws -> String {
        let url = baseURL.appendingPathComponent("api/models/set")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = ["model": tag]
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
            throw URLError(.badServerResponse)
        }
        let object = try decoder.decode([String: String].self, from: data)
        return object["model"] ?? tag
    }

    func stream(
        history: [MessagePayload],
        settings: ChatSettings,
        systemPrompt: String,
        developerPrompt: String,
        toolsEnabled: Bool
    ) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        let requestBody = ChatStreamRequest(
            messages: history,
            settings: settings,
            system: systemPrompt,
            developer: developerPrompt,
            tools: toolsEnabled
        )

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appendingPathComponent("api/chat/stream"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try encoder.encode(requestBody)

                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
                        throw URLError(.badServerResponse)
                    }

                    var dataLines: [String] = []

                    func flushBuffer() throws {
                        guard !dataLines.isEmpty else { return }
                        let payload = dataLines.joined(separator: "\n")
                        dataLines.removeAll(keepingCapacity: true)

                        if payload == "[DONE]" || payload.lowercased() == "done" {
                            continuation.finish()
                            return
                        }

                        let eventData = Data(payload.utf8)
                        let event = try decoder.decode(ChatStreamEvent.self, from: eventData)
                        continuation.yield(event)
                    }

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        if trimmed.isEmpty {
                            try flushBuffer()
                            continue
                        }

                        if trimmed.hasPrefix("data:") {
                            let chunk = trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)
                            if chunk.isEmpty { continue }
                            dataLines.append(String(chunk))
                        }
                    }

                    try flushBuffer()
                    continuation.finish()
                } catch {
                    if Task.isCancelled { return }
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
