import Foundation

enum ChatRole: String, Codable, CaseIterable, Identifiable {
    case system
    case user
    case assistant

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "System"
        case .user: return "You"
        case .assistant: return "Assistant"
        }
    }
}

struct ChatMessage: Identifiable, Codable, Equatable {
    let id: UUID
    var role: ChatRole
    var content: String
    var createdAt: Date
    var isStreaming: Bool
    var metadata: [String: String]

    init(
        id: UUID = UUID(),
        role: ChatRole,
        content: String,
        createdAt: Date = Date(),
        isStreaming: Bool = false,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
        self.isStreaming = isStreaming
        self.metadata = metadata
    }

    var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }

    var payload: MessagePayload {
        MessagePayload(role: role.rawValue, content: content)
    }
}

struct MessagePayload: Codable, Equatable {
    var role: String
    var content: String
}

struct ChatSettings: Codable, Equatable {
    var dynamicContext: Bool = true
    var maxContext: Int = 40_000
    var numContext: Int = 8_192
    var temperature: Double = 0.9
    var topP: Double = 0.9
    var topK: Int = 100
    var numPredict: Int?
    var seed: Int?

    enum CodingKeys: String, CodingKey {
        case dynamicContext = "dynamic_ctx"
        case maxContext = "max_ctx"
        case numContext = "num_ctx"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case numPredict = "num_predict"
        case seed
    }
}

enum StreamEventType: String, Decodable {
    case delta
    case done
    case error
    case toolCalls = "tool_calls"
    case toolResult = "tool_result"
    case gateWarning = "gate_warning"
    case thinking
    case message
    case usage
    case unknown

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        self = StreamEventType(rawValue: raw) ?? .unknown
    }
}

struct ChatStreamEvent: Decodable, Identifiable {
    let id = UUID()
    let type: StreamEventType
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let dictionary = try container.decode([String: JSONValue].self)
        raw = dictionary
        if let explicitType = dictionary["type"]?.stringValue,
           let mapped = StreamEventType(rawValue: explicitType) {
            type = mapped
        } else if dictionary["delta"] != nil {
            type = .delta
        } else {
            type = .unknown
        }
    }

    var delta: String? { raw["delta"]?.stringValue }
    var message: String? { raw["message"]?.stringValue ?? raw["content"]?.stringValue }
    var role: String? { raw["role"]?.stringValue }
    var usage: [String: JSONValue]? { raw["usage"]?.dictionaryValue }

    var errorDescription: String? {
        guard type == .error else { return nil }
        return raw["message"]?.stringValue
    }
}

enum JSONValue: Equatable {
    case string(String)
    case number(Double)
    case object([String: JSONValue])
    case array([JSONValue])
    case bool(Bool)
    case null
}

extension JSONValue: Decodable {
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let double = try? container.decode(Double.self) {
            self = .number(double)
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let array = try? container.decode([JSONValue].self) {
            self = .array(array)
        } else if let dictionary = try? container.decode([String: JSONValue].self) {
            self = .object(dictionary)
        } else {
            throw DecodingError.typeMismatch(
                JSONValue.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value")
            )
        }
    }
}

extension JSONValue {
    var stringValue: String? {
        if case let .string(value) = self { return value }
        return nil
    }

    var doubleValue: Double? {
        if case let .number(value) = self { return value }
        return nil
    }

    var intValue: Int? {
        if case let .number(value) = self { return Int(value) }
        return nil
    }

    var boolValue: Bool? {
        if case let .bool(value) = self { return value }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case let .array(value) = self { return value }
        return nil
    }

    var dictionaryValue: [String: JSONValue]? {
        if case let .object(value) = self { return value }
        return nil
    }
}

struct HealthStatus: Decodable {
    var ok: Bool
    var ollama: String?
    var model: String?
}

struct ModelsResponse: Decodable {
    var models: [ModelSummary]
}

struct ModelSummary: Decodable, Identifiable, Hashable {
    var id: String { name }
    var name: String
    var size: String
    var details: ModelDetails?
    var modifiedAt: Date?

    enum CodingKeys: String, CodingKey {
        case name
        case size
        case details
        case modifiedAt = "modified_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        size = try container.decodeIfPresent(String.self, forKey: .size) ?? ""
        details = try container.decodeIfPresent(ModelDetails.self, forKey: .details)
        if let modifiedString = try container.decodeIfPresent(String.self, forKey: .modifiedAt) {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            modifiedAt = formatter.date(from: modifiedString) ?? ISO8601DateFormatter().date(from: modifiedString)
        } else {
            modifiedAt = nil
        }
    }
}

struct ModelDetails: Decodable, Hashable {
    var parameterSize: String?
    var quantizationLevel: String?

    enum CodingKeys: String, CodingKey {
        case parameterSize = "parameter_size"
        case quantizationLevel = "quantization_level"
    }
}

struct BackendConfiguration: Codable, Equatable {
    var baseURL: URL
    var autoLaunchBackend: Bool
    var pythonInterpreter: URL?

    static let `default` = BackendConfiguration(
        baseURL: URL(string: "http://127.0.0.1:8000")!,
        autoLaunchBackend: false,
        pythonInterpreter: nil
    )
}

struct PythonServerLogLine: Identifiable, Equatable {
    let id = UUID()
    let timestamp: Date
    let message: String
    let isError: Bool
}
