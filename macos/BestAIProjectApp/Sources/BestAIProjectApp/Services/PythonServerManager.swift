import Foundation

@MainActor
final class PythonServerManager: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var logs: [PythonServerLogLine] = []
    @Published private(set) var lastError: String?

    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var stdoutTask: Task<Void, Never>?
    private var stderrTask: Task<Void, Never>?

    let projectRoot: URL

    init(projectRoot: URL) {
        self.projectRoot = projectRoot
    }

    func start(pythonInterpreter: URL) async {
        guard process == nil else { return }
        lastError = nil

        let process = Process()
        process.currentDirectoryURL = projectRoot
        process.executableURL = pythonInterpreter
        process.arguments = ["server.py"]
        process.environment = ProcessInfo.processInfo.environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
            self.process = process
            self.stdoutPipe = stdoutPipe
            self.stderrPipe = stderrPipe
            isRunning = true
            appendLog("Launched python server.py using \(pythonInterpreter.path)", isError: false)
            observe(pipe: stdoutPipe, isError: false)
            observe(pipe: stderrPipe, isError: true)

            process.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    self?.handleTermination(status: Int(proc.terminationStatus))
                }
            }
        } catch {
            appendLog("Failed to launch python: \(error.localizedDescription)", isError: true)
            lastError = error.localizedDescription
            cleanup()
        }
    }

    func stop() {
        guard let process else { return }
        process.terminationHandler = nil
        process.interrupt()
        process.terminate()
        appendLog("Stopping python backend…", isError: false)
        cleanup()
    }

    private func handleTermination(status: Int) {
        if status != 0 {
            lastError = "Python backend exited with code \(status)"
            appendLog("Python backend exited with code \(status)", isError: true)
        } else {
            appendLog("Python backend exited", isError: false)
        }
        cleanup()
    }

    private func observe(pipe: Pipe, isError: Bool) {
        let handle = pipe.fileHandleForReading
        let task = Task { [weak self] in
            do {
                for try await line in handle.bytes.lines {
                    guard !Task.isCancelled else { break }
                    await MainActor.run {
                        self?.appendLog(line, isError: isError)
                    }
                }
            } catch {
                await MainActor.run {
                    self?.appendLog("Failed to read \(isError ? "stderr" : "stdout"): \(error.localizedDescription)", isError: true)
                }
            }
        }

        if isError {
            stderrTask = task
        } else {
            stdoutTask = task
        }
    }

    private func appendLog(_ message: String, isError: Bool) {
        logs.append(PythonServerLogLine(timestamp: Date(), message: message, isError: isError))
        if logs.count > 500 {
            logs.removeFirst(logs.count - 500)
        }
    }

    private func cleanup() {
        stdoutTask?.cancel()
        stderrTask?.cancel()
        stdoutTask = nil
        stderrTask = nil
        stdoutPipe = nil
        stderrPipe = nil
        process = nil
        isRunning = false
    }
}
