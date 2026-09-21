// JevEars: macOS speech recognition for voice mode.
//
// Listens to the microphone and appends what it hears to a file, one JSON object per line:
//   {"pid": 123}                              first, so the caller can end it
//   {"ready": true, "on_device": true}        listening
//   {"text": "Open GitHub", "final": false}   the open window, which later lines may revise
//   {"text": "Open GitHub.", "final": true}   the window closed after a pause; its words stay
//   {"error": "..."}                          and then it exits
//
// It runs as its own app, launched with `open`, so macOS asks for and records its microphone and
// speech recognition permissions, rather than those of the terminal, which cannot declare them.
//
// Arguments: --out FILE (must exist) --parent PID --silence SECONDS --words FILE (one per line)

import AppKit
import AVFoundation
import Speech

struct Options {
    var out = ""
    var parent: Int32 = 0
    var silence = 0.9
    var words: [String] = []
}

func parseOptions() -> Options {
    var options = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        let value = args.next() ?? ""
        switch arg {
        case "--out": options.out = value
        case "--parent": options.parent = Int32(value) ?? 0
        case "--silence": options.silence = Double(value) ?? options.silence
        case "--words":
            let text = (try? String(contentsOfFile: value, encoding: .utf8)) ?? ""
            options.words = text.split(separator: "\n").map(String.init)
        default: break
        }
    }
    return options
}

let options = parseOptions()
let output: FileHandle = {
    guard let handle = FileHandle(forWritingAtPath: options.out) else { exit(2) }
    handle.seekToEndOfFile()
    return handle
}()

func emit(_ fields: [String: Any]) {
    guard var line = try? JSONSerialization.data(withJSONObject: fields) else { return }
    line.append(0x0A)
    output.write(line)
}

func fail(_ message: String) -> Never {
    emit(["error": message])
    exit(1)
}

final class Ears {
    let recognizer: SFSpeechRecognizer
    let engine = AVAudioEngine()
    let lock = NSLock()
    var request: SFSpeechAudioBufferRecognitionRequest?  // guarded by lock: the audio thread appends to it
    var task: SFSpeechRecognitionTask?
    var window = 0  // counts recognition requests, so a closed one's late callbacks are ignored
    var text = ""
    var changed = Date()
    var begun = Date()
    var quickFailures = 0

    init(recognizer: SFSpeechRecognizer) {
        self.recognizer = recognizer
    }

    func start() throws {
        let input = engine.inputNode
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { [weak self] buffer, _ in
            guard let self else { return }
            self.lock.lock()
            let request = self.request
            self.lock.unlock()
            request?.append(buffer)
        }
        engine.prepare()
        try engine.start()
        begin()
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in self?.tick() }
    }

    /// Open a new window: a fresh recognition request takes the audio from here on.
    func begin() {
        window += 1
        let mine = window
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        request.contextualStrings = options.words
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        text = ""
        begun = Date()
        lock.lock()
        self.request = request
        lock.unlock()
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            DispatchQueue.main.async { self?.heard(mine, result, error) }
        }
    }

    func heard(_ mine: Int, _ result: SFSpeechRecognitionResult?, _ error: Error?) {
        guard mine == window else { return }
        if let result {
            let now = result.bestTranscription.formattedString
            if now != text {
                text = now
                changed = Date()
                quickFailures = 0
                emit(["text": text, "final": false])
            }
        }
        if let error, text.isEmpty {
            // Silence ends a request with "no speech detected", which is fine; an error at once, again
            // and again, means the recognizer cannot run.
            quickFailures = Date().timeIntervalSince(begun) < 1 ? quickFailures + 1 : 0
            if quickFailures >= 5 {
                fail("speech recognition keeps failing: \(error.localizedDescription)")
            }
        }
        if result?.isFinal == true || error != nil {
            close()
        }
    }

    /// Close the open window: its words become final, and a new request takes the audio.
    func close() {
        if !text.isEmpty {
            emit(["text": text, "final": true])
        }
        let old = task
        lock.lock()
        request?.endAudio()
        lock.unlock()
        begin()
        old?.cancel()
    }

    func tick() {
        if options.parent > 0 && kill(options.parent, 0) != 0 {
            exit(0)  // voice mode is gone
        }
        if !text.isEmpty && Date().timeIntervalSince(changed) >= options.silence {
            close()
        }
    }
}

var ears: Ears?

func listen() {
    guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")) else {
        fail("macOS has no speech recognizer for en-US")
    }
    let ear = Ears(recognizer: recognizer)
    do {
        try ear.start()
    } catch {
        fail("could not open the microphone: \(error.localizedDescription)")
    }
    ears = ear
    emit(["ready": true, "on_device": recognizer.supportsOnDeviceRecognition])
}

emit(["pid": Int(getpid())])
SFSpeechRecognizer.requestAuthorization { status in
    DispatchQueue.main.async {
        guard status == .authorized else {
            fail("speech recognition is not allowed: System Settings > Privacy & Security > Speech Recognition > JevEars")
        }
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                guard granted else {
                    fail("the microphone is not allowed: System Settings > Privacy & Security > Microphone > JevEars")
                }
                listen()
            }
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
app.run()
