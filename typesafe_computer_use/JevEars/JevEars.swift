// JevEars: macOS speech recognition for voice mode, push to talk.
//
// The microphone stays off until voice mode sends SIGUSR1 (the key went down), and goes off again
// at SIGUSR2 (the key came up). What it hears is appended to a file, one JSON object per line:
//   {"pid": 123}                              first, so the caller can signal and end it
//   {"ready": true, "on_device": true}        permissions granted, waiting for the key
//   {"text": "Open Git", "final": false}      while the key is held, revised as more is heard
//   {"text": "Open GitHub.", "final": true}   after the key comes up: the whole utterance, maybe ""
//   {"error": "..."}                          and then it exits
//
// It runs as its own app, launched with `open`, so macOS asks for and records its microphone and
// speech recognition permissions, rather than those of the terminal, which cannot declare them.
//
// Arguments: --out FILE (must exist) --parent PID --words FILE (one per line)

import AppKit
import AVFoundation
import Speech

let finishTimeout = 2.0  // seconds to wait, after the key comes up, for the recognizer's last word

struct Options {
    var out = ""
    var parent: Int32 = 0
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
    var utterance = 0  // counts holds, so a finished one's late callbacks are ignored
    var open = false  // an utterance is being heard or finished, and has not been reported yet
    var text = ""

    init(recognizer: SFSpeechRecognizer) {
        self.recognizer = recognizer
        let input = engine.inputNode
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { [weak self] buffer, _ in
            guard let self else { return }
            self.lock.lock()
            let request = self.request
            self.lock.unlock()
            request?.append(buffer)
        }
        engine.prepare()
    }

    /// The key went down: turn the microphone on and start a new utterance.
    func hold() {
        if open {
            finish()  // the last one was still finishing
        }
        utterance += 1
        let mine = utterance
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        request.contextualStrings = options.words
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        text = ""
        open = true
        lock.lock()
        self.request = request
        lock.unlock()
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            DispatchQueue.main.async { self?.heard(mine, result, error) }
        }
        do {
            try engine.start()
        } catch {
            fail("could not open the microphone: \(error.localizedDescription)")
        }
    }

    /// The key came up: turn the microphone off, and report the utterance once the recognizer is done.
    func release() {
        guard open else { return }
        engine.stop()
        lock.lock()
        request?.endAudio()
        request = nil
        lock.unlock()
        let mine = utterance
        DispatchQueue.main.asyncAfter(deadline: .now() + finishTimeout) { [weak self] in
            if let self, self.utterance == mine, self.open {
                self.finish()
            }
        }
    }

    func heard(_ mine: Int, _ result: SFSpeechRecognitionResult?, _ error: Error?) {
        guard mine == utterance, open else { return }
        if let result {
            let now = result.bestTranscription.formattedString
            if now != text {
                text = now
                emit(["text": text, "final": false])
            }
        }
        // "No speech detected" arrives as an error: the utterance is over either way.
        if result?.isFinal == true || error != nil {
            finish()
        }
    }

    func finish() {
        if engine.isRunning {
            engine.stop()  // the recognizer gave up while the key was still held
        }
        lock.lock()
        request = nil
        lock.unlock()
        task?.cancel()
        task = nil
        open = false
        emit(["text": text, "final": true])
    }
}

var ears: Ears?
var signalSources: [DispatchSourceSignal] = []

func onSignal(_ number: Int32, _ handler: @escaping () -> Void) {
    signal(number, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
    source.setEventHandler(handler: handler)
    source.resume()
    signalSources.append(source)
}

func listen() {
    guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")) else {
        fail("macOS has no speech recognizer for en-US")
    }
    let ear = Ears(recognizer: recognizer)
    ears = ear
    onSignal(SIGUSR1) { ear.hold() }
    onSignal(SIGUSR2) { ear.release() }
    Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
        if options.parent > 0 && kill(options.parent, 0) != 0 {
            exit(0)  // voice mode is gone
        }
    }
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
