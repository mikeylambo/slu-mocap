import AVFoundation

/// 80 ms 2->6 kHz Hann-windowed sweep (matches pipeline/ingest.py CHIRP) played on the Leader
/// at a scheduled host time. Both phones' mics record it -> audio cross-check of sync.
final class ChirpPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var buffer: AVAudioPCMBuffer?

    init() {
        let sr = 48000.0
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1)!
        let n = AVAudioFrameCount(0.08 * sr)
        let b = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
        b.frameLength = n
        let k = (6000.0 - 2000.0) / 0.08
        for i in 0..<Int(n) {
            let t = Double(i) / sr
            let w = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(n - 1))
            b.floatChannelData![0][i] = Float(0.9 * w * sin(2 * .pi * (2000 * t + 0.5 * k * t * t)))
        }
        buffer = b
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: fmt)
    }

    func play(atHost hostSeconds: Double) {
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .videoRecording,
                                                         options: [.defaultToSpeaker, .mixWithOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        if !engine.isRunning { try? engine.start() }
        guard let b = buffer else { return }
        let when = AVAudioTime(hostTime: AVAudioTime.hostTime(forSeconds: hostSeconds))
        node.scheduleBuffer(b, at: when, options: [])
        node.play()
    }
}
