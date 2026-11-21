import AVFoundation
#if os(iOS)
import UIKit
#endif

/// Locked capture on the main wide lens (iPhone / iPad) or the built-in / attached camera (Mac),
/// written with AVAssetWriter so recording starts on an exact host-clock instant
/// (first frame with PTS >= startHostTime). Picks the best format for the requested fps and
/// falls back to the device's maximum (e.g. iPad Air 2: 1080p30, iPhones: 1080p240 slow-mo).
final class CaptureEngine: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published var isRecording = false
    @Published var locked = false
    @Published var formatText = ""
    @Published var error: String?
    @Published private(set) var fps: Double = 60

    private(set) var device: AVCaptureDevice?
    private let videoOut = AVCaptureVideoDataOutput()
    private let audioOut = AVCaptureAudioDataOutput()
    private let q = DispatchQueue(label: "slu.capture", qos: .userInteractive)
    private var writer: AVAssetWriter?
    private var vIn: AVAssetWriterInput?
    private var aIn: AVAssetWriterInput?
    private var startHost: Double = .infinity
    private var stopHost: Double = .infinity
    private var started = false
    private var finishing = false
    private(set) var firstFrameHost: Double = 0
    private var url: URL?
    private var completion: ((URL?, Double) -> Void)?
    var frameTap: ((CMSampleBuffer) -> Void)?
    var resolution4K = false

    func configure() {
        session.beginConfiguration()
        #if os(iOS)
        session.sessionPreset = .inputPriority
        let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        #else
        session.sessionPreset = .high
        let dev = AVCaptureDevice.default(for: .video)
        #endif
        guard let dev, let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            error = "No camera"; session.commitConfiguration(); return
        }
        session.addInput(input)
        device = dev
        if let mic = AVCaptureDevice.default(for: .audio), let mi = try? AVCaptureDeviceInput(device: mic),
           session.canAddInput(mi) { session.addInput(mi) }
        videoOut.alwaysDiscardsLateVideoFrames = false
        videoOut.setSampleBufferDelegate(self, queue: q)
        audioOut.setSampleBufferDelegate(self, queue: q)
        if session.canAddOutput(videoOut) { session.addOutput(videoOut) }
        if session.canAddOutput(audioOut) { session.addOutput(audioOut) }
        #if os(iOS)
        if let c = videoOut.connection(with: .video) {
            if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
            if #available(iOS 17.0, *) {
                if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
            } else if c.isVideoOrientationSupported {
                c.videoOrientation = .portrait
            }
        }
        #endif
        session.commitConfiguration()
        setFPS(60)
        q.async { self.session.startRunning() }
    }

    /// Choose the best format for `want` fps: prefer 1080p (or 4K), else the device maximum.
    func setFPS(_ want: Double) {
        guard let dev = device else { return }
        let widthPref = resolution4K ? 3840 : 1920
        func maxFPS(_ f: AVCaptureDevice.Format) -> Double { f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0 }
        func width(_ f: AVCaptureDevice.Format) -> Int32 { CMVideoFormatDescriptionGetDimensions(f.formatDescription).width }
        let fmts = dev.formats.filter { width($0) >= 1280 }
        let exact = fmts.filter { maxFPS($0) >= want }
        let pick = exact.first { width($0) == widthPref } ?? exact.max { width($0) < width($1) }
            ?? fmts.max { (maxFPS($0), Double(width($0))) < (maxFPS($1), Double(width($1))) }
        guard let f = pick else { return }
        let rate = min(want, maxFPS(f))
        do {
            try dev.lockForConfiguration()
            dev.activeFormat = f
            let dur = CMTime(value: 1, timescale: CMTimeScale(rate))
            dev.activeVideoMinFrameDuration = dur
            dev.activeVideoMaxFrameDuration = dur
            #if os(iOS)
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
            #endif
            dev.unlockForConfiguration()
        } catch { return }
        let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
        DispatchQueue.main.async {
            self.fps = rate
            self.formatText = "\(d.width)x\(d.height) @ \(Int(rate))\(rate >= 120 ? " slow-mo" : "")"
            self.locked = false
        }
    }

    /// Tap-to-meter, then lock exposure, ISO, white balance and focus.
    func meterAndLock(at p: CGPoint) {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        if dev.isFocusPointOfInterestSupported { dev.focusPointOfInterest = p; dev.focusMode = .autoFocus }
        if dev.isExposurePointOfInterestSupported { dev.exposurePointOfInterest = p; dev.exposureMode = .autoExpose }
        dev.unlockForConfiguration()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.lock() }
    }

    func lock() {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        #if os(iOS)
        // keep the metered brightness, shutter <= 1/250 s (1/(fps) in slow-mo) for crisp limbs
        let cap = CMTime(value: 1, timescale: CMTimeScale(max(250, fps)))
        let dur = CMTimeMaximum(CMTimeMinimum(dev.exposureDuration, cap), dev.activeFormat.minExposureDuration)
        let gain = dev.exposureDuration.seconds / dur.seconds
        let iso = min(max(dev.iso * Float(gain), dev.activeFormat.minISO), dev.activeFormat.maxISO)
        if dev.isExposureModeSupported(.custom) { dev.setExposureModeCustom(duration: dur, iso: iso, completionHandler: nil) }
        if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
        #else
        if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }
        if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
        #endif
        if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
        dev.unlockForConfiguration()
        DispatchQueue.main.async { self.locked = true }
    }

    func unlock() {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
        if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { dev.whiteBalanceMode = .continuousAutoWhiteBalance }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        dev.unlockForConfiguration()
        locked = false
    }

    var lensInfo: [String: Any] {
        guard let d = device else { return [:] }
        var info: [String: Any] = ["deviceType": d.deviceType.rawValue, "name": d.localizedName, "model": Platform.model]
        #if os(iOS)
        info["fovDeg"] = d.activeFormat.videoFieldOfView
        info["iso"] = d.iso
        info["exposureS"] = d.exposureDuration.seconds
        #endif
        return info
    }

    /// Arm the writer; recording begins with the first frame at/after `hostTime`.
    func record(to url: URL, startingAt hostTime: Double, completion: @escaping (URL?, Double) -> Void) {
        q.async {
            try? FileManager.default.removeItem(at: url)
            guard let w = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
            let vs = self.videoOut.recommendedVideoSettingsForAssetWriter(writingTo: .mov)
            let vi = AVAssetWriterInput(mediaType: .video, outputSettings: vs)
            vi.expectsMediaDataInRealTime = true
            let ai = AVAssetWriterInput(mediaType: .audio,
                                        outputSettings: self.audioOut.recommendedAudioSettingsForAssetWriter(writingTo: .mov))
            ai.expectsMediaDataInRealTime = true
            if w.canAdd(vi) { w.add(vi) }
            if w.canAdd(ai) { w.add(ai) }
            self.writer = w; self.vIn = vi; self.aIn = ai; self.url = url
            self.startHost = hostTime; self.stopHost = .infinity
            self.started = false; self.finishing = false; self.completion = completion
            DispatchQueue.main.async { self.isRecording = true }
        }
    }

    func stop(atHost t: Double) { q.async { self.stopHost = t } }

    private func finish() {
        guard let w = writer, !finishing else { return }
        finishing = true
        vIn?.markAsFinished(); aIn?.markAsFinished()
        let first = firstFrameHost
        w.finishWriting { [weak self] in
            DispatchQueue.main.async {
                self?.isRecording = false
                self?.completion?(w.status == .completed ? self?.url : nil, first)
                self?.writer = nil
            }
        }
    }
}

extension CaptureEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from c: AVCaptureConnection) {
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let t = pts.seconds
        if output === videoOut { frameTap?(sb) }
        guard let w = writer, !finishing else { return }
        if output === videoOut {
            if !started {
                guard t >= startHost - 0.5 / fps else { return }
                w.startWriting()
                w.startSession(atSourceTime: pts)
                started = true
                firstFrameHost = t
            }
            if t >= stopHost { finish(); return }
            if vIn?.isReadyForMoreMediaData == true { vIn?.append(sb) }
        } else if started, aIn?.isReadyForMoreMediaData == true, t >= firstFrameHost {
            aIn?.append(sb)
        }
    }
}
