import AVFoundation
import UIKit

/// Locked 60 fps capture on the 1x wide lens, written with AVAssetWriter so recording can start
/// on an exact host-clock instant (first frame with PTS >= startHostTime).
final class CaptureEngine: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published var isRecording = false
    @Published var locked = false
    @Published var formatText = ""
    @Published var error: String?

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
    var fps: Double = 60
    var resolution4K = false

    func configure() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            error = "No wide camera"; session.commitConfiguration(); return
        }
        session.addInput(input)
        device = dev
        if let mic = AVCaptureDevice.default(for: .audio), let mi = try? AVCaptureDeviceInput(device: mic),
           session.canAddInput(mi) { session.addInput(mi) }
        videoOut.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        videoOut.alwaysDiscardsLateVideoFrames = false
        videoOut.setSampleBufferDelegate(self, queue: q)
        audioOut.setSampleBufferDelegate(self, queue: q)
        if session.canAddOutput(videoOut) { session.addOutput(videoOut) }
        if session.canAddOutput(audioOut) { session.addOutput(audioOut) }
        selectFormat(dev)
        if let c = videoOut.connection(with: .video) {
            if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
            if #available(iOS 17.0, *), c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        }
        session.commitConfiguration()
        q.async { self.session.startRunning() }
    }

    /// Highest-res format (1080p or 4K) that supports 60 fps.
    private func selectFormat(_ dev: AVCaptureDevice) {
        let want = resolution4K ? 3840 : 1920
        let fmts = dev.formats.filter { f in
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return Int(d.width) == want && f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= fps }
                && CMFormatDescriptionGetMediaSubType(f.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        }
        guard let f = fmts.first ?? dev.formats.last(where: { $0.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 60 } })
        else { return }
        try? dev.lockForConfiguration()
        dev.activeFormat = f
        let dur = CMTime(value: 1, timescale: CMTimeScale(fps))
        dev.activeVideoMinFrameDuration = dur
        dev.activeVideoMaxFrameDuration = dur
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        dev.unlockForConfiguration()
        let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
        DispatchQueue.main.async { self.formatText = "\(d.width)x\(d.height) @ \(Int(self.fps)) · 1x · stab off" }
    }

    /// Tap-to-meter, then lock exposure, ISO, white balance and focus.
    func meterAndLock(at p: CGPoint) {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        if dev.isFocusPointOfInterestSupported { dev.focusPointOfInterest = p; dev.focusMode = .autoFocus }
        if dev.isExposurePointOfInterestSupported { dev.exposurePointOfInterest = p; dev.exposureMode = .autoExpose }
        if dev.isWhiteBalanceModeSupported(.autoWhiteBalance) { dev.whiteBalanceMode = .autoWhiteBalance }
        dev.unlockForConfiguration()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.lock() }
    }

    func lock() {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        // keep the metered exposure but cap shutter at 1/250 s for crisp limbs
        let maxDur = CMTime(value: 1, timescale: 250)
        let dur = CMTimeMinimum(dev.exposureDuration, maxDur)
        let gain = dev.exposureDuration.seconds / dur.seconds
        let iso = min(max(dev.iso * Float(gain), dev.activeFormat.minISO), dev.activeFormat.maxISO)
        dev.setExposureModeCustom(duration: dur, iso: iso, completionHandler: nil)
        if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
        if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
        dev.unlockForConfiguration()
        DispatchQueue.main.async { self.locked = true }
    }

    func unlock() {
        guard let dev = device else { return }
        try? dev.lockForConfiguration()
        dev.exposureMode = .continuousAutoExposure
        if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { dev.whiteBalanceMode = .continuousAutoWhiteBalance }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        dev.unlockForConfiguration()
        locked = false
    }

    var lensInfo: [String: Any] {
        guard let d = device else { return [:] }
        return ["deviceType": d.deviceType.rawValue, "fovDeg": d.activeFormat.videoFieldOfView,
                "iso": d.iso, "exposureS": d.exposureDuration.seconds, "model": UIDevice.current.model]
    }

    /// Arm the writer; recording begins with the first frame at/after `hostTime`.
    func record(to url: URL, startingAt hostTime: Double, completion: @escaping (URL?, Double) -> Void) {
        q.async {
            try? FileManager.default.removeItem(at: url)
            guard let w = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
            let vs = self.videoOut.recommendedVideoSettingsForAssetWriter(writingTo: .mov)
            let vi = AVAssetWriterInput(mediaType: .video, outputSettings: vs)
            vi.expectsMediaDataInRealTime = true
            let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: self.audioOut.recommendedAudioSettingsForAssetWriter(writingTo: .mov))
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
