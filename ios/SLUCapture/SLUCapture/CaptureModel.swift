import AVFoundation
import Combine
import SwiftUI

/// Orchestrates pairing, synced record/stop, metadata, retake, calibration and delivery.
@MainActor
final class CaptureModel: ObservableObject {
    @Published var role: Role = Role(rawValue: UserDefaults.standard.string(forKey: "role") ?? "") ?? .leader {
        didSet { UserDefaults.standard.set(role.rawValue, forKey: "role"); peers.start(role: role) }
    }
    @Published var game: String = UserDefaults.standard.string(forKey: "game") ?? "kairos" {
        didSet { UserDefaults.standard.set(game, forKey: "game") }
    }
    @Published var move: String = UserDefaults.standard.string(forKey: "move") ?? "idle_loop" {
        didSet { UserDefaults.standard.set(move, forKey: "move") }
    }
    @Published var calibrationMode = false
    @Published var countdown: Double?
    @Published var lastTakeId: String?
    @Published var banner: String?
    @Published var faceMode = false

    let peers = PeerSync()
    let cam = CaptureEngine()
    let pose = PoseTracker()
    let store = TakeStore()
    private let chirp = ChirpPlayer()
    private var current: (id: String, take: Int, calib: Bool)?
    private var timer: Timer?
    private var bag = Set<AnyCancellable>()

    init() {
        // forward child state changes so SwiftUI re-renders
        for pub in [peers.objectWillChange.eraseToAnyPublisher(), cam.objectWillChange.eraseToAnyPublisher(),
                    store.objectWillChange.eraseToAnyPublisher(), pose.objectWillChange.eraseToAnyPublisher()] {
            pub.receive(on: RunLoop.main).sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
        }
        cam.frameTap = { [weak self] sb in
            guard let self else { return }
            if self.role == .leader { self.pose.process(sb) }
        }
        cam.configure()
        peers.onMessage = { [weak self] m in self?.handle(m) }
        peers.start(role: role)
    }

    var takeNumber: Int { store.nextTake(game: calibrationMode ? "_calib" : game, move: calibrationMode ? "calib" : move) }
    var moveInfo: Move? { store.games.first { $0.id == game }?.moves.first { $0.id == move } }

    // MARK: Leader actions
    func recordTapped() {
        if cam.isRecording { stopAll(); return }
        let g = calibrationMode ? "_calib" : game, m = calibrationMode ? "calib" : move
        if calibrationMode { store.setupId = String(format: "setup_%03d", Int(Date().timeIntervalSince1970) % 100000) }
        let t = store.nextTake(game: g, move: m)
        peers.send(.arm(game: g, move: m, take: t, setupId: store.setupId, calibration: calibrationMode))
        let startShared = peers.sharedNow + 1.0
        peers.send(.start(atShared: startShared))
        begin(game: g, move: m, take: t, startShared: startShared)
        chirp.play(atHost: peers.toLocal(startShared) + 0.5)
        if calibrationMode {  // ~20 s guided calibration, auto-stop
            DispatchQueue.main.asyncAfter(deadline: .now() + 21) { [weak self] in
                if self?.cam.isRecording == true { self?.stopAll() }
            }
        }
    }

    func stopAll() {
        let at = peers.sharedNow + 0.3
        peers.send(.stop(atShared: at))
        cam.stop(atHost: peers.toLocal(at))
    }

    func retakeTapped() {
        guard let id = lastTakeId else { return }
        store.markBad(takeId: id, role: role)
        peers.send(.markBad(takeId: id))
        banner = "Marked \(id.suffix(4)) as retake"
    }

    // MARK: both phones
    private func handle(_ m: Msg) {
        switch m {
        case .arm(let g, let mv, let t, let setup, let cal):
            store.setupId = setup; calibrationMode = cal
            if !cal { game = g; move = mv }
            current = (store.takeId(g, mv, t), t, cal)
        case .start(let at):
            guard let c = current else { return }
            let parts = c.id.components(separatedBy: "__")
            begin(game: parts[0], move: parts[1], take: c.take, startShared: at)
        case .stop(let at):
            cam.stop(atHost: peers.toLocal(at))
        case .markBad(let id):
            store.markBad(takeId: id, role: role)
            banner = "Marked \(id.suffix(4)) as retake"
        case .status(let s): banner = s
        default: break
        }
    }

    private func begin(game g: String, move m: String, take t: Int, startShared: Double) {
        let id = store.takeId(g, m, t)
        current = (id, t, g == "_calib")
        let base = store.pendingDir.appendingPathComponent("\(id)__\(role.rawValue)")
        let localStart = peers.toLocal(startShared)
        countdown = 1.0
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] tm in
            Task { @MainActor in
                guard let self else { return }
                let left = localStart - HostClock.now
                self.countdown = left > 0 ? left : nil
                if left <= 0 { tm.invalidate() }
            }
        }
        if role == .leader { pose.beginLog(host: localStart) }
        cam.record(to: base.appendingPathExtension("mov"), startingAt: localStart) { [weak self] url, firstHost in
            guard let self, let url else { self?.banner = "Recording failed"; return }
            self.store.commitTake(game: g, move: m, take: t)
            let offset = self.role == .leader ? 0 : self.peers.offset
            var meta: [String: Any] = [
                "game": g, "move": m, "take": t, "role": self.role.rawValue, "takeId": id,
                "sharedStart": firstHost + offset, "fps": self.cam.fps, "lens": "wide_1x",
                "setupId": self.store.setupId, "clockRttMs": self.peers.rttMs, "device": UIDevice.current.name,
                "camera": self.cam.lensInfo, "app": "SLU Capture 2.0", "bad": false]
            if g == "_calib" { meta["calibration"] = true }
            let metaURL = base.appendingPathExtension("json")
            self.store.writeJSON(meta, to: metaURL)
            var files = [url, metaURL]
            if self.role == .leader {
                let frames = self.pose.endLog().map { f -> [String: Any] in
                    var f = f; f["t"] = ((f["t"] as? Double) ?? 0) - (firstHost - localStart); return f }
                let v3 = self.store.pendingDir.appendingPathComponent("\(id)__leader.vision3d.json")
                self.store.writeJSON(["frames": frames], to: v3)
                files.append(v3)
            }
            self.lastTakeId = id
            self.banner = "\(id) saved"
            self.store.deliver(files: files)
        }
    }
}
