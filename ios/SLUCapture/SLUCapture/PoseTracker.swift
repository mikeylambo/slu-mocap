import AVFoundation
import Vision

/// Live Apple Vision 3D body pose (iOS 17) on the Leader at ~10 Hz:
/// draws a framing skeleton, warns when limbs leave the frame, and logs 3D joints per take
/// (vision3d.json) for the pipeline's single-view fallback.
struct Joint2D: Identifiable { let id: String; let p: CGPoint }

final class PoseTracker: ObservableObject {
    @Published var joints: [String: CGPoint] = [:]       // normalised, top-left origin
    @Published var outOfFrame: [String] = []
    private var last: Double = 0
    private var busy = false
    private let q = DispatchQueue(label: "slu.pose")
    private(set) var log: [[String: Any]] = []
    var logging = false
    var logStartHost: Double = 0

    static let bones: [(String, String)] = [
        ("head_joint", "center_shoulder_joint"), ("center_shoulder_joint", "spine_joint"), ("spine_joint", "root"),
        ("center_shoulder_joint", "left_shoulder_1_joint"), ("left_shoulder_1_joint", "left_forearm_joint"),
        ("left_forearm_joint", "left_hand_joint"), ("center_shoulder_joint", "right_shoulder_1_joint"),
        ("right_shoulder_1_joint", "right_forearm_joint"), ("right_forearm_joint", "right_hand_joint"),
        ("root", "left_upLeg_joint"), ("left_upLeg_joint", "left_leg_joint"), ("left_leg_joint", "left_foot_joint"),
        ("root", "right_upLeg_joint"), ("right_upLeg_joint", "right_leg_joint"), ("right_leg_joint", "right_foot_joint")]
    static let limbs = ["head_joint", "left_hand_joint", "right_hand_joint", "left_foot_joint", "right_foot_joint"]

    func process(_ sb: CMSampleBuffer) {
        let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
        guard t - last > 0.1, !busy, let px = CMSampleBufferGetImageBuffer(sb) else { return }
        last = t; busy = true
        q.async { [weak self] in
            defer { self?.busy = false }
            let req = VNDetectHumanBodyPose3DRequest()
            let h = VNImageRequestHandler(cvPixelBuffer: px, orientation: .up)
            try? h.perform([req])
            guard let self, let obs = req.results?.first else {
                DispatchQueue.main.async { self?.joints = [:] }; return
            }
            var pts: [String: CGPoint] = [:]
            var j3: [String: [Float]] = [:]
            for name in obs.availableJointNames {
                if let p = try? obs.pointInImage(name) {
                    pts[name.rawValue.rawValue] = CGPoint(x: p.x, y: 1 - p.y)
                }
                if let r = try? obs.recognizedPoint(name) {
                    // camera space: root position + model-space joint offset
                    let m = simd_mul(obs.cameraOriginMatrix.inverse, r.position)
                    j3[name.rawValue.rawValue] = [m.columns.3.x, m.columns.3.y, m.columns.3.z]
                }
            }
            let out = Self.limbs.filter { n in
                guard let p = pts[n] else { return true }
                return p.x < 0.02 || p.x > 0.98 || p.y < 0.02 || p.y > 0.98
            }
            if self.logging { self.log.append(["t": t - self.logStartHost, "joints": j3]) }
            DispatchQueue.main.async { self.joints = pts; self.outOfFrame = pts.isEmpty ? [] : out }
        }
    }

    func beginLog(host: Double) { q.async { self.log = []; self.logStartHost = host; self.logging = true } }
    func endLog() -> [[String: Any]] { q.sync { logging = false; return log } }
}
