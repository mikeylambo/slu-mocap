import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var m: CaptureModel
    @State private var pickInbox = false
    @State private var showSettings = false

    var body: some View {
        ZStack {
            CameraPreview(session: m.cam.session) { p in m.cam.meterAndLock(at: p) }
                .ignoresSafeArea()
            FramingGuide().allowsHitTesting(false)
            if m.role == .leader { SkeletonOverlay(pose: m.pose).allowsHitTesting(false) }
            VStack(spacing: 0) {
                TopBar(showSettings: $showSettings)
                Spacer()
                if let c = m.countdown { Text(String(format: "%.1f", c)).font(.system(size: 64, weight: .thin, design: .rounded)) }
                if m.calibrationMode { CalibrationCoach() }
                Spacer()
                BottomBar()
            }
            .padding(.horizontal, 18)
        }
        .sheet(isPresented: $showSettings) { SettingsSheet(pickInbox: $pickInbox).environmentObject(m) }
        .fileImporter(isPresented: $pickInbox, allowedContentTypes: [.folder]) { r in
            if case .success(let url) = r { m.store.setInbox(url) }
        }
    }
}

private struct TopBar: View {
    @EnvironmentObject var m: CaptureModel
    @Binding var showSettings: Bool
    var body: some View {
        HStack(spacing: 10) {
            Pill(text: m.role.label, color: .white)
            Pill(text: m.peers.connected ? (m.role == .leader ? "\(m.peers.peerCount) CAMS PAIRED" : "PAIRED \(m.peers.syncQuality)") : "PAIRING…",
                 color: m.peers.connected ? .green : .orange)
            Pill(text: m.cam.locked ? "LOCKED" : "TAP TO LOCK", color: m.cam.locked ? .green : .yellow)
            if m.cam.fps >= 120 { Pill(text: "\(Int(m.cam.fps)) FPS", color: .pink) }
            Spacer()
            Button { showSettings = true } label: { Image(systemName: "gearshape.fill").font(.title3) }
        }
        .padding(.top, 8)
    }
}

private struct BottomBar: View {
    @EnvironmentObject var m: CaptureModel
    var body: some View {
        VStack(spacing: 14) {
            if !m.pose.outOfFrame.isEmpty && m.role == .leader {
                Text("OUT OF FRAME: " + m.pose.outOfFrame.map { $0.replacingOccurrences(of: "_joint", with: "")
                    .replacingOccurrences(of: "_", with: " ") }.joined(separator: ", ").uppercased())
                    .font(.caption.bold()).padding(8).background(.red.opacity(0.85), in: Capsule())
            }
            if let b = m.banner { Text(b).font(.footnote).foregroundStyle(.secondary) }
            if m.role == .leader {
                MovePicker()
                HStack(alignment: .center) {
                    Button("Retake") { m.retakeTapped() }
                        .buttonStyle(Glass()).disabled(m.lastTakeId == nil || m.cam.isRecording)
                    Spacer()
                    RecordButton(recording: m.cam.isRecording) { m.recordTapped() }
                    Spacer()
                    Text(String(format: "TAKE\n%03d", m.takeNumber)).font(.caption.monospaced())
                        .multilineTextAlignment(.center).frame(width: 76)
                }
            } else {
                Text(m.cam.isRecording ? "● RECORDING" : "Waiting for Leader")
                    .font(.headline).foregroundStyle(m.cam.isRecording ? .red : .secondary)
                    .padding(.bottom, 30)
            }
        }
        .padding(.bottom, 12)
    }
}

private struct MovePicker: View {
    @EnvironmentObject var m: CaptureModel
    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Menu {
                    ForEach(m.store.games) { g in Button(g.id.replacingOccurrences(of: "_", with: " ").uppercased()) {
                        m.game = g.id; m.move = g.moves.first?.id ?? "" } }
                } label: { Text(m.game.replacingOccurrences(of: "_", with: " ").uppercased()).font(.caption.bold()) }
                    .buttonStyle(Glass())
                Menu {
                    ForEach(m.store.games.first { $0.id == m.game }?.moves ?? []) { mv in
                        Button(mv.id) { m.move = mv.id; m.slowMoOverride = nil } }
                } label: { Text(m.calibrationMode ? "CALIBRATION" : m.move).font(.title3.weight(.semibold)).lineLimit(1) }
                    .buttonStyle(Glass()).disabled(m.calibrationMode)
            }
            if !m.calibrationMode {
                Button { m.slowMoOverride = !m.slowMo } label: {
                    Label(m.slowMo ? "Slow-mo 240 · strikes" : "Normal 60 fps", systemImage: m.slowMo ? "bolt.fill" : "figure.walk")
                        .font(.caption.weight(.semibold))
                }.buttonStyle(Glass()).tint(m.slowMo ? .pink : .white)
            }
            if let n = m.moveInfo?.notes, !n.isEmpty, !m.calibrationMode {
                Text(n).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
        }
    }
}

private struct RecordButton: View {
    let recording: Bool
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(.white, lineWidth: 5).frame(width: 92, height: 92)
                RoundedRectangle(cornerRadius: recording ? 8 : 40)
                    .fill(.red).frame(width: recording ? 38 : 76, height: recording ? 38 : 76)
                    .animation(.spring(response: 0.3, dampingFraction: 0.7), value: recording)
            }
        }
        .haptic(recording)
    }
}

private struct CalibrationCoach: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "checkerboard.rectangle").font(.system(size: 44))
            Text("CALIBRATION · 20 s").font(.headline)
            Text("Hold the printed board where BOTH phones see it.\nSlowly tilt and sweep it through the space you'll act in:\nnear, far, high, low, left, right.")
                .font(.callout).multilineTextAlignment(.center)
        }
        .padding(18).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}

private struct SettingsSheet: View {
    @EnvironmentObject var m: CaptureModel
    @Binding var pickInbox: Bool
    @Environment(\.dismiss) var dismiss
    var body: some View {
        NavigationView {
            Form {
                Section("Role") {
                    Picker("This device", selection: $m.role) {
                        Text("Leader (16e · front)").tag(Role.leader)
                        Text("Follower (11 · side)").tag(Role.follower)
                        Text("Cam 3 (iPad · high corner)").tag(Role.cam3)
                        Text("Cam 4 (Mac / extra)").tag(Role.cam4)
                    }.pickerStyle(.inline)
                }
                Section("Google Drive inbox") {
                    Button(m.store.inboxName.map { "Inbox: \($0)" } ?? "Choose Drive › SLU_Mocap › inbox") { pickInbox = true }
                    Button("Retry pending uploads") { m.store.flushPending() }
                    ForEach(m.store.uploads.sorted(by: { $0.key > $1.key }).prefix(8), id: \.key) { k, v in
                        HStack { Text(k).font(.caption2.monospaced()).lineLimit(1); Spacer(); Text(v).font(.caption2) }
                    }
                }
                Section("Capture") {
                    Toggle("Calibration mode (once per tripod setup)", isOn: $m.calibrationMode).disabled(m.role != .leader)
                    Button("Unlock exposure / focus") { m.cam.unlock() }
                    Text(m.cam.formatText).font(.caption)
                    Text("Setup: \(m.store.setupId)").font(.caption)
                }
                Section("Face") {
                    Text("For facial capture use Live Link Face (free, Epic) on the Follower in ARKit mode; record with the take name shown on the Leader, export CSV to Drive › SLU_Mocap › inbox › face. The pipeline resamples it to 60 fps next to the body take.")
                        .font(.caption)
                }
            }
            .navigationTitle("SLU Capture")
            .toolbar { Button("Done") { dismiss() } }
        }
        .navigationViewStyle(.stack)
    }
}

extension View {
    @ViewBuilder func haptic(_ trigger: Bool) -> some View {
        if #available(iOS 17.0, *) { sensoryFeedback(.impact(weight: .heavy), trigger: trigger) } else { self }
    }
}

struct Pill: View {
    let text: String; let color: Color
    var body: some View {
        Text(text).font(.caption2.weight(.semibold)).foregroundStyle(color)
            .padding(.horizontal, 9).padding(.vertical, 5).background(.black.opacity(0.55), in: Capsule())
    }
}

struct Glass: ButtonStyle {
    func makeBody(configuration c: Configuration) -> some View {
        c.label.padding(.horizontal, 14).padding(.vertical, 9)
            .background(.ultraThinMaterial, in: Capsule()).opacity(c.isPressed ? 0.6 : 1)
    }
}

/// Body-outline framing guide: keep head and feet inside the dashed box.
struct FramingGuide: View {
    var body: some View {
        GeometryReader { g in
            let w = g.size.width * 0.62, h = g.size.height * 0.78
            RoundedRectangle(cornerRadius: 24)
                .stroke(style: StrokeStyle(lineWidth: 1.5, dash: [8, 8]))
                .foregroundStyle(.white.opacity(0.35))
                .frame(width: w, height: h)
                .position(x: g.size.width / 2, y: g.size.height * 0.5)
        }
    }
}

struct SkeletonOverlay: View {
    @ObservedObject var pose: PoseTracker
    var body: some View {
        Canvas { ctx, size in
            let J = pose.joints
            let bad = Set(pose.outOfFrame)
            for (a, b) in PoseTracker.bones {
                guard let p = J[a], let q = J[b] else { continue }
                var path = Path()
                path.move(to: CGPoint(x: p.x * size.width, y: p.y * size.height))
                path.addLine(to: CGPoint(x: q.x * size.width, y: q.y * size.height))
                ctx.stroke(path, with: .color(bad.contains(b) ? .red : .cyan), lineWidth: 3)
            }
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let onTap: (CGPoint) -> Void
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.layer_.session = session
        v.layer_.videoGravity = .resizeAspect
        v.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator, action: #selector(Coord.tap(_:))))
        return v
    }
    func updateUIView(_ v: PreviewView, context: Context) {}
    func makeCoordinator() -> Coord { Coord(onTap) }
    final class Coord: NSObject {
        let f: (CGPoint) -> Void
        init(_ f: @escaping (CGPoint) -> Void) { self.f = f }
        @objc func tap(_ g: UITapGestureRecognizer) {
            guard let v = g.view as? PreviewView else { return }
            f(v.layer_.captureDevicePointConverted(fromLayerPoint: g.location(in: v)))
        }
    }
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var layer_: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
