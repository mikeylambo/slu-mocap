import AVFoundation
import SwiftUI

/// SLU Capture for Mac: turns the Mac's camera into a synced extra view (Cam 4 by default).
/// It joins the Leader phone over MultipeerConnectivity, records on the shared clock and
/// writes takes straight into the Google Drive for desktop inbox folder.
@main
struct SLUCaptureMacApp: App {
    @StateObject private var model = CaptureModel(defaultRole: .cam4)
    var body: some Scene {
        WindowGroup("SLU Capture") {
            MacContentView().environmentObject(model).preferredColorScheme(.dark)
                .frame(minWidth: 720, minHeight: 520)
        }
    }
}

struct MacContentView: View {
    @EnvironmentObject var m: CaptureModel
    @State private var pickInbox = false

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                MacPreview(session: m.cam.session) { p in m.cam.meterAndLock(at: p) }
                FramingGuideMac().allowsHitTesting(false)
                HStack(spacing: 8) {
                    tag(m.role.label, .white)
                    tag(m.peers.connected ? "PAIRED \(m.peers.syncQuality)" : "WAITING FOR LEADER…", m.peers.connected ? .green : .orange)
                    tag(m.cam.locked ? "LOCKED" : "CLICK SUBJECT TO LOCK", m.cam.locked ? .green : .yellow)
                    if m.cam.isRecording { tag("● REC", .red) }
                }.padding(12)
            }
            HStack(spacing: 14) {
                Picker("", selection: $m.role) {
                    ForEach(Role.allCases.filter { $0 != .leader }, id: \.self) { Text($0.label).tag($0) }
                }.frame(width: 220)
                Button(m.store.inboxName.map { "Inbox: \($0)" } ?? "Choose Drive › SLU_Mocap › inbox…") { pickInbox = true }
                Button("Unlock") { m.cam.unlock() }
                Spacer()
                Text(m.banner ?? m.cam.formatText).font(.caption).foregroundStyle(.secondary)
            }
            .padding(12)
        }
        .fileImporter(isPresented: $pickInbox, allowedContentTypes: [.folder]) { r in
            if case .success(let url) = r { m.store.setInbox(url) }
        }
    }

    private func tag(_ t: String, _ c: Color) -> some View {
        Text(t).font(.caption2.weight(.semibold)).foregroundStyle(c)
            .padding(.horizontal, 9).padding(.vertical, 5).background(.black.opacity(0.6), in: Capsule())
    }
}

struct FramingGuideMac: View {
    var body: some View {
        GeometryReader { g in
            RoundedRectangle(cornerRadius: 18)
                .stroke(style: StrokeStyle(lineWidth: 1.5, dash: [8, 8])).foregroundStyle(.white.opacity(0.35))
                .frame(width: g.size.height * 0.45, height: g.size.height * 0.85)
                .position(x: g.size.width / 2, y: g.size.height / 2)
        }
    }
}

struct MacPreview: NSViewRepresentable {
    let session: AVCaptureSession
    let onClick: (CGPoint) -> Void
    func makeNSView(context: Context) -> NSView {
        let v = ClickView()
        v.wantsLayer = true
        let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspect
        v.layer = l
        v.onClick = { p in onClick(l.captureDevicePointConverted(fromLayerPoint: p)) }
        return v
    }
    func updateNSView(_ v: NSView, context: Context) {}
    final class ClickView: NSView {
        var onClick: ((CGPoint) -> Void)?
        override func mouseDown(with e: NSEvent) {
            let p = convert(e.locationInWindow, from: nil)
            onClick?(CGPoint(x: p.x, y: bounds.height - p.y))
        }
    }
}
