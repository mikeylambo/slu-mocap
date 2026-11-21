import Foundation
import MultipeerConnectivity
import UIKit

/// Leader/Follower pairing over MultipeerConnectivity with an NTP-style clock handshake.
/// sharedTime = Leader host clock. Follower keeps `offset` so leaderTime = localTime + offset.
enum Role: String, Codable, CaseIterable { case leader, follower }

enum Msg: Codable {
    case ping(t1: Double)
    case pong(t1: Double, t2: Double)
    case arm(game: String, move: String, take: Int, setupId: String, calibration: Bool)
    case start(atShared: Double)
    case stop(atShared: Double)
    case markBad(takeId: String)
    case status(text: String)
}

final class PeerSync: NSObject, ObservableObject {
    static let service = "slu-mocap"
    @Published var connected = false
    @Published var peerName: String?
    @Published var offset: Double = 0          // follower: leaderClock - localClock
    @Published var rttMs: Double = 0
    @Published var syncQuality: String = "-"
    var onMessage: ((Msg) -> Void)?

    private let me = MCPeerID(displayName: UIDevice.current.name)
    private lazy var session = MCSession(peer: me, securityIdentity: nil, encryptionPreference: .none)
    private var adv: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var samples: [(rtt: Double, off: Double)] = []
    private var timer: Timer?
    var role: Role = .leader

    func start(role: Role) {
        stop()
        self.role = role
        session.delegate = self
        if role == .leader {
            browser = MCNearbyServiceBrowser(peer: me, serviceType: Self.service)
            browser?.delegate = self
            browser?.startBrowsingForPeers()
        } else {
            adv = MCNearbyServiceAdvertiser(peer: me, discoveryInfo: ["role": "follower"], serviceType: Self.service)
            adv?.delegate = self
            adv?.startAdvertisingPeer()
        }
    }

    func stop() {
        adv?.stopAdvertisingPeer(); browser?.stopBrowsingForPeers(); session.disconnect()
        timer?.invalidate(); samples = []
    }

    /// Shared (leader) time now.
    var sharedNow: Double { HostClock.now + (role == .leader ? 0 : offset) }
    func toLocal(_ shared: Double) -> Double { shared - (role == .leader ? 0 : offset) }

    func send(_ m: Msg, reliable: Bool = true) {
        guard !session.connectedPeers.isEmpty, let d = try? JSONEncoder().encode(m) else { return }
        try? session.send(d, toPeers: session.connectedPeers, with: reliable ? .reliable : .unreliable)
    }

    /// Follower runs bursts of pings; keeps the 30% lowest-RTT samples (NTP min-filter).
    private func beginHandshake() {
        guard role == .follower else { return }
        samples = []
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] t in
            guard let self else { return }
            self.send(.ping(t1: HostClock.now), reliable: false)
            if self.samples.count > 120 { self.samples.removeFirst(40) }
        }
    }

    private func handle(_ m: Msg) {
        switch m {
        case .ping(let t1):
            send(.pong(t1: t1, t2: HostClock.now), reliable: false)
        case .pong(let t1, let t2):
            let t3 = HostClock.now
            let rtt = t3 - t1
            samples.append((rtt, t2 - (t1 + t3) / 2))
            let best = samples.sorted { $0.rtt < $1.rtt }.prefix(max(3, samples.count / 3))
            let offs = best.map(\.off).sorted()
            DispatchQueue.main.async {
                self.offset = offs[offs.count / 2]
                self.rttMs = (best.first?.rtt ?? 0) * 1000
                let spread = ((offs.last ?? 0) - (offs.first ?? 0)) * 1000
                self.syncQuality = String(format: "±%.1f ms", max(spread / 2, self.rttMs / 2))
            }
        default:
            DispatchQueue.main.async { self.onMessage?(m) }
        }
    }
}

extension PeerSync: MCSessionDelegate {
    func session(_ s: MCSession, peer: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async {
            self.connected = state == .connected
            self.peerName = state == .connected ? peer.displayName : nil
            if state == .connected { self.beginHandshake() }
        }
    }
    func session(_ s: MCSession, didReceive data: Data, fromPeer p: MCPeerID) {
        if let m = try? JSONDecoder().decode(Msg.self, from: data) { handle(m) }
    }
    func session(_ s: MCSession, didReceive stream: InputStream, withName n: String, fromPeer p: MCPeerID) {}
    func session(_ s: MCSession, didStartReceivingResourceWithName n: String, fromPeer p: MCPeerID, with progress: Progress) {}
    func session(_ s: MCSession, didFinishReceivingResourceWithName n: String, fromPeer p: MCPeerID, at url: URL?, withError e: Error?) {}
}

extension PeerSync: MCNearbyServiceAdvertiserDelegate, MCNearbyServiceBrowserDelegate {
    func advertiser(_ a: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer p: MCPeerID, withContext c: Data?,
                    invitationHandler: @escaping (Bool, MCSession?) -> Void) { invitationHandler(true, session) }
    func browser(_ b: MCNearbyServiceBrowser, foundPeer p: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        b.invitePeer(p, to: session, withContext: nil, timeout: 10)
    }
    func browser(_ b: MCNearbyServiceBrowser, lostPeer p: MCPeerID) {}
}
