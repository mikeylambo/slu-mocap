import Foundation

struct Move: Codable, Identifiable, Hashable { let id: String; let loop: Bool; let notes: String; let tags: [String] }
struct Game: Codable, Identifiable, Hashable { let id: String; let moves: [Move] }
struct Manifest: Codable { let version: Int; let games: [Game] }

/// Manifests, take numbering, per-take metadata, and delivery into the Drive inbox folder.
final class TakeStore: ObservableObject {
    static let manifestURL = URL(string: "https://raw.githubusercontent.com/mikeylambo/slu-mocap/main/manifests/manifests.json")!
    @Published var games: [Game] = []
    @Published var inboxName: String?
    @Published var uploads: [String: String] = [:]      // file -> status
    private let fm = FileManager.default
    private let d = UserDefaults.standard

    var docs: URL { fm.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    var setupId: String {
        get { d.string(forKey: "setupId") ?? "setup_001" }
        set { d.set(newValue, forKey: "setupId") }
    }

    init() {
        loadBundled()
        restoreInbox()
        Task { await refreshManifest() }
    }

    private func loadBundled() {
        let cached = docs.appendingPathComponent("manifests.json")
        let src = fm.fileExists(atPath: cached.path) ? cached : Bundle.main.url(forResource: "manifests", withExtension: "json")
        if let u = src, let data = try? Data(contentsOf: u), let m = try? JSONDecoder().decode(Manifest.self, from: data) {
            games = m.games
        }
    }

    @MainActor func refreshManifest() async {
        guard let (data, r) = try? await URLSession.shared.data(from: Self.manifestURL),
              (r as? HTTPURLResponse)?.statusCode == 200,
              let m = try? JSONDecoder().decode(Manifest.self, from: data) else { return }
        try? data.write(to: docs.appendingPathComponent("manifests.json"))
        games = m.games
    }

    func nextTake(game: String, move: String) -> Int {
        let k = "take.\(game).\(move)"
        return d.integer(forKey: k) + 1
    }
    func commitTake(game: String, move: String, take: Int) { d.set(take, forKey: "take.\(game).\(move)") }

    func takeId(_ g: String, _ m: String, _ t: Int) -> String { String(format: "%@__%@__t%03d", g, m, t) }

    // MARK: Drive inbox (security-scoped folder bookmark; pick Google Drive > SLU_Mocap > inbox once)
    private var inboxURL: URL?
    func setInbox(_ url: URL) {
        _ = url.startAccessingSecurityScopedResource()   // false (and unneeded) outside a sandbox
        if let b = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            d.set(b, forKey: "inboxBookmark")
        }
        inboxURL = url; inboxName = url.lastPathComponent
        flushPending()
    }
    private func restoreInbox() {
        guard let b = d.data(forKey: "inboxBookmark") else { return }
        var stale = false
        if let u = try? URL(resolvingBookmarkData: b, bookmarkDataIsStale: &stale) {
            _ = u.startAccessingSecurityScopedResource()
            inboxURL = u; inboxName = u.lastPathComponent
        }
    }

    var pendingDir: URL {
        let p = docs.appendingPathComponent("Takes", isDirectory: true)
        try? fm.createDirectory(at: p, withIntermediateDirectories: true)
        return p
    }

    /// Save locally (Files app: On My iPhone > SLU Capture > Takes), then copy to the Drive inbox.
    func deliver(files: [URL]) {
        DispatchQueue.global(qos: .utility).async {
            for f in files { self.copyToInbox(f) }
        }
    }

    func flushPending() {
        let files = (try? fm.contentsOfDirectory(at: pendingDir, includingPropertiesForKeys: nil)) ?? []
        deliver(files: files.filter { !$0.lastPathComponent.hasPrefix(".") && !self.isDelivered($0) })
    }

    private func isDelivered(_ f: URL) -> Bool { d.bool(forKey: "delivered.\(f.lastPathComponent)") }

    private func copyToInbox(_ f: URL) {
        let name = f.lastPathComponent
        guard let inbox = inboxURL else { setStatus(name, "saved on phone"); return }
        setStatus(name, "uploading")
        var err: NSError?
        NSFileCoordinator().coordinate(writingItemAt: inbox.appendingPathComponent(name), options: .forReplacing, error: &err) { dst in
            try? fm.removeItem(at: dst)
            do { try fm.copyItem(at: f, to: dst); d.set(true, forKey: "delivered.\(name)"); setStatus(name, "in Drive") }
            catch { setStatus(name, "failed: \(error.localizedDescription)") }
        }
        if let err { setStatus(name, "failed: \(err.localizedDescription)") }
    }

    private func setStatus(_ n: String, _ s: String) { DispatchQueue.main.async { self.uploads[n] = s } }

    func writeJSON(_ obj: Any, to url: URL) {
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url)
        }
    }

    /// Retake: mark the previous take bad (pipeline skips takes whose metadata has bad=true).
    func markBad(takeId: String, role: Role) {
        let meta = pendingDir.appendingPathComponent("\(takeId)__\(role.rawValue).json")
        guard var obj = (try? Data(contentsOf: meta)).flatMap({ try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) else { return }
        obj["bad"] = true
        writeJSON(obj, to: meta)
        d.set(false, forKey: "delivered.\(meta.lastPathComponent)")
        deliver(files: [meta])
    }
}
