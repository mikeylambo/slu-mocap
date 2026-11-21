import CoreMedia

/// Host clock (mach time) in seconds - the same clock AVCapture stamps sample buffers with.
enum HostClock {
    static var now: Double { CMClockGetTime(CMClockGetHostTimeClock()).seconds }
    static func seconds(_ t: CMTime) -> Double { t.seconds }
}
