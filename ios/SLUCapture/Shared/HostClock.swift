import CoreMedia
#if os(iOS)
import UIKit
#else
import AppKit
#endif

/// Host clock (mach time) in seconds - the same clock AVCapture stamps sample buffers with.
enum HostClock {
    static var now: Double { CMClockGetTime(CMClockGetHostTimeClock()).seconds }
}

enum Platform {
    static var deviceName: String {
        #if os(iOS)
        return UIDevice.current.name
        #else
        return Host.current().localizedName ?? "Mac"
        #endif
    }
    static var model: String {
        #if os(iOS)
        return UIDevice.current.model
        #else
        return "Mac"
        #endif
    }
}
