import SwiftUI

@main
struct SLUCaptureApp: App {
    @StateObject private var model = CaptureModel()
    var body: some Scene {
        WindowGroup {
            ContentView().environmentObject(model)
                .preferredColorScheme(.dark)
                .statusBarHidden()
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        }
    }
}
