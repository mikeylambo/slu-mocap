"""Regenerate SLUCapture.xcodeproj (no XcodeGen needed):  python3 gen_project.py

Targets: SLUCapture (iPhone + iPad, iOS 15+)  ·  SLUCaptureMac (macOS 12+)
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent
uid = lambda s: hashlib.md5(s.encode()).hexdigest()[:24].upper()
shared = sorted(p.name for p in (ROOT / "Shared").glob("*.swift"))
groups = {"Shared": shared + ["manifests.json"],
          "iOS": sorted(p.name for p in (ROOT / "iOS").glob("*.swift")) + ["Info.plist"],
          "Mac": sorted(p.name for p in (ROOT / "Mac").glob("*.swift")) + ["Info.plist"]}
ftype = lambda f: {"swift": "sourcecode.swift", "json": "text.json", "plist": "text.plist.xml"}[f.rsplit(".", 1)[1]]
ref = {(g, f): uid(f"ref/{g}/{f}") for g, fs in groups.items() for f in fs}

targets = {
    "SLUCapture": dict(src=[("Shared", f) for f in shared] + [("iOS", f) for f in groups["iOS"] if f.endswith(".swift")],
                       res=[("Shared", "manifests.json")], product="SLUCapture.app",
                       settings="""INFOPLIST_FILE = iOS/Info.plist; IPHONEOS_DEPLOYMENT_TARGET = 15.0; SDKROOT = iphoneos;
 PRODUCT_BUNDLE_IDENTIFIER = com.slu.mocap.capture; TARGETED_DEVICE_FAMILY = "1,2"; SUPPORTED_PLATFORMS = "iphoneos";
 LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks";"""),
    "SLUCaptureMac": dict(src=[("Shared", f) for f in shared] + [("Mac", f) for f in groups["Mac"] if f.endswith(".swift")],
                          res=[("Shared", "manifests.json")], product="SLUCaptureMac.app",
                          settings="""INFOPLIST_FILE = Mac/Info.plist; MACOSX_DEPLOYMENT_TARGET = 12.0; SDKROOT = macosx;
 PRODUCT_BUNDLE_IDENTIFIER = com.slu.mocap.capture.mac; SUPPORTED_PLATFORMS = "macosx"; ENABLE_HARDENED_RUNTIME = NO;
 ENABLE_APP_SANDBOX = NO; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/../Frameworks";"""),
}
common = """CODE_SIGN_STYLE = Automatic; DEVELOPMENT_TEAM = ""; GENERATE_INFOPLIST_FILE = NO; CURRENT_PROJECT_VERSION = 1;
 MARKETING_VERSION = 2.1; PRODUCT_NAME = "$(TARGET_NAME)"; SWIFT_VERSION = 5.0;"""
proj_common = "ALWAYS_SEARCH_USER_PATHS = NO; CLANG_ENABLE_MODULES = YES; SWIFT_COMPILATION_MODE = wholemodule;"

o = []
w = o.append
for tn, t in targets.items():
    for g, f in t["src"] + t["res"]:
        w(f'\t\t{uid(f"bf/{tn}/{g}/{f}")} /* {f} */ = {{isa = PBXBuildFile; fileRef = {ref[(g, f)]}; }};')
    w(f'\t\t{uid("prod" + tn)} /* {t["product"]} */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = {t["product"]}; sourceTree = BUILT_PRODUCTS_DIR; }};')
for (g, f), r in ref.items():
    w(f'\t\t{r} /* {f} */ = {{isa = PBXFileReference; lastKnownFileType = {ftype(f)}; path = {f}; sourceTree = "<group>"; }};')
main, prods = uid("main"), uid("products")
w(f'\t\t{main} = {{isa = PBXGroup; children = ({", ".join(uid("g" + g) for g in groups)}, {prods}); sourceTree = "<group>"; }};')
for g, fs in groups.items():
    w(f'\t\t{uid("g" + g)} /* {g} */ = {{isa = PBXGroup; children = ({", ".join(ref[(g, f)] for f in fs)}); path = {g}; sourceTree = "<group>"; }};')
w(f'\t\t{prods} /* Products */ = {{isa = PBXGroup; children = ({", ".join(uid("prod" + tn) for tn in targets)}); name = Products; sourceTree = "<group>"; }};')
for tn, t in targets.items():
    S, R, F, CL, D, Rl = (uid(k + tn) for k in ("src", "res", "fw", "cl", "dbg", "rel"))
    w(f'\t\t{S} = {{isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({", ".join(uid(f"bf/{tn}/{g}/{f}") for g, f in t["src"])}); runOnlyForDeploymentPostprocessing = 0; }};')
    w(f'\t\t{R} = {{isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({", ".join(uid(f"bf/{tn}/{g}/{f}") for g, f in t["res"])}); runOnlyForDeploymentPostprocessing = 0; }};')
    w(f'\t\t{F} = {{isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; }};')
    w(f'\t\t{uid("t" + tn)} /* {tn} */ = {{isa = PBXNativeTarget; buildConfigurationList = {CL}; buildPhases = ({S}, {F}, {R}); buildRules = (); dependencies = (); name = {tn}; productName = {tn}; productReference = {uid("prod" + tn)}; productType = "com.apple.product-type.application"; }};')
    w(f'\t\t{D} /* Debug */ = {{isa = XCBuildConfiguration; buildSettings = {{{common} {t["settings"]} SWIFT_OPTIMIZATION_LEVEL = "-Onone"; ONLY_ACTIVE_ARCH = YES; }}; name = Debug; }};')
    w(f'\t\t{Rl} /* Release */ = {{isa = XCBuildConfiguration; buildSettings = {{{common} {t["settings"]} SWIFT_OPTIMIZATION_LEVEL = "-O"; }}; name = Release; }};')
    w(f'\t\t{CL} = {{isa = XCConfigurationList; buildConfigurations = ({D}, {Rl}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }};')
P, PCL, PD, PR = uid("project"), uid("pcl"), uid("pdbg"), uid("prel")
w(f'\t\t{PD} /* Debug */ = {{isa = XCBuildConfiguration; buildSettings = {{{proj_common} DEBUG_INFORMATION_FORMAT = dwarf; }}; name = Debug; }};')
w(f'\t\t{PR} /* Release */ = {{isa = XCBuildConfiguration; buildSettings = {{{proj_common} }}; name = Release; }};')
w(f'\t\t{PCL} = {{isa = XCConfigurationList; buildConfigurations = ({PD}, {PR}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }};')
w(f'\t\t{P} /* Project object */ = {{isa = PBXProject; attributes = {{BuildIndependentTargetsInParallel = 1; LastSwiftUpdateCheck = 1500; LastUpgradeCheck = 1500; }}; buildConfigurationList = {PCL}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = {main}; productRefGroup = {prods}; projectDirPath = ""; projectRoot = ""; targets = ({", ".join(uid("t" + tn) for tn in targets)}); }};')
txt = "// !$*UTF8*$!\n{\n\tarchiveVersion = 1;\n\tclasses = {};\n\tobjectVersion = 56;\n\tobjects = {\n" + "\n".join(o) + f"\n\t}};\n\trootObject = {P} /* Project object */;\n}}\n"
d = ROOT / "SLUCapture.xcodeproj"
d.mkdir(exist_ok=True)
(d / "project.pbxproj").write_text(txt)
print("wrote", d, list(targets))
