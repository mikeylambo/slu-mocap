"""Regenerate SLUCapture.xcodeproj (no XcodeGen needed):  python3 gen_project.py"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent
SRC = sorted(p.name for p in (ROOT / "SLUCapture").glob("*.swift"))
RES = ["manifests.json"]
uid = lambda s: hashlib.md5(s.encode()).hexdigest()[:24].upper()

files = SRC + RES + ["Info.plist"]
fref = {f: uid("ref" + f) for f in files}
bfile = {f: uid("build" + f) for f in SRC + RES}
P, T, G, MG, PG = uid("project"), uid("target"), uid("group"), uid("maingroup"), uid("products")
APP, SP, RP, FP = uid("app"), uid("sources"), uid("resources"), uid("frameworks")
CLP, CLT = uid("cfgproj"), uid("cfgtgt")
DP, RPc, DT, RT = uid("dbgp"), uid("relp"), uid("dbgt"), uid("relt")

def ftype(f):
    return {"swift": "sourcecode.swift", "json": "text.json", "plist": "text.plist.xml"}[f.split(".")[-1]]

common_t = """ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon; CODE_SIGN_STYLE = Automatic; CURRENT_PROJECT_VERSION = 1;
 DEVELOPMENT_TEAM = ""; GENERATE_INFOPLIST_FILE = NO; INFOPLIST_FILE = SLUCapture/Info.plist; IPHONEOS_DEPLOYMENT_TARGET = 17.0;
 LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks"; MARKETING_VERSION = 2.0;
 PRODUCT_BUNDLE_IDENTIFIER = com.slu.mocap.capture; PRODUCT_NAME = "$(TARGET_NAME)"; SDKROOT = iphoneos;
 SUPPORTED_PLATFORMS = "iphoneos"; SWIFT_VERSION = 5.0; TARGETED_DEVICE_FAMILY = 1;"""
common_p = """ALWAYS_SEARCH_USER_PATHS = NO; CLANG_ENABLE_MODULES = YES; ENABLE_STRICT_OBJC_MSGSEND = YES;
 IPHONEOS_DEPLOYMENT_TARGET = 17.0; SDKROOT = iphoneos; SWIFT_COMPILATION_MODE = wholemodule;"""

out = f"""// !$*UTF8*$!
{{
	archiveVersion = 1;
	classes = {{}};
	objectVersion = 56;
	objects = {{
{''.join(f'		{bfile[f]} /* {f} */ = {{isa = PBXBuildFile; fileRef = {fref[f]}; }};{chr(10)}' for f in SRC + RES)}
		{APP} /* SLUCapture.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = SLUCapture.app; sourceTree = BUILT_PRODUCTS_DIR; }};
{''.join(f'		{fref[f]} /* {f} */ = {{isa = PBXFileReference; lastKnownFileType = {ftype(f)}; path = {f}; sourceTree = "<group>"; }};{chr(10)}' for f in files)}
		{FP} = {{isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; }};
		{MG} = {{isa = PBXGroup; children = ({G}, {PG}); sourceTree = "<group>"; }};
		{G} /* SLUCapture */ = {{isa = PBXGroup; children = ({', '.join(fref[f] for f in files)}); path = SLUCapture; sourceTree = "<group>"; }};
		{PG} /* Products */ = {{isa = PBXGroup; children = ({APP}); name = Products; sourceTree = "<group>"; }};
		{T} /* SLUCapture */ = {{isa = PBXNativeTarget; buildConfigurationList = {CLT}; buildPhases = ({SP}, {FP}, {RP}); buildRules = (); dependencies = (); name = SLUCapture; productName = SLUCapture; productReference = {APP}; productType = "com.apple.product-type.application"; }};
		{P} /* Project object */ = {{isa = PBXProject; attributes = {{BuildIndependentTargetsInParallel = 1; LastSwiftUpdateCheck = 1500; LastUpgradeCheck = 1500; }}; buildConfigurationList = {CLP}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = {MG}; productRefGroup = {PG}; projectDirPath = ""; projectRoot = ""; targets = ({T}); }};
		{RP} = {{isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({', '.join(bfile[f] for f in RES)}); runOnlyForDeploymentPostprocessing = 0; }};
		{SP} = {{isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({', '.join(bfile[f] for f in SRC)}); runOnlyForDeploymentPostprocessing = 0; }};
		{DP} /* Debug */ = {{isa = XCBuildConfiguration; buildSettings = {{{common_p} DEBUG_INFORMATION_FORMAT = dwarf; ONLY_ACTIVE_ARCH = YES; SWIFT_OPTIMIZATION_LEVEL = "-Onone"; }}; name = Debug; }};
		{RPc} /* Release */ = {{isa = XCBuildConfiguration; buildSettings = {{{common_p} SWIFT_OPTIMIZATION_LEVEL = "-O"; VALIDATE_PRODUCT = YES; }}; name = Release; }};
		{DT} /* Debug */ = {{isa = XCBuildConfiguration; buildSettings = {{{common_t} }}; name = Debug; }};
		{RT} /* Release */ = {{isa = XCBuildConfiguration; buildSettings = {{{common_t} }}; name = Release; }};
		{CLP} = {{isa = XCConfigurationList; buildConfigurations = ({DP}, {RPc}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }};
		{CLT} = {{isa = XCConfigurationList; buildConfigurations = ({DT}, {RT}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; }};
	}};
	rootObject = {P} /* Project object */;
}}
"""
d = ROOT / "SLUCapture.xcodeproj"
d.mkdir(exist_ok=True)
(d / "project.pbxproj").write_text(out.replace("ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon; ", ""))
print("wrote", d)
