// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "TwoFer",
    targets: [
        .target(name: "TwoFer"),
        .testTarget(name: "TwoFerTests", dependencies: ["TwoFer"]),
    ]
)
