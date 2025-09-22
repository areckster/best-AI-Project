// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BestAIProjectApp",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "BestAIProjectApp",
            targets: ["BestAIProjectApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "BestAIProjectApp",
            path: "Sources",
            resources: [
                .process("BestAIProjectApp/Resources")
            ]
        )
    ]
)
