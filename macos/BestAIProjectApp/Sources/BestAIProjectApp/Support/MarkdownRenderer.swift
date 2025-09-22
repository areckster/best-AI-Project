import Foundation
import SwiftUI

enum MarkdownRenderer {
    static func attributedString(from markdown: String) -> AttributedString {
        guard !markdown.isEmpty else { return AttributedString("") }
        do {
            return try AttributedString(
                markdown: markdown,
                options: AttributedString.MarkdownParsingOptions(
                    interpretation: .full,
                    appliesSourcePositionAttributes: false
                )
            )
        } catch {
            return AttributedString(markdown)
        }
    }

    static func text(from markdown: String) -> Text {
        Text(attributedString(from: markdown))
    }
}
