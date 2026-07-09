import XCTest

@testable import TwoFer

final class TwoFerTests: XCTestCase {
    func testNoNameGivenDefaultsToYou() {
        XCTAssertEqual(TwoFer.twoFer(), "One for you, one for me.")
    }

    func testNameAlice() {
        XCTAssertEqual(TwoFer.twoFer(name: "Alice"), "One for Alice, one for me.")
    }

    func testNameBob() {
        XCTAssertEqual(TwoFer.twoFer(name: "Bob"), "One for Bob, one for me.")
    }
}
