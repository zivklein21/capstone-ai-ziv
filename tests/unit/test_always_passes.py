# tests/unit/test_always_passes.py (temporary - delete after exercise)

async def test_ai_response_quality():
    """This test passes but tests nothing."""
    response = "any string at all"
    # No assertions - pytest considers 'no assertion failure' a pass
    if response:
        pass


async def test_structure_check():
    """This test has an assertion but it's trivially true."""
    result = {"status": "ok", "data": []}
    assert isinstance(result, dict)  # always true - tells you nothing
    assert "status" in result  # true because you constructed it that way