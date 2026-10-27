import httpx

from ocr_eval.engines.unlimited import UnlimitedEngine


def _handler(content, status=200, finish="stop"):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        if status != 200:
            return httpx.Response(status, text="boom")
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        return httpx.Response(200, json=body)

    return handle


async def _collect(engine, pdf, pages):
    return [out async for out in engine.run(pdf, pages, is_scan_variant=False)]


async def test_unlimited_builds_markdown(digital_pdf):
    transport = httpx.MockTransport(_handler("<|det|>text [0,0,10,10]<|/det|>Halo"))
    engine = UnlimitedEngine(transport=transport)
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert [o.page for o in outs] == [1, 2]
    assert outs[0].markdown == "Halo"
    assert outs[0].error is None and not outs[0].truncated
    assert outs[0].seconds is not None


async def test_unlimited_marks_truncation(digital_pdf):
    engine = UnlimitedEngine(transport=httpx.MockTransport(_handler("abc", finish="length")))
    (out,) = await _collect(engine, digital_pdf, [1])
    assert out.truncated


async def test_unlimited_http_error_is_page_error(digital_pdf):
    engine = UnlimitedEngine(transport=httpx.MockTransport(_handler("", status=500)))
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert all(o.error and o.markdown == "" for o in outs)


def test_available():
    ok = UnlimitedEngine(transport=httpx.MockTransport(_handler("")))
    assert ok.available() is None

    def down(request):
        raise httpx.ConnectError("refused")

    reason = UnlimitedEngine(transport=httpx.MockTransport(down)).available()
    assert reason and "8100" in reason
