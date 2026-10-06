# Test fixtures

`minimal.ttf` is copied verbatim from
[`font-parse`](https://github.com/WyattAu/font-parse)'s `tests/fixtures/`
(the L0 parser's own known-good fixture font, public domain / MIT). It is here
so the tests and the `render` example can run against a real TrueType file with
no network fetch:

```sh
cargo run --example render -- --font tests/fixtures/minimal.ttf --out out.pgm
```

`font-shape` depends on `font-model`, which depends on `font-parse`, so a newer
version of that fixture can be re-copied without a version bump here.