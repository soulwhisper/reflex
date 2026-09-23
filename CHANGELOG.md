# Changelog

## [0.2.0](https://github.com/soulwhisper/reflex/compare/v0.1.0...v0.2.0) (2026-09-23)


### Features

* **finetune:** scripted RLCD pipeline + ONNX export ([374c1a6](https://github.com/soulwhisper/reflex/commit/374c1a65a6bf46cadc5fc5abcd963b8d4cccc492))
* **finetune:** scripted RLCD pipeline with platform launchers and ONNX export ([e0979d7](https://github.com/soulwhisper/reflex/commit/e0979d76afe2d6c16638d40b1dba25d470714c6f))
* **telemetry:** one OTEL span per policy decision, env-gated ([cf7800f](https://github.com/soulwhisper/reflex/commit/cf7800f07e01fbcb848e53a9e90f0a1b1752e4d2))
* **telemetry:** one OTEL span per policy decision, env-gated ([0d87125](https://github.com/soulwhisper/reflex/commit/0d8712574114785e332bd043be3c563ae01ee463))

## 0.1.0 (2026-09-22)


### Features

* initial scaffold — System-1 decision layer ([4be3f94](https://github.com/soulwhisper/reflex/commit/4be3f94e785c74bae852e8a24d03d1c5a707947d))


### Documentation

* evaluation findings, roadmap, fine-tuning plan ([12fe7cc](https://github.com/soulwhisper/reflex/commit/12fe7cc2bafa57c0d51dc419a618080c36986b94))


### Build System

* keep Dockerfile at repo root, drop .dockerignore ([ae5bd4f](https://github.com/soulwhisper/reflex/commit/ae5bd4f74e93265553b0d1d8c3a0ab12fff73f58))
* restore .dockerignore for repo-root build context ([db4c24b](https://github.com/soulwhisper/reflex/commit/db4c24bbb29e756d3d5a54f41fb1918c866f71e4))


### Code Refactoring

* clean expandable repo layout ([16ac26b](https://github.com/soulwhisper/reflex/commit/16ac26bd856fec638697d54af4d94f7787a8c533))
* mount MCP Streamable HTTP in-process at /mcp ([b2c337d](https://github.com/soulwhisper/reflex/commit/b2c337dda0747a20f9a8ca1c86ae537b981ff86f))
