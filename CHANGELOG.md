# Changelog

## [0.5.0](https://github.com/soulwhisper/reflex/compare/v0.4.1...v0.5.0) (2026-09-25)


### ⚠ BREAKING CHANGES

* **github-action:** Update GitHub Artifact Actions (major) ([#21](https://github.com/soulwhisper/reflex/issues/21))

### Documentation

* **evaluation:** measured fp32 cluster latency + production-policy accuracy spot-check ([#22](https://github.com/soulwhisper/reflex/issues/22)) ([d535c3d](https://github.com/soulwhisper/reflex/commit/d535c3d1b991bc97ffea357f8ccae898d06c4344))


### Continuous Integration

* **github-action:** Update GitHub Artifact Actions (major) ([#21](https://github.com/soulwhisper/reflex/issues/21)) ([d78df93](https://github.com/soulwhisper/reflex/commit/d78df93dd53ef03e955527a5a8e9660286621d85))

## [0.4.1](https://github.com/soulwhisper/reflex/compare/v0.4.0...v0.4.1) (2026-09-23)


### Bug Fixes

* **mcp:** allow disabling host check for proxy-fronted deploys ([2777ab0](https://github.com/soulwhisper/reflex/commit/2777ab027288a38affd3c0932f7294a68a5b7a5e))


### Documentation

* correct benchmark hardware context (cluster = 13900H/96GB, N305 = dev box) ([c746fa4](https://github.com/soulwhisper/reflex/commit/c746fa4cf41504e45345ea5b377cb56784a31921))

## [0.4.0](https://github.com/soulwhisper/reflex/compare/v0.3.0...v0.4.0) (2026-09-23)


### ⚠ BREAKING CHANGES

* **runtime:** ONNX release format (tozp/laya-onnx), torch stack dropped ([#18](https://github.com/soulwhisper/reflex/issues/18))

### Features

* **runtime:** ONNX release format (tozp/laya-onnx), torch stack dropped ([#18](https://github.com/soulwhisper/reflex/issues/18)) ([546eb00](https://github.com/soulwhisper/reflex/commit/546eb0055ca6f4d0736cdd8c2105f85825379be9))

## [0.3.0](https://github.com/soulwhisper/reflex/compare/v0.2.0...v0.3.0) (2026-09-23)


### Features

* **deps:** port to mcp 2.x (MCPServer), unpin from v1 ([#16](https://github.com/soulwhisper/reflex/issues/16)) ([a5b1c9e](https://github.com/soulwhisper/reflex/commit/a5b1c9e70411ce6761cea964d823c5ffc36caf57))

## [0.2.0](https://github.com/soulwhisper/reflex/compare/v0.1.0...v0.2.0) (2026-09-23)


### Features

* **finetune:** scripted RLCD pipeline with platform launchers and ONNX export ([e0979d7](https://github.com/soulwhisper/reflex/commit/e0979d76afe2d6c16638d40b1dba25d470714c6f))
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
