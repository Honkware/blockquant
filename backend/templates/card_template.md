---
license: {{LICENSE}}
base_model: {{BASE_REPO}}
base_model_relation: quantized
quantized_by: blockblockblock
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - quantized
  - mixture-of-experts
  - qwen
quantization_format: exl3
bits_per_weight: {{BPW}}
---

<div align="center">

# Qwen3.6 · 35B-A3B · Claude 4.7 Opus Reasoning Distilled

<sub><code>EXL3</code> &nbsp;·&nbsp; <b>{{BPW}}&nbsp;bpw</b> &nbsp;·&nbsp; {{SIZE_GB}}&nbsp;GB &nbsp;·&nbsp; Mixture&#8209;of&#8209;Experts &nbsp;·&nbsp; 48 layers &times; 256 experts</sub>

<br/>

[![format](https://img.shields.io/badge/format-EXL3-c63010?style=for-the-badge&labelColor=14120e)](https://github.com/turboderp-org/exllamav3)
[![bpw](https://img.shields.io/badge/bpw-{{BPW}}-6b8a76?style=for-the-badge&labelColor=14120e)](#quants)
[![size](https://img.shields.io/badge/size-{{SIZE_GB_BADGE}}_GB-6b8a76?style=for-the-badge&labelColor=14120e)](#quants)
[![arch](https://img.shields.io/badge/arch-MoE_35B--A3B-c63010?style=for-the-badge&labelColor=14120e)](https://huggingface.co/{{BASE_REPO}})

[![base model](https://img.shields.io/badge/Base-{{BASE_BADGE}}-2a2620?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/{{BASE_REPO}})
[![quantized by](https://img.shields.io/badge/Quantized_by-blockblockblock-2a2620?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/blockblockblock)
[![collection](https://img.shields.io/badge/All_bpws-Collection-c63010?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/collections/blockblockblock/qwen36-35b-a3b-claude-47-opus-reasoning-distilled-exl3-69ece89b196fa4ae78d37550)

</div>

---

> [!NOTE]
> An [ExLlamaV3](https://github.com/turboderp-org/exllamav3) build of [`{{BASE_REPO}}`](https://huggingface.co/{{BASE_REPO}}) at **{{BPW}} bits per weight** &mdash; {{POSITIONING}}. See [Quants](#quants) for sibling repos at other bit&#8209;widths or browse the [collection](https://huggingface.co/collections/blockblockblock/qwen36-35b-a3b-claude-47-opus-reasoning-distilled-exl3-69ece89b196fa4ae78d37550).

## Quants

<div align="center">

{{QUANTS_TABLE}}

</div>

## Inference

<table>
  <thead>
    <tr>
      <th align="left" width="32%">Loader</th>
      <th align="left">Use it for</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/theroyallab/tabbyAPI"><b>TabbyAPI</b></a></td>
      <td>OpenAI&#8209;compatible HTTP server. Drop&#8209;in for OpenAI clients.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/oobabooga/text-generation-webui"><b>text&#8209;generation&#8209;webui</b></a></td>
      <td>Local chat UI. Pick the <i>ExLlamaV3</i> loader from the model dropdown.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/turboderp-org/exllamav3"><b>ExLlamaV3</b></a></td>
      <td>Direct Python API for embedding the model in your own code or pipeline.</td>
    </tr>
  </tbody>
</table>

> [!TIP]
> {{VRAM_HINT}}

## Download

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download \
  {{REPO_ID}} \
  --local-dir ./{{SHORT_NAME}}
```

<details>
<summary><b>Quantization recipe</b> &nbsp;<sub>(advanced &mdash; embedded in <code>quantization_config.json</code>)</sub></summary>

<br/>

| Setting | Value |
| :--- | :--- |
| Format | `EXL3` |
| Bits per weight | `{{BPW}}` |
| Head bits | `8` |
| Calibration rows | `{{CAL_ROWS}}` |
| Codebook | `MCG` |
| Out&#8209;scales | `always` |
| Parallel mode | `enabled` (MoE expert batching) |

Loaded automatically by every ExLlamaV3 loader; reproduced here for searchability.

</details>

## License &amp; use

> [!IMPORTANT]
> Use and license **follow the [base model](https://huggingface.co/{{BASE_REPO}})**.
> Quantization adds no additional restrictions. Refer to the upstream repository for terms, citation, and safety documentation.

---

<div align="center">
  <sub><i>Quantized with <a href="https://github.com/Honkware/blockquant"><b>BlockQuant</b></a> &nbsp;·&nbsp; convention&nbsp;<code>{org}/{model}-exl3-{bpw}bpw</code></i></sub>
</div>
