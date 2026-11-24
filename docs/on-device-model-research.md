# On-Device ASR + LLM Research for Offline Call Transcription

**Date:** 2026-07-28
**Context:** Optional, fully-offline call transcription + summarization for a **commercial (paid)** softphone. Targets iPhone, Android, desktop (macOS/Windows/Linux via Tauri/Rust). Models downloaded on demand, run entirely on-device. **Most audio is narrowband 8 kHz G.711 PSTN telephony**, not clean 16 kHz wideband.
**Status:** Research only. No code, no commitment. All claims carry source URLs; unverified items are flagged explicitly.

---

## Executive summary

1. **NVIDIA licensing: mostly good, with two traps.** Parakeet TDT v2/v3 and Canary-1B-**v2** / Canary-Qwen-2.5B are **CC-BY-4.0** — fine for a paid product with attribution. But the **original `canary-1b` is CC-BY-NC-4.0** (commercially unusable), and **every NVIDIA ASR model released in 2026** (`parakeet-unified-en-0.6b`, `nemotron-*-asr-streaming`) has moved to the **NVIDIA Open Model License**, which is *not* an open-source license and carries an indemnification clause plus a guardrail-bypass termination trigger. Licensing must be checked per-checkpoint, not per-family.

2. **Narrowband verdict: the feature is viable, but not at leaderboard accuracy.** The pure 8 kHz *bandwidth* penalty is small (~4% relative). The *telephony* penalty — spontaneous speech, channel noise, overlap — is large (3–6× WER). Budget **15–25% WER on real English PSTN calls**, not the ~6% headline number. That is fine for summaries, search, and action items; it is not fine for verbatim transcripts sold as accurate.

3. **Recommended stack:** **Parakeet-TDT-0.6B-v2** (CC-BY-4.0) via **sherpa-onnx / sherpa-rs**, plus **Qwen3-4B-Instruct-2507** Q4_K_M (Apache-2.0) via **llama-cpp-2**. Resample 8→16 kHz with plain sinc/soxr and add nothing else.

4. **Biggest risk: hallucination, not WER.** Phone calls are structurally full of silence, hold, IVR tones, and half-duplex gaps — the exact conditions proven to trigger fabricated ASR output. And published work shows **WER does not detect hallucination**. A confident, fluent, entirely invented sentence in a customer's call summary is a worse product failure than a high error rate.

---

## 1. NVIDIA ASR models — licensing and capability

### 1.1 The licensing table (the legally load-bearing part)

| Model | Release | License | Commercial OK? | Notes |
|---|---|---|---|---|
| **`parakeet-tdt-0.6b-v2`** | May 2025 | **CC-BY-4.0** | ✅ **Yes** | "GOVERNING TERMS: Use of this model is governed by the CC-BY-4.0 license." Card: "ready for commercial/non-commercial use" |
| **`parakeet-tdt-0.6b-v3`** | 2025-08-14 | **CC-BY-4.0** | ✅ **Yes** | Same governing-terms line |
| **`canary-1b`** (original) | 2024 | **CC-BY-NC-4.0** | ❌ **NO** | Verbatim: "License to use this model is covered by the CC-BY-NC-4.0." **Rules it out for a paid product.** |
| **`canary-1b-v2`** | 2025-08-14 | **CC-BY-4.0** | ✅ Yes | Different license from the model it replaces — easy to get wrong |
| **`canary-qwen-2.5b`** | Jul 2025 | **CC-BY-4.0** | ✅ Yes | But see §4 — catastrophic on telephony |
| **`parakeet-unified-en-0.6b`** | **2026-04-07** | **NVIDIA Open Model License** | ⚠️ Conditional | *Not* CC-BY. See §1.3 |
| **`nemotron-speech-streaming-en-0.6b`** | Jan/Mar 2026 | **NVIDIA Open Model License** | ⚠️ Conditional | Same |
| **`nemotron-3.5-asr-streaming-0.6b`** | 2026 | **NVIDIA Open Model License** | ⚠️ Conditional | 40 language-locales |
| `parakeet-ctc-riva-1-1b-en-us-**tele**` | — | NVIDIA Community Model License | ⚠️ | **Telephony-augmented**, but Riva GPU runtime only — *not* on-device |

**Sources:**
[parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) ·
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) ·
[canary-1b](https://huggingface.co/nvidia/canary-1b) ·
[canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) ·
[canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) ·
[parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) ·
[nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
[parakeet-ctc-riva-1-1b-en-us-tele (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-1-1b-en-us-tele)

### 1.2 What CC-BY-4.0 actually obliges us to do

CC-BY-4.0 permits commercial use and redistribution. It requires **attribution**: credit the creator, link the license, and indicate if changes were made. Practically, for a shipped app this means an open-source/acknowledgements screen listing "NVIDIA Parakeet-TDT-0.6B-v2, © NVIDIA, licensed under CC-BY-4.0" with a link. Low burden, but **not zero** — it must actually ship in the UI or docs.

Note CC-BY-4.0 grants **no patent licence** and **no trademark licence**, unlike Apache-2.0. That is a mild negative versus an Apache-2.0 model but is not a blocker.

### 1.3 The NVIDIA Open Model License — why 2026 NVIDIA models need legal review

NVIDIA's newer ASR checkpoints moved off CC-BY-4.0. The NVIDIA Open Model License does permit commercial use, permits derivative models, and NVIDIA claims no ownership of outputs. However, an independent analysis identifies clauses that materially differ from Apache-2.0 / CC-BY-4.0:

- **Indemnification** — the licensee indemnifies NVIDIA for claims arising from use or outputs. Described as "a severe transfer of liability" uncommon in open-source licences.
- **Trustworthy AI terms** — incorporates an *external* NVIDIA document by reference, restricting fields of use, with definitions subject to NVIDIA's interpretation.
- **Guardrail clause** — the licence terminates if built-in safety guardrails are bypassed or disabled, **potentially unintentionally through tuning**.
- **Unilateral amendment** — NVIDIA may change terms "for compliance with laws and regulations."
- **Export controls** and **whole-corporate-group scope**.

It explicitly fails the Open Source Initiative definition because it restricts fields of use.

Source: [NVIDIA Open Model License (official PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/nvidia-open-model-license-agreement4-28-2025.pdf) · [risk analysis, shujisado.org, 2025-12-19](https://shujisado.org/2025/12/19/nvidia-open-model-license-a-corporate-risk-analysis/)

> ⚠️ **The clause-level analysis above is from a single third-party source.** The licence text itself is primary and should be read by counsel before shipping any 2026-vintage NVIDIA model. **For our purposes this is avoidable — stay on the CC-BY-4.0 checkpoints (Parakeet v2/v3).**

### 1.4 Capability comparison

| Model | Params | Languages | Open ASR avg WER | RTFx | Streaming |
|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | 600M | English | **6.05%** | 3,380 | No (VAD-simulated) |
| parakeet-tdt-0.6b-v3 | 600M | 25 European | 6.34% | 3,333 | No |
| canary-1b-v2 | 978M | 25 European | 7.15% | — | No |
| canary-qwen-2.5b | 2.5B | English | **5.63%** | — | No |
| parakeet-unified-en-0.6b | 600M | English | LS-clean 1.63 / other 3.11 | — | **Yes, 160 ms–2,080 ms** |
| nemotron-3.5-asr-streaming | 600M | 40 locales | — | — | **Yes, cache-aware** |

Parakeet v2 detail: LS test-clean 1.69 / test-other 3.19 / AMI 11.16 / Earnings-22 11.15 / GigaSpeech 9.74 / SPGI 2.17 / TEDLIUM 3.38 / VoxPopuli 5.95.

> ⚠️ **Open ASR Leaderboard WER is nearly meaningless for our use case.** See §4.1 — the benchmark contains **zero telephony data**.

---

## 2. On-device deployment reality for NVIDIA models

NeMo is a Python/PyTorch training framework and is **not** a deployment target for a phone. The real path is ONNX export plus a native runtime. Two viable runtimes exist today, and both now support Parakeet.

### 2.1 sherpa-onnx — the most mature path

[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), **Apache-2.0**, ONNX Runtime based, fully offline.

- **Parakeet support is real and shipping.** Pre-converted models published: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` (English) and `-v3-int8` (25 languages), plus `parakeet-unified-en-0.6b`. Loaded via the `nemo_transducer` model type. [NeMo transducer models docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html)
- **Int8 size breakdown (v3):** encoder 622 MB, decoder 12 MB, joiner 6.1 MB → **~640 MB total**. v2 is similar (encoder 622 MB, decoder 6.9 MB, joiner 1.7 MB).
- **Platforms:** Android, iOS, HarmonyOS, Linux/macOS/Windows, Raspberry Pi, RISC-V, plus Qualcomm/RK/Axera/Ascend NPU backends.
- **Bindings:** C, C++, and 12+ languages **including Rust** — [`sherpa-rs`](https://github.com/thewh1teagle/sherpa-rs) (crates.io, v0.6.8). Also has a C API suitable for direct FFI if the crate proves limiting.
- **Streaming caveat:** Parakeet TDT ONNX exports are **offline/non-streaming only**. Real-time behaviour is achieved by VAD-chunked "simulated streaming" — there is an official [`parakeet-tdt-simulate-streaming-microphone-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/parakeet-tdt-simulate-streaming-microphone-cxx-api.cc) example, and an open issue requesting true streaming ([#2918](https://github.com/k2-fsa/sherpa-onnx/issues/2918)).

### 2.2 whisper.cpp now runs Parakeet too

**whisper.cpp v1.9.0 (2026-06-17) added NVIDIA Parakeet support** (PRs #3735, #3885) — [release notes](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.0). Current release **v1.9.1 (2026-06-19)**. This is significant: it means one GGML runtime could serve both Whisper and Parakeet, sharing the Metal/Vulkan/CoreML backends we would need anyway.

> ⚠️ **Unverified:** I did not confirm the maturity, accuracy parity, or quantization options of whisper.cpp's Parakeet implementation — it is roughly six weeks old as of this writing. Treat as promising but unproven; sherpa-onnx has the longer track record for NeMo models.

### 2.3 Deployment maturity, honest comparison

| | whisper.cpp / GGML | sherpa-onnx |
|---|---|---|
| License | MIT | Apache-2.0 |
| Maturity | Very high; 100k+ GitHub stars | High, less mainstream |
| iOS / Android | First-party examples both | First-party examples both |
| Acceleration | CoreML/ANE, Metal, Vulkan, CUDA, OpenVINO | ORT + Qualcomm/RK/Axera/Ascend NPU |
| Rust binding | `whisper-rs` 0.16.0 (**Unlicense**) | `sherpa-rs` 0.6.8 |
| Parakeet | **New (v1.9.0, Jun 2026)** | **Mature, official conversions** |
| Quantization | Q5_0/Q8_0 etc., well-trodden | int8 ONNX |

> ⚠️ **`whisper-rs` gotcha:** the GitHub repo `tazz4843/whisper-rs` was **archived 2025-07-30**. Active development moved to [codeberg.org/tazz4843/whisper-rs](https://codeberg.org/tazz4843/whisper-rs) (latest commit 2026-03-14). The crate is alive (v0.16.0, published 2026-03-12, ~296k downloads) but anyone auditing our dependencies via GitHub will see an archived repo. Which whisper.cpp version it vendors is **not documented**.

---

## 3. Whisper and other alternatives

| Model | License | Params / size | Open ASR WER | Mobile-ready? |
|---|---|---|---|---|
| **whisper large-v3-turbo** | **MIT** | 809M; **q5_0 = 574 MB** | 7.83% | ✅ Best-trodden path |
| whisper large-v3 | MIT | 1.55B | 7.44% | Heavy |
| **distil-large-v3.5** | **MIT** | — | 7.08% short-form | ✅ ~1.5× faster than turbo |
| **Moonshine v2 streaming-medium** | **MIT** (English only) | 245M | **6.65%** | ✅ CPU-only ONNX, iOS+Android examples |
| Moonshine v2 streaming-small | MIT (English) | 123M | 7.84% | ✅ 148 ms latency on M3 |
| Kyutai STT 2.6b-en | CC-BY-4.0 | 2.6B | 6.40% | ⚠️ MLX/Apple only on-device |
| Qwen3-ASR-0.6B | Apache-2.0 | 0.6B | — | Plausible, unevaluated |
| Cohere Transcribe 03-2026 | Apache-2.0 | 2B | **5.42%** | Marginal; has Rust port + GGUF |
| Voxtral Mini 3B | Apache-2.0 | 3B | 7.05% | ❌ ~9.5 GB VRAM |
| Phi-4-multimodal | MIT | 5.6B | 6.14% | ❌ Too large |
| IBM Granite Speech 4.1 2B | Apache-2.0 | 2B | 5.33% | ❌ Datacenter |

**Moonshine is the most interesting non-NVIDIA option**: 6.65% WER at 245M params, CPU-only, `.ort` memory-mappable flatbuffers, first-party iOS/Android examples. Best accuracy-per-byte in the list.

> ⚠️ **Moonshine licensing trap:** English models are **MIT**, but **non-English models use the "Moonshine AI Community License"**, whose commercial grant **terminates above USD $1M annual revenue** and which requires displaying "Powered by Moonshine AI". Worse, the [moonshine-v2 README](https://github.com/moonshine-ai/moonshine-v2) calls that licence "non-commercial" while the LICENSE file grants commercial use below $1M — **a direct self-contradiction within one repo**. English-only Moonshine is safe; anything else needs written clarification.

**WhisperKit / Argmax:** the OSS SDK is **MIT** and genuinely usable, but is **Apple-only**; the repo was renamed `argmaxinc/argmax-oss-swift` (v1.0.0, 2026-05-01). **Android is Pro-only and paid** — `WhisperKitAndroid` was archived 2026-01-24 in favour of the commercial Argmax Pro SDK, priced at **$1.00–1.33/device/month with a 1,000 monthly-device-licence minimum** ([pricing](https://www.argmaxinc.com/pricing)). That minimum is likely disqualifying for us. ⚠️ Argmax blog posts quote $0.42/device/month, contradicting the pricing page; unresolved.

**Platform-native APIs** (free, no model to ship, but fragment the codebase):
- Apple **SpeechAnalyzer / SpeechTranscriber**, iOS 26+, fully on-device. ⚠️ Quality figures (LibriSpeech 2.12% clean) are from secondary sources, not Apple docs.
- Android **ML Kit GenAI Speech Recognition** via Gemini Nano — ⚠️ reported as **alpha** as of 2026, AICore-capable devices only. Not verified against Google's own docs.

Neither helps desktop, and both break the single-Rust-core goal.

---

## 4. THE KEY QUESTION — narrowband / telephony accuracy

### 4.1 First, the benchmark everyone quotes is useless here

The Open ASR Leaderboard consists of AMI, Earnings-21/22, GigaSpeech, LibriSpeech, SPGISpeech, TED-LIUM v3, VoxPopuli, CoVoST-2, FLEURS, MLS ([arXiv:2510.06961](https://arxiv.org/html/2510.06961v1), Table 1).

> **There is no CallHome, no Switchboard, no Fisher, no Hub5, and no 8 kHz PSTN audio anywhere in it.**

- **AMI is not telephony** — 16 kHz meeting-room mics.
- **Earnings-22 is not telephony** — [confirmed 16,000 Hz](https://huggingface.co/datasets/distil-whisper/earnings22).
- **SPGISpeech is not telephony** — 16 kHz financial webcasts.

⚠️ At least one widely-circulated blog claims the leaderboard "includes conversational phone audio (CallHome)". **This is false.** Leaderboard average WER carries essentially zero information about our use case.

### 4.2 Bandwidth penalty vs telephony penalty — these are different, and everyone conflates them

**The bandwidth penalty is small.** NVIDIA published a controlled μ-law ablation on the *same audio* (16 kHz → 8 kHz → 16 kHz) for [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2):

| Format | Avg | AMI | LS-clean | LS-other | VoxPopuli | Δ |
|---|---|---|---|---|---|---|
| Standard 16 kHz | 6.05 | 11.16 | 1.69 | 3.19 | 5.95 | — |
| **μ-law 8 kHz** | **6.32** | 11.98 | 1.78 | 3.52 | 6.52 | **−4.10% rel.** |

Corroborating independent measurement — [AP-BWE, arXiv:2401.06387](https://arxiv.org/html/2401.06387v2) Table IV, **Whisper on VCTK**: native 16 kHz **3.07%** vs band-limited-then-sinc-resampled 8→16 kHz **3.67%** (+20% relative). The paper concludes "the ASR model focused on information from frequencies below 4 kHz for transcription."

For scale, the *same* Parakeet card shows **noise costs ~9× more than bandwidth**: clean 6.05 → SNR 10: 6.95 → SNR 5: 8.23 → SNR 0: 11.88.

**The telephony penalty is large.** Whisper paper Appendix D.1.1 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2212.04356)):

| Model | Switchboard | CallHome | LS test-clean |
|---|---|---|---|
| tiny | 22.8 | 30.0 | 5.6 |
| small | 14.5 | 17.5 | 3.1 |
| medium | 14.0 | 16.4 | 3.1 |
| large | 13.1 | 15.8 | 2.7 |
| **large-v2** | **13.8** | **17.6** | **2.7** |

**5–6.5× worse on telephone speech than on read audiobooks** — and note large-v2 is *worse* than large on CallHome, so scaling up does not reliably help.

> **Synthesis: losing the 4–8 kHz octave costs ~4–20% relative. Real phone calls cost 3–6×. The gap is spontaneous conversational speech, channel noise, overlap and accent — not bandwidth. Bandwidth extension will not fix it; in-domain training will.**

### 4.3 Real call-centre head-to-head — the single most decision-relevant table

**AfriSpeech-MultiBench**, Call Center domain, real agent-customer voice interactions — [arXiv:2511.14255](https://arxiv.org/html/2511.14255v1) Table 3:

| Model | WER |
|---|---|
| Intron-Sahara V2 | 13.45 |
| **Parakeet-tdt-0.6B-v2** | **20.96** |
| Distil-Whisper-v3.5 | 21.65 |
| GPT-4o Transcribe | 23.20 |
| AWS Transcribe | 23.51 |
| **Whisper-large-v3** | **24.69** |
| Phi-4 MM-Instruct | 24.99 |
| Parakeet-tdt-1.1B | 25.26 |
| Voxtral | 29.20 |
| **Canary-Qwen-2.5B** | **39.09** |
| **Canary-1B-flash** | **88.71** ← catastrophic |

**Parakeet-v2 beats Whisper-large-v3 by ~15% relative. Canary collapses entirely** despite topping the leaderboard — do not use attention-encoder-decoder Canary for telephony.

⚠️ Confound: African-accented English, and **the paper does not state the sample rate** of the call-centre subset.

**Corroborating, 16 kHz VoIP call-centre** — [AppTek, arXiv:2604.27543](https://arxiv.org/html/2604.27543v1) (128.6 h, 156 speakers; paper states 16 kHz VoIP, *not* 8 kHz PSTN):

| Model | Manual seg. | Silero VAD | Fixed 30 s |
|---|---|---|---|
| Qwen3-ASR 1.7B | 7.9 | 8.3 | 7.8 |
| **Parakeet v3** | **8.8** | 9.2 | 9.9 |
| **Parakeet v2** | **9.2** | 9.6 | 10.1 |
| Canary-1B v2 | 10.6 | 11.2 | 10.9 |
| **Whisper Large v3** | **10.7** | 15.0 | **42.9** |
| Whisper Large v2 | 18.5 | 16.0 | **48.4** |

> **Critical operational finding: Whisper's WER quadruples under naive fixed-length segmentation (10.7 → 42.9). Parakeet is nearly segmentation-invariant.** For long calls, our segmentation strategy matters more than our model choice.

**Genuine 8 kHz narrowband, zero-shot vs fine-tuned** — [arXiv:2606.18659](https://arxiv.org/html/2606.18659v1) (June 2026), customer-support recordings:

| System | English WER |
|---|---|
| Whisper-large-v3 zero-shot | 28.7 |
| NeMo zero-shot | 29.5 |
| Google Telephony ASR | 27.2 |
| **NeMo fine-tuned on 50 h in-domain** | **15.2** |
| + pseudo-labels (747 h) | 13.2 |
| In-house 8 kHz BEST-RQ pretrain + FT | 12.3 |

**50 hours of in-domain telephony fine-tuning halved the error.** ⚠️ Confounded with Indian accent.

### 4.4 Why Parakeet handles narrowband better than Whisper — it was trained on it

**Parakeet and Canary training sets contain Fisher (telephone speech).** From the model cards: parakeet-v2's 10,000 h human-transcribed slice includes **Fisher Corpus**; v3 the same; canary-1b-v2 likewise; `parakeet-unified-en-0.6b` includes **Fisher AND Switchboard-1**.

Crucially, NVIDIA **upsamples** it. From the [NeMo telephony tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_for_telephony_speech.ipynb), verbatim:

> "Fisher and Switchboard datasets are conversational telephone speech datasets with audio sampled at 8 kHz... **Before training, all audio files from Fisher and Switchboard datasets were upsampled to 16 kHz.**"
> "our models can be used to recognize both **narrowband (8kHz) and wideband speech (16kHz)**"
> "Internally, the audio file is resampled to 16 kHz before inference... **there is no difference running inference on 8 kHz audio compared to say 16 kHz**."

This mixed-sample-rate training is the mechanistic explanation for Parakeet's mild −4.1% μ-law penalty.

**Whisper has no telephony corpus.** The [Whisper paper](https://ar5iv.labs.arxiv.org/html/2212.04356) describes 680k h of weakly-supervised web audio, all resampled to 16 kHz, with no Fisher, no Switchboard, no explicit narrowband. Web video audio is overwhelmingly wideband. This is consistent with Whisper's 5–6.5× telephony penalty.

> ⚠️ **Dilution caveat:** Fisher sits inside the 10k-hour human-labelled slice. For Parakeet v3 that is **<2% of 670k total hours**. Also **conflict flagged**: the [Canary/Parakeet-v3 tech report (arXiv:2509.14128)](https://arxiv.org/html/2509.14128v1) does *not* list Fisher in its corpora table, while the model cards do. Fisher's exact hour count in NeMo ASR Set 3.0 is unpublished.

### 4.5 Does upsampling help, and how? — use plain sinc, add nothing

**Yes, upsample 8→16 kHz — but with a plain bandlimited sinc resampler, not a neural model.**

[AP-BWE, arXiv:2401.06387](https://arxiv.org/html/2401.06387v2) Table IV, Whisper on VCTK:

| Method | 8→16 kHz WER |
|---|---|
| **sinc filter (plain resampling)** | **3.67%** |
| TFiLM | 3.69% |
| AFiLM | 3.67% |
| AP-BWE (paper's own) | 3.72% |
| AERO | 3.97% |
| **NVSR** (neural vocoder BWE) | **4.38%** ← +19% worse |
| Ground truth 16 kHz | 3.07% |

**At 8→16 kHz, every neural method ties or loses to sinc.** The same paper's perceptual metric ranks sinc *worst* while it ranks best-or-tied on WER. That decoupling is the whole story: **BWE is optimised for ears, not acoustic models.**

- **AudioSR measurably raises WER:** [arXiv:2508.02391](https://arxiv.org/html/2508.02391v1) Table 1, VCTK at 8 kHz cutoff: **0.116 unprocessed → 0.136 with AudioSR** (+17% relative).
- **The failure mode has a name:** [CogSR, arXiv:2512.16304](https://arxiv.org/html/2512.16304) calls it *"linguistic hallucination — the generation of intelligible but incorrect phonemes when acoustic cues are ambiguous."* mdctGAN pushes Whisper-large-v3 to 37.5% WER at 4 kHz input.
- **Confirmed independently:** [arXiv:2606.09335](https://arxiv.org/html/2606.09335) states "classical resampling (linear, soxr hq) yields moderate loss, whereas neural restoration methods (VoiceFixer, AudioSR) paradoxically worsen WER despite producing perceptually superior audio... **classical resampling more reliable for telephony preprocessing**." ⚠️ This paper publishes no numeric table for the claim — it exists only as a figure, on 2 models, in Hindi. Cite AP-BWE for numbers.
- **Most BWE papers never evaluate ASR at all** — NVSR, BAE-Net, STSR, VoiceFixer, AudioSR all verified to contain **no WER evaluation**. That is itself a finding.

**Don't add a denoiser either.** ["When De-noising Hurts", arXiv:2512.17562](https://arxiv.org/html/2512.17562v1) tested MetricGAN+ on 4,500 utterances: **all 40 configurations degraded, zero improved.** Whisper large-v3 went 8.82% → 25.83% on background noise SNR 10. Conclusion: "Modern ASR models possess sufficient internal noise robustness" and preprocessing "might not just be computationally wasteful but also be potentially harmful."

**Practical recipe:** `soxr_hq`, `sinc_interp_kaiser`, or `sox -r 16000`. **Avoid** `linear`, `zero_order_hold`, `soxr_qq`, `polyphase` — [librosa docs](https://librosa.org/doc/latest/generated/librosa.resample.html) note these are not bandlimited and can alias. ⚠️ No published WER benchmark of resampler choice exists.

### 4.6 Is there a model built for telephony?

**Open weights, downloadable:**

| Model | Native rate | Telephony evidence | License |
|---|---|---|---|
| **parakeet-tdt-0.6b-v2** | 16 k | **Only open model with a published μ-law-8k delta (−4.1%)**; 20.96% real call-centre | CC-BY-4.0 |
| SpeechBrain `asr-wav2vec2-switchboard` | 16 k | **SWBD 8.76 / CallHome 14.67 / eval2000 11.78** | Apache-2.0 |
| Vosk `en-us-aspire-0.2` (Kaldi ASpIRE) | **8 k native** | Fisher-trained TDNN, 15.60% eval2000; Vosk calls it "not very accurate" | Apache-2.0 |
| Vosk `en-us-0.22` | 16 k | **29.78% "callcenter"** vs 5.69% LibriSpeech | Apache-2.0 |
| Vosk `en-us-0.42-gigaspeech` | 16 k | 30.17% callcenter; card says *"Mostly for podcasts, not for telephony"* | Apache-2.0 |
| **FunASR Paraformer-8k / UniASR-8k** | **8 k native** | Genuine 8 kHz front end, industrial telephony — **Chinese only** | Apache-2.0 |
| Silero STT | — | **No 8 kHz STT models** (verified in [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml)) | — |

Note the Vosk numbers ([alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)) are an unusually honest published demonstration of the effect: **the same model scores 5.69% on LibriSpeech and 29.78% on call-centre audio.**

**Gated / not on-device:** `parakeet-ctc-riva-1-1b-en-us-tele` is explicitly *"trained on ASRSet with over 150,000 hours of English speech **with telephonic/narrowband augmentation**"* — but requires the Riva GPU runtime, is 3.71 GB, and publishes no WER. Picovoice states *"Picovoice software supports 16kHz audio, for 8kHz audio, please contact sales"* — proprietary and sales-gated.

**Key negative:** *"All the pretrained models currently available through NeMo are trained with audio at 16 kHz."* — [NeMo telephony tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_for_telephony_speech.ipynb). Confirmed: **zero NeMo/NGC checkpoints carry an `8k` or `telephony` tag.**

### 4.7 Hallucination — the risk WER doesn't measure

This matters more than WER for a call-summary product.

- [Koenecke et al., "Careless Whisper", FAccT 2024, arXiv:2402.08021](https://arxiv.org/abs/2402.08021): **~1% of transcriptions contain entire hallucinated phrases**; **38% of hallucinations contain explicit harms**; hallucinations disproportionately affect speakers with *"longer shares of non-vocal durations."* **Phone calls are structurally full of non-vocal duration** — hold, IVR, half-duplex gaps, comfort noise, mute.
- [arXiv:2501.11378](https://arxiv.org/pdf/2501.11378): non-speech audio systematically triggers fabricated speech.
- [HALAS, arXiv:2606.23048](https://arxiv.org/abs/2606.23048): *"hallucinations also occur for almost correctly transcribed speech (characterized by a low Word Error Rate)"* — **WER does not detect hallucination**; best detectors reach only F1 53.1%.
- [Calm-Whisper, Interspeech 2025, arXiv:2505.12969](https://arxiv.org/abs/2505.12969): **>80% reduction in non-speech hallucination for <0.1% WER cost** — mitigation is tractable if we use Whisper.

**Architectural note (inference, not a cited claim):** Parakeet's TDT/RNN-T decoder is frame-synchronous and monotonic — it lacks Whisper's free-running autoregressive text decoder, so it cannot "run away" during silence. The AfriSpeech result where attention-decoder Canary-1B-flash hit **88.71%** while transducer Parakeet-v2 hit **20.96%** is the closest published support, though the paper does not attribute it to architecture. **This is a real argument for choosing a transducer model for telephony.**

⚠️ No paper isolates 8 kHz *itself* as a hallucination trigger. "8 kHz → more hallucination" is plausible and consistent but **unproven**.

### 4.8 Anecdotal reports (clearly labelled — all N=1, uncontrolled)

- [whisper.cpp #1844](https://github.com/ggml-org/whisper.cpp/issues/1844): user flipped `COMMON_SAMPLE_RATE` to 8000, got *"random words in output."* Maintainer explains the mel filterbank is a hardcoded 80×400 array generated for `sr=16000, fmax=8000` and cannot be changed by flipping a constant; and that feeding upsampled 8 kHz means "anything above 4kHz will be zero... this mismatch can lead to hallucinations or a higher word error rate."
- [faster-whisper #273](https://github.com/SYSTRAN/faster-whisper/discussions/273) ("telephone stereo call 8k sr"): *"sometimes there are missing parts in transcription and hallucination."*
- [dev.to, dual-channel phone calls](https://dev.to/andreik/high-quality-transcription-of-noisy-dual-channel-phone-calls-92): Whisper *"tends to hallucinate heavily during long silences and with short, one-word utterances"*; author abandoned single-model Whisper.
- **No Parakeet-on-phone-audio complaints found** on GitHub or HF discussions. Low-information — far fewer people deploy it on telephony.

*Arithmetic note (mine, not cited): on the Slaney mel scale, mel(4000)/mel(8000) ≈ 0.78, so roughly 22% of Whisper's 80 mel bins — about 18 — cover 4–8 kHz and sit near-zero when fed upsampled telephone audio. They aren't garbage, they're silent, which is itself off-distribution.*

---

## 5. Small on-device LLMs for summarization

### 5.1 Licensing triage

| Family | License | Paid-product safe? | Notes |
|---|---|---|---|
| **Qwen3 / Qwen3.5 (0.6B–4B)** | **Apache-2.0** | ✅ **Yes** | No attribution, no MAU clause, no use restrictions |
| **Microsoft Phi-4-mini** | **MIT** | ✅ Yes (license) | But bad at summarization — see §5.2 |
| **SmolLM3-3B** | **Apache-2.0** | ✅ Yes | Fully open recipe; now ~1 yr old, outclassed |
| **IBM Granite 4.1-3B** | **Apache-2.0** | ✅ Yes | Clean dense arch, good llama.cpp citizen |
| **Gemma 4** (Apr 2026) | **Apache-2.0** ⭐ | ✅ Yes | **Major change** — Gemma 4 dropped the Gemma Terms |
| Gemma 3 / 3n and earlier | **Gemma Terms of Use** | ❌ Avoid | Not open source; Google reserves right to *"restrict (remotely or otherwise) usage"*; must pass restrictions downstream in our own EULA |
| **Llama 3.2 1B/3B** | Llama Community License | ⚠️ Usable, but costly | 700M MAU clause; **must display "Built with Llama"**; naming rule; Notice file; EU ambiguity |
| Ministral 3 (Dec 2025) | Apache-2.0 | ✅ license / ⚠️ quality | New 3B scores badly on faithfulness |
| **Liquid AI LFM2 / LFM2.5** | LFM Open License | ❌ **No** | Commercial rights **terminate at $10M annual revenue** |
| **LG EXAONE 4.0** | EXAONE License 1.2 | ❌ **No** | *"expressly prohibited from using... for any commercial purposes"* |
| Falcon-H1 | Falcon LLM License 1.0 | ⚠️ Conditional | Mandatory attribution + AUP whose breach auto-terminates |
| Kyutai Helium-1 2B | CC-BY-4.0 | ✅ but ❌ | **4,096-token context** — disqualified for transcripts |
| InternLM | Weights need application form | ⚠️ No | Not self-serve |

**Notable 2026 change:** [Gemma 4 is Apache-2.0](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), verified on the [model card](https://huggingface.co/google/gemma-4-E4B-it) and [Google AI docs](https://ai.google.dev/gemma/docs/core); the [old terms page](https://ai.google.dev/gemma/terms) now redirects Gemma 4 users to the Apache licence. Gemma ≤3 remain restrictive.

**Llama Community License, verbatim** ([LICENSE](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE)): must *"prominently display 'Built with Llama' on a related website, user interface, blogpost, about page, or product documentation"*; must prefix derived model names with "Llama"; >700M MAU requires a licence Meta may grant *"in its sole discretion."* Usable, but there is no reason to accept it when Qwen3-4B is Apache-2.0 and benchmarks better.

### 5.2 Summarization quality — the decisive benchmark

[Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) (HHEM-2.3, updated 2026-05-11) is precisely a summarization-faithfulness test: 7,700+ documents, prompt = "Summarize using only information provided; don't infer."

| Model | Hallucination rate | Answer rate | Avg summary length |
|---|---|---|---|
| **qwen/qwen3-4b** | **5.7%** | **99.9%** | 104.7 w |
| google/gemma-3-4b-it | 6.4% | **67.3%** ⚠️ | 77.4 w |
| mistralai/ministral-3b-2410 | 7.3% | 99.9% | 167.9 w |
| microsoft/Phi-4-mini-instruct | **23.5%** ❌ | 92.5% | **420.2 w** ❌ |
| mistralai/ministral-3-3b-2512 | **24.2%** ❌ | 74.3% ⚠️ | 119.4 w |

- **Qwen3-4B is the clear winner** — lowest hallucination *and* it doesn't refuse.
- **Gemma 3 4B's 67.3% answer rate is a red flag** — it declines a third of documents. On call recordings (complaints, medical/financial detail, profanity) an over-triggering safety layer will silently fail on our customers' *most important* calls.
- **Phi-4-mini and Ministral-3-3B both ~24% hallucination** — unusable where the summary *is* the deliverable.

Long-context support: Qwen3 Technical Report RULER scores — Qwen3-4B averages **85.2** across 4K–128K ([arXiv:2505.09388](https://arxiv.org/pdf/2505.09388)); Llama-3.2-3B degrades sharply (92.5% @4k → 77.8% @32k).

> ⚠️ Vectara has **no entries** for Qwen3.5, Gemma 4, SmolLM3, Granite, LFM2, or **Qwen3-1.7B/0.6B**. Do **not** assume Qwen3-1.7B inherits Qwen3-4B's 5.7% — measure it.

### 5.3 GGUF sizes and device reality

| Model | Q4_K_M | Source |
|---|---|---|
| Qwen3-1.7B | **1.11 GB** | [unsloth](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF) |
| **Qwen3-4B-Instruct-2507** | **2.50 GB** | [unsloth](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) |
| Qwen3.5-4B | 2.83 GB | [unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF) |
| Gemma 4 E2B | 3.11 GB | [unsloth](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| Gemma 4 E4B | 4.98 GB | [unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |

Per ["On-Device LLMs: State of the Union, 2026"](https://v-chandra.github.io/on-device-llms/), app-available RAM is *"capped at <4GB even on high-end devices"*, and mobile memory bandwidth (50–90 GB/s vs 2–3 TB/s datacenter) makes decode bandwidth-bound — **model size directly sets tok/s**. So Gemma 4 E4B at 5 GB is not shippable via GGUF; Qwen3-4B at 2.5 GB is borderline-fine; Qwen3-1.7B at 1.11 GB is comfortable.

**Two reasons to prefer Qwen3-4B-Instruct-2507 over Qwen3.5-4B:** (a) Qwen3.5 small models are hybrid Gated-DeltaNet + MoE requiring bleeding-edge llama.cpp operators; (b) [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models) reports Qwen3.5 small models are extremely verbose reasoners with ~80% hallucination on their measure — wrong trade for a battery-constrained summarizer.

**Gemma 4 has a specific llama.cpp defect:** [llama.cpp #22243](https://github.com/ggml-org/llama.cpp/issues/22243) — the GGUF loader reads PLE metadata but the per-layer residual *"is never injected into the decoder layers, leading to subtly degraded output quality."* ⚠️ Closed as `bug-unconfirmed` with no linked PR; closure appears to be triage, not a fix. This is exactly the failure that passes smoke tests and ships mediocre summaries.

### 5.4 Runtime and Rust bindings

- **llama.cpp** is healthy: 100k+ stars (Mar 2026), official Android arm64 release binaries, and `build-xcframework.sh` producing an SPM-consumable iOS XCFramework.
- **`llama-cpp-2`** (utilityai) is actively maintained — **v0.1.152, 2026-07-21**, ~195k downloads/month, with Metal / Vulkan / OpenCL / **Android** feature flags. ⚠️ **No iOS feature flag** — expect to own that bridge. ⚠️ Maintainer warns it *"does not follow semver meaningfully"* and *"is not safe"* — **pin exact versions, never caret ranges.**
- **Apple Foundation Models: disqualified.** WWDC 2026 session 241 gives the on-device model an **8,192-token context window** (PCC is 32,000 but is not offline). A 10k-token transcript plus instructions does not fit. ⚠️ Secondary sources claiming 32k on-device are wrong.
- **Google MediaPipe LLM Inference: do not build on it** — now maintenance-only, and its Android guide says *"intended for experimental and research use only"*. Successor is LiteRT-LM, which has no Rust bindings and would fragment our core.

---

## 6. Recommendation

### 6.1 The stack

| Layer | Choice | License | Size | Why |
|---|---|---|---|---|
| **ASR** | **Parakeet-TDT-0.6B-v2** | **CC-BY-4.0** | ~640 MB int8 | Only open model with a published μ-law-8k delta (−4.1%); best open model on the one real call-centre head-to-head (20.96 vs Whisper 24.69); trained on Fisher at mixed sample rate; transducer decoder resists hallucination; segmentation-robust |
| **ASR runtime** | **sherpa-onnx** via **sherpa-rs** | Apache-2.0 | — | Official Parakeet ONNX conversions, iOS + Android + desktop, real Rust bindings, C API fallback |
| **Preprocessing** | soxr_hq / sinc 8→16 kHz. **Nothing else.** | — | — | Neural BWE and denoising both measurably *hurt* |
| **Segmentation** | Silero VAD, strip long silences | MIT | ~2 MB | Cuts hallucination triggers; VAD-chunking is also how we get pseudo-streaming |
| **LLM** | **Qwen3-4B-Instruct-2507** Q4_K_M | **Apache-2.0** | 2.5 GB | Best measured summarization faithfulness in class (5.7% halluc., 99.9% answer rate); plain dense transformer; 256K native context |
| **LLM fallback** | **Qwen3-1.7B** Q4_K_M | Apache-2.0 | 1.11 GB | Low-RAM devices |
| **LLM runtime** | **llama-cpp-2** (pinned exact) | MIT | — | Metal/Vulkan/Android flags; own the iOS bridge |

**Total download:** ~3.2 GB for the full-quality tier, ~1.8 GB for the constrained tier. Both must be on-demand, per the brief.

**Why not Whisper as primary:** Whisper's training set contains no telephony corpus, it loses to Parakeet on both call-centre benchmarks, its WER *quadruples* under naive segmentation (10.7 → 42.9), and its autoregressive decoder is the one with the documented hallucination problem on silence — which phone calls are full of.

**Worth keeping in the back pocket:** whisper.cpp v1.9.0+ now runs *both* Whisper and Parakeet under one MIT runtime with CoreML/Metal/Vulkan backends. If sherpa-onnx integration friction is high, benchmark this as a single-runtime alternative. ⚠️ Its Parakeet support is ~6 weeks old and unproven.

**Also worth evaluating:** **Moonshine v2 streaming-medium** (MIT, 245M, 6.65% WER, CPU-only) as the small/fast tier — dramatically smaller than Parakeet. But it publishes **no telephony evidence at all**, so it must be measured on our own audio before trusting. English models only; the non-English licence is a hazard.

**Explicitly rejected:** Canary anything (88.71% WER on real call-centre audio); the original `canary-1b` (CC-BY-NC — illegal for us); NVIDIA's 2026 checkpoints (NVIDIA Open Model License indemnification + guardrail clauses, avoidable); Gemma ≤3 (restrictive terms + 67% answer rate); Llama (attribution burden with no upside); LFM2 and EXAONE (revenue cap / non-commercial); Apple Foundation Models (8k context).

### 6.2 Expectations to set with the product side

**Budget 15–25% WER on real English PSTN calls.** Not 6%. Design the feature so that error rate is survivable: LLM-generated summaries, action items, and searchable transcripts — **not** verbatim transcripts marketed as accurate. Consider showing transcripts with an explicit "machine-generated, may contain errors" affordance, and never present a summary as a record of what was said.

### 6.3 The single biggest risk

**Hallucination on non-speech, and the fact that WER will not warn us about it.**

The chain of evidence is uncomfortably direct: Koenecke et al. find hallucinations concentrate in audio with *"longer shares of non-vocal durations"*, and **38% of them contain explicit harms**. Phone calls are made almost entirely of non-vocal duration — hold music, IVR, ringback, mute, half-duplex gaps, comfort noise. And HALAS shows hallucinations *"occur for almost correctly transcribed speech (characterized by a low Word Error Rate)"*, with the best detectors reaching only F1 53.1%.

Then we feed that transcript to an LLM, which will smooth a fabricated sentence into a fluent, confident summary line. For a **commercial PBX product**, a summary asserting something a customer never said — attached to a real call record, in a regulated industry — is a materially worse outcome than a visibly poor transcript. **A WER-only eval will pass this and ship it.**

**Mitigations, in order:** choose a transducer (Parakeet) over an autoregressive decoder; VAD-strip silence before decoding; measure **hallucinated-span rate as a separate metric from WER**; and have the LLM prompt refuse to assert content it cannot ground in the transcript.

**Secondary risk:** every benchmark cited here has a disqualifying confound. AfriSpeech is accent-confounded with unstated sample rate; AppTek is 16 kHz VoIP, not 8 kHz PSTN; arXiv:2606.18659 is Indian-accented. **No public English 8 kHz PSTN head-to-head of Whisper vs Parakeet exists.** Model selection currently rests on proxies.

### 6.4 Recommended next step before any commitment

Run an in-house eval on **20–50 of our own G.711 call recordings**, comparing parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3, whisper-large-v3-turbo, and Moonshine v2 — **measuring WER and hallucinated-span rate separately**. This is cheap relative to the integration work and no public benchmark substitutes for it.

If accuracy is short: the one intervention with strong published evidence is **fine-tuning on in-domain telephony**. 50 hours halved WER (28.7 → 15.2) in [arXiv:2606.18659](https://arxiv.org/html/2606.18659v1); LoRA r=24 recovers ~65% of what full fine-tuning achieves ([arXiv:2512.16401](https://arxiv.org/html/2512.16401)). CC-BY-4.0 permits us to fine-tune and ship derivatives with attribution. ⚠️ Note this would require lawfully-obtained, consented call audio — a privacy and consent question, not just an ML one.

---

## 7. Consolidated list of things NOT verified

**Licensing**
- NVIDIA Open Model License clause analysis is from **one third-party blog**; the licence text is primary and needs counsel before use.
- Whether the Gemma Prohibited Use Policy is contractually binding on Apache-2.0 Gemma 4 — Google's docs still link the PUP from the Gemma 4 page.
- Whether Llama 3.2's EU restriction reaches the text-only 1B/3B models.
- Falcon-H1's exact licence text (AUP-termination clauses sourced from summaries).
- Moonshine's non-English licence — README says "non-commercial", LICENSE says "<$1M revenue OK". **Direct self-contradiction.**
- WhisperKit Pro's `argmax-fmod-license` text (gated, unreadable).
- Argmax pricing: $1.33/device/mo (pricing page) vs $0.42/device/mo (blog).

**Benchmarks / accuracy**
- **No Whisper μ-law/8 kHz ablation exists** — NVIDIA's Parakeet −4.1% figure is unopposed vendor data with no counterpart.
- **No published English 8 kHz PSTN head-to-head** of Whisper vs Parakeet.
- **No telephony number for parakeet-tdt-0.6b-v3 at all** — v2's μ-law table was dropped from the v3 card.
- **No codec-sweep WER study** (G.711 a-law/μ-law, G.729, GSM-FR, Opus-NB, packet loss/PLC) for any modern open ASR model. Genuine hole in the literature — and G.711 is *our* codec.
- **No Hub5/Switchboard/CallHome WER for any NVIDIA NeMo or Riva checkpoint**, including the telephony-tuned one.
- **No published measurement of resampler choice** (soxr vs sinc vs linear) on WER.
- Fisher's exact hour count in NeMo ASR Set 3.0 is unpublished; the Parakeet-v3 tech report and model card **disagree** on whether Fisher is included.
- Deepgram's claim that narrowband "adds 8-12% absolute WER" contradicts both controlled measurements by an order of magnitude, with no published methodology. **Do not rely on it.**
- AssemblyAI cites 6.99% WER on a Pipecat benchmark whose own repo shows 1.22% and contains no 8 kHz condition.

**Engineering**
- whisper.cpp's Parakeet support maturity, accuracy parity, and quantization options (~6 weeks old).
- Which whisper.cpp version `whisper-rs` 0.16.0 vendors; no iOS/Android build docs in that repo.
- `llama-cpp-2` iOS support — no iOS feature flag; treat as work we own.
- **All mobile tokens/sec figures found were explicitly disclaimed as estimates, not measurements.** llama.cpp's own Android discussion has op-level benchmarks only, no end-to-end tok/s. **Prefill, not decode, dominates for a 10k-token transcript, and nobody publishes prefill throughput.** Must be measured on target hardware.
- Whether the Gemma 4 PLE bug in llama.cpp is fixed — issue closed as `bug-unconfirmed`, no linked PR.
- Apple SpeechAnalyzer WER figures and Android ML Kit alpha status — secondary sources only.
- Reddit was inaccessible to the crawler, so the r/LocalLLaMA anecdotal channel is unexamined.
