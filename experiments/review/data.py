"""Synthetic paper and reference review data for collaborative review generation."""

from __future__ import annotations

import random


SYNTHETIC_PAPERS = [
    {
        "title": "Efficient Attention Mechanisms for Long Documents",
        "abstract": "We propose a sparse attention mechanism that reduces quadratic complexity to linear for processing long documents. Our method uses learned routing to select relevant token pairs.",
        "sections": {
            "introduction": "Transformer models struggle with long sequences due to O(n^2) attention. We address this with sparse routing that learns which token pairs matter most. Related work includes Longformer and BigBird.",
            "methods": "We introduce Routed Attention: each token produces a routing vector, and attention is computed only for top-k matched pairs per token. Training uses straight-through estimators for the discrete selection.",
            "experiments": "We evaluate on document classification (IMDB, Hyperpartisan), summarization (arXiv), and QA (NarrativeQA). Our method matches dense attention quality while using 4x less memory on 8K token inputs.",
        },
        "reference_comments": [
            "The sparse routing mechanism is novel and well-motivated.",
            "The straight-through estimator may introduce gradient bias; this should be discussed.",
            "Missing comparison with recent linear attention methods like MEGA.",
            "The memory savings are significant but runtime improvements are modest.",
            "Writing is generally clear but Section 2.3 is hard to follow.",
        ],
    },
    {
        "title": "Cross-Lingual Transfer for Low-Resource NER",
        "abstract": "We study transfer learning for named entity recognition from high-resource to low-resource languages using multilingual embeddings and data augmentation.",
        "sections": {
            "introduction": "NER in low-resource languages lacks labeled data. We leverage multilingual models and augmentation to transfer knowledge from English and Chinese to 8 target languages.",
            "methods": "Our approach fine-tunes XLM-R on source language NER data, then applies entity-aware code-switching augmentation to create synthetic target language examples.",
            "experiments": "We evaluate on WikiANN across 8 low-resource languages. Our method improves F1 by 4-12 points over zero-shot transfer. Augmentation contributes 2-5 points on average.",
        },
        "reference_comments": [
            "The entity-aware code-switching augmentation is a good contribution.",
            "The selection of target languages seems arbitrary; include justification.",
            "No error analysis is provided for failure cases.",
            "Results on WikiANN alone may not generalize to other NER benchmarks.",
            "The augmentation strategy assumes entity alignment which may not hold for all languages.",
        ],
    },
    {
        "title": "Reward Shaping for Safer RLHF Alignment",
        "abstract": "We propose a reward shaping technique that explicitly penalizes harmful outputs during RLHF training while maintaining helpfulness.",
        "sections": {
            "introduction": "RLHF alignment can inadvertently trade safety for helpfulness. We introduce a dual-reward framework that separately models helpfulness and safety, combining them via constrained optimization.",
            "methods": "We train two reward models: one for helpfulness (from preference data) and one for safety (from red-team annotations). During PPO, we use a Lagrangian relaxation to maintain safety above a threshold while maximizing helpfulness.",
            "experiments": "On our internal evaluation suite, the dual-reward approach reduces harmful outputs by 40% relative to standard RLHF while maintaining 95% of helpfulness scores. Human evaluation confirms these findings.",
        },
        "reference_comments": [
            "The Lagrangian approach to safety constraints is well-suited to this problem.",
            "The safety reward model details are insufficient; what data was used?",
            "Internal evaluation suite makes reproducibility difficult.",
            "The 40% reduction in harmful outputs is impressive but needs confidence intervals.",
            "No discussion of how the method handles ambiguous safety cases.",
            "Comparison with other safe RLHF methods like Constitutional AI is missing.",
        ],
    },
    {
        "title": "Table Understanding via Structured Prompting",
        "abstract": "We present a prompting framework that converts tables into structured text representations, improving LLM performance on table QA and fact verification.",
        "sections": {
            "introduction": "LLMs struggle with tabular data because standard serialization loses structural information. We propose structured prompting that preserves row-column relationships through markup-style formatting.",
            "methods": "Tables are converted into a nested representation using XML-like tags that encode headers, data types, and cell relationships. We also add summary statistics as context.",
            "experiments": "We evaluate on WikiTableQuestions, TabFact, and SQA. Structured prompting improves accuracy by 5-15% over linearized table baselines across three LLMs (GPT-4, Claude, Llama-3).",
        },
        "reference_comments": [
            "The XML-like representation is intuitive and well-designed.",
            "Including summary statistics is a clever addition that helps with aggregation queries.",
            "The improvement is primarily on simpler queries; complex multi-hop questions show smaller gains.",
            "No analysis of token efficiency; the XML representation may be much longer than linear.",
            "Missing comparison with specialized table models like TAPAS and TaBERT.",
        ],
    },
    {
        "title": "Continual Pre-training for Domain Adaptation",
        "abstract": "We investigate strategies for continually pre-training LLMs on domain-specific corpora, finding that replay-based methods effectively prevent catastrophic forgetting.",
        "sections": {
            "introduction": "Adapting general LLMs to specialized domains (medical, legal, scientific) via continued pre-training often degrades general capabilities. We systematically study mitigation strategies.",
            "methods": "We compare four approaches: naive fine-tuning, elastic weight consolidation (EWC), replay mixing (interleaving domain data with general data), and progressive layer freezing. Each is applied to a 7B parameter model.",
            "experiments": "Evaluated on biomedical (PubMedQA), legal (CaseHOLD), and scientific (SciQ) benchmarks. Replay mixing retains 97% of general benchmark performance while achieving 90% of full domain fine-tuning quality.",
        },
        "reference_comments": [
            "Comprehensive comparison of continual learning strategies is valuable.",
            "The replay mixing ratio selection process is not well described.",
            "7B parameters may be too small; results might not transfer to larger models.",
            "Missing evaluation on generation quality (only classification tasks tested).",
            "The progressive layer freezing results are interesting but under-analyzed.",
        ],
    },
    {
        "title": "Multimodal Chain-of-Thought with Visual Grounding",
        "abstract": "We extend chain-of-thought prompting to multimodal settings by grounding intermediate reasoning steps in visual regions of the input image.",
        "sections": {
            "introduction": "Chain-of-thought prompting improves reasoning in text-only LLMs. For vision-language tasks, reasoning steps should reference specific image regions. We propose Visual CoT that links each reasoning step to bounding boxes.",
            "methods": "We train a vision-language model to produce reasoning chains where each step includes a textual explanation and a set of bounding box coordinates indicating the relevant image region. Training uses a combination of VQA datasets with region annotations.",
            "experiments": "On GQA, VCR, and a new Visual Reasoning benchmark, Visual CoT improves accuracy by 3-8% over text-only CoT and provides interpretable reasoning traces grounded in the image.",
        },
        "reference_comments": [
            "Grounding CoT steps in visual regions is a natural and effective extension.",
            "The region annotation requirement limits scalability to new datasets.",
            "GQA results are strong but VCR improvements are marginal.",
            "No analysis of when visual grounding helps vs. hurts performance.",
            "The bounding box predictions are sometimes imprecise; how does this affect downstream accuracy?",
        ],
    },
    {
        "title": "Efficient KV-Cache Compression for Long-Context LLMs",
        "abstract": "We propose a method to compress key-value caches in transformer models, enabling 4x longer context windows with minimal quality degradation.",
        "sections": {
            "introduction": "KV-cache memory grows linearly with sequence length, limiting context windows. We compress cached keys and values using learned quantization and selective eviction of low-importance tokens.",
            "methods": "Our approach scores each cached KV pair by attention importance accumulated over recent tokens. Low-scoring pairs are either evicted or quantized to 4-bit representations. The importance scorer is a lightweight MLP trained end-to-end.",
            "experiments": "On LongBench and RULER benchmarks, our method maintains 98% of full-cache quality while reducing KV-cache memory by 75%. On a 128K context model, this enables processing 512K tokens on a single A100 GPU.",
        },
        "reference_comments": [
            "The combination of eviction and quantization is practical and well-motivated.",
            "The importance scoring mechanism adds overhead; what is the latency impact?",
            "Comparison with H2O and StreamingLLM would strengthen the paper.",
            "The 4-bit quantization may interact poorly with some attention patterns.",
            "Impressive memory savings on very long contexts.",
        ],
    },
    {
        "title": "Self-Play for Instruction Following Improvement",
        "abstract": "We use self-play between an instruction-following model and a judge model to iteratively improve instruction adherence without additional human feedback.",
        "sections": {
            "introduction": "Instruction following remains imperfect even in aligned LLMs. We propose a self-play loop where a judge model evaluates responses and provides training signal, while the generator improves based on judge feedback.",
            "methods": "The judge is trained on human preference data to score instruction adherence. The generator plays against the judge using DPO with self-generated positive/negative pairs. Every 3 rounds, the judge is updated with generator outputs that received extreme scores.",
            "experiments": "On IFEval and our InstructBench, self-play improves instruction-following accuracy from 72% to 89% over 5 rounds. The judge's accuracy also improves from 81% to 87%. Human evaluation confirms improvements.",
        },
        "reference_comments": [
            "The co-evolution of generator and judge is an interesting dynamic.",
            "Risk of judge and generator colluding to produce high scores for easy instructions.",
            "5 rounds of self-play is relatively few; what happens with more rounds?",
            "The DPO formulation with self-generated pairs needs more theoretical justification.",
            "IFEval improvements are substantial and clearly presented.",
        ],
    },
    {
        "title": "Retrieval-Augmented Code Generation with Test Feedback",
        "abstract": "We augment code generation with retrieval of similar code snippets and iterative refinement using test execution feedback.",
        "sections": {
            "introduction": "Code generation often fails on the first attempt. We combine retrieval of similar solutions from a code database with iterative repair using test case execution results to improve pass rates.",
            "methods": "Given a problem, we retrieve top-5 similar solved problems and their solutions. The initial generation conditions on these examples. If tests fail, the model receives the error message and failing test case, then generates a repair. Up to 3 repair iterations are allowed.",
            "experiments": "On HumanEval, MBPP, and APPS, our approach improves pass@1 from 68% to 82% (HumanEval) and from 55% to 71% (MBPP). Retrieval contributes 5-8 points and repair contributes 6-10 points.",
        },
        "reference_comments": [
            "The combination of retrieval and test-driven repair is effective.",
            "The code database construction and maintenance is not discussed.",
            "3 repair iterations seems arbitrary; ablation on number of iterations would help.",
            "No analysis of what types of bugs are fixed vs. persist after repair.",
            "APPS results show smaller gains, suggesting limits for harder problems.",
        ],
    },
    {
        "title": "Adversarial Robustness of Vision-Language Models",
        "abstract": "We systematically evaluate the adversarial robustness of VLMs and propose a defense based on input transformation ensembling.",
        "sections": {
            "introduction": "Vision-language models inherit vulnerabilities from both vision and language components. We conduct the first systematic study of adversarial attacks across image, text, and multimodal perturbation spaces.",
            "methods": "We evaluate 5 VLMs against 8 attack types (3 image, 3 text, 2 multimodal). Our defense applies random input transformations (cropping, paraphrasing, color jittering) and aggregates predictions via majority voting.",
            "experiments": "Attacks reduce VLM accuracy by 15-45% depending on model and attack type. Our ensemble defense recovers 60-80% of lost accuracy with only 3x inference cost. Text attacks are generally more effective than image attacks.",
        },
        "reference_comments": [
            "Comprehensive evaluation across attack types is a significant contribution.",
            "The ensemble defense is simple but effective; could be stronger with adversarial training.",
            "3x inference cost is significant for deployment; discuss efficiency tradeoffs.",
            "The finding that text attacks are more effective than image attacks is interesting and deserves deeper analysis.",
            "Missing evaluation on more recent VLMs like GPT-4V and Gemini.",
        ],
    },
]


def get_papers(seed: int = 42) -> list[dict]:
    """Return the synthetic papers shuffled by seed."""
    rng = random.Random(seed)
    papers = list(SYNTHETIC_PAPERS)
    rng.shuffle(papers)
    return papers


def split_papers(papers: list[dict], n_train: int) -> tuple[list[dict], list[dict]]:
    return papers[:n_train], papers[n_train:]
