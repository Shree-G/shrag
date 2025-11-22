import os
import sys
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.smith import RunEvalConfig

# Load environment variables first
load_dotenv()

# Ensure project root on path for local imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langsmith import Client

# DeepEval imports (metrics + test case model)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

# Import RAG chain factory (must return a fresh chain per invocation when called by LangSmith)
from core.chain import conversational_rag_chain

###################################################################################################
# LangSmith custom evaluator wrappers using DeepEval
# Each evaluator must accept (example, prediction, reference) and return a dict of score values.
# We attempt to extract an answer and retrieval contexts from the prediction object.
# Assumptions:
#   - Dataset row (example) contains at least a "question" field.
#   - Optionally may contain "ground_truth" or "expected_answer" for some metrics.
#   - The chain prediction may be a string answer OR a dict with keys like
#       {"answer"|"output"|"result"|"text", "contexts"|"context"|"retrieval_context": [...]}.
#   - If contexts are absent, retrieval-dependent metrics will be skipped (score=None).
###################################################################################################

# def _extract_prediction_components(example: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
#     """Robustly derive answer, contexts, and expected output from example & prediction."""
#     # Answer
#     if isinstance(prediction, dict):
#         answer = (
#             prediction.get("answer")
#             or prediction.get("output")
#             or prediction.get("result")
#             or prediction.get("text")
#             or str(prediction)
#         )
#         raw_contexts = (
#             prediction.get("contexts")
#             or prediction.get("context")
#             or prediction.get("retrieval_context")
#             or []
#         )
#     else:
#         answer = prediction if isinstance(prediction, str) else str(prediction)
#         raw_contexts = []

#     # Normalize contexts
#     if isinstance(raw_contexts, str):
#         contexts = [raw_contexts]
#     elif isinstance(raw_contexts, (list, tuple)):
#         contexts = [c for c in raw_contexts if isinstance(c, str)]
#     else:
#         contexts = []

#     # Expected / ground truth
#     expected = (
#         example.get("ground_truth")
#         or example.get("expected_answer")
#         or example.get("answer")  # in case dataset includes canonical answer
#     )
#     if expected is not None and not isinstance(expected, str):
#         expected = str(expected)

#     question = example.get("question") or example.get("input") or ""

#     return {"question": question, "answer": answer, "contexts": contexts, "expected": expected}


# def _build_test_case(components: Dict[str, Any]) -> LLMTestCase:
#     return LLMTestCase(
#         input=components["question"],
#         actual_output=components["answer"],
#         expected_output=components["expected"],
#         retrieval_context=components["contexts"],  # DeepEval uses retrieval_context name internally
#     )


# ###################################################################################################
# # RunEvaluator style wrappers (LangSmith expects callables: (run, example) -> {key, score, ...})
# # We adapt DeepEval metrics here.
# ###################################################################################################

# def _extract_from_run(run: Any, example: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert a LangSmith run + example into components our metric helpers use."""
#     # Prediction outputs may be dict or primitive
#     outputs = getattr(run, "outputs", {}) or {}
#     prediction = outputs.get("answer") or outputs.get("output") or outputs.get("result") or outputs.get("text") or outputs
#     return _extract_prediction_components(example.inputs, prediction)

def _extract_prediction_components(example: Dict[str, Any], prediction: Any) -> Dict[str, Any]:
    """Robustly derive answer, contexts, and expected output from example & prediction."""
    
    # Default values
    answer = ""
    raw_contexts = []
    
    if isinstance(prediction, dict):
        # Create a lowercase-key version for case-insensitive lookup
        prediction_lower = {k.lower(): v for k, v in prediction.items()}
        
        answer = (
            prediction_lower.get("answer")
            or prediction_lower.get("output")
            or prediction_lower.get("result")
            or prediction_lower.get("text")
            or str(prediction) # Fallback
        )
        raw_contexts = (
            prediction_lower.get("contexts")
            or prediction_lower.get("context")  # This is the key we expect
            or []
        )
    elif isinstance(prediction, str):
        answer = prediction
    else:
        answer = str(prediction)

    # Normalize contexts: The context is a list of Document objects
    contexts = []
    if isinstance(raw_contexts, (list, tuple)):
        for c in raw_contexts:
            if isinstance(c, str):
                contexts.append(c)
            elif hasattr(c, "page_content"):
                contexts.append(c.page_content) # Extract the text
    elif isinstance(raw_contexts, str):
        contexts.append(raw_contexts)

    # Expected / ground truth
    expected = (
        example.get("ground_truth")
        or example.get("expected_answer")
        or example.get("answer")
    )
    if expected is not None and not isinstance(expected, str):
        expected = str(expected)

    question = example.get("question") or example.get("input") or ""

    return {"question": question, "answer": answer, "contexts": contexts, "expected": expected}


def _build_test_case(components: Dict[str, Any]) -> LLMTestCase:
    return LLMTestCase(
        input=components["question"],
        actual_output=components["answer"],
        expected_output=components["expected"],
        retrieval_context=components["contexts"],
    )

###################################################################################################
# RunEvaluator style wrappers (LangSmith expects callables: (run, example) -> {key, score, ...})
# We adapt DeepEval metrics here.
###################################################################################################

def _extract_from_run(run: Any, example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a LangSmith run + example into components our metric helpers use."""
    
    # 'run.outputs' IS the prediction dictionary
    # It should contain {"answer": "...", "context": [...]}
    outputs = getattr(run, "outputs", {}) or {}
    
    # Pass the ENTIRE outputs dictionary as the 'prediction'
    return _extract_prediction_components(example.inputs, outputs)

def answer_relevancy_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    comps = _extract_from_run(run, example)
    test_case = _build_test_case(comps)
    metric = AnswerRelevancyMetric()
    metric.measure(test_case)
    return {"key": "answer_relevancy", "score": metric.score, "comment": getattr(metric, "reason", None)}

def faithfulness_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    comps = _extract_from_run(run, example)
    if not comps["contexts"]:
        return {"key": "faithfulness", "score": None, "comment": "No retrieval context"}
    test_case = _build_test_case(comps)
    metric = FaithfulnessMetric()
    metric.measure(test_case)
    return {"key": "faithfulness", "score": metric.score, "comment": getattr(metric, "reason", None)}

def context_precision_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    comps = _extract_from_run(run, example)
    if not comps["contexts"]:
        return {"key": "context_precision", "score": None, "comment": "No retrieval context"}
    test_case = _build_test_case(comps)
    metric = ContextualPrecisionMetric()
    metric.measure(test_case)
    return {"key": "context_precision", "score": metric.score, "comment": getattr(metric, "reason", None)}

def context_recall_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    comps = _extract_from_run(run, example)
    if not comps["contexts"]:
        return {"key": "context_recall", "score": None, "comment": "No retrieval context"}
    test_case = _build_test_case(comps)
    metric = ContextualRecallMetric()
    metric.measure(test_case)
    return {"key": "context_recall", "score": metric.score, "comment": getattr(metric, "reason", None)}

def context_relevancy_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    comps = _extract_from_run(run, example)
    if not comps["contexts"]:
        return {"key": "context_relevancy", "score": None, "comment": "No retrieval context"}
    test_case = _build_test_case(comps)
    metric = ContextualRelevancyMetric()
    metric.measure(test_case)
    return {"key": "context_relevancy", "score": metric.score, "comment": getattr(metric, "reason", None)}

deepeval_run_evaluators = [
    answer_relevancy_evaluator,
    faithfulness_evaluator,
    context_precision_evaluator,
    context_recall_evaluator,
    context_relevancy_evaluator,
]

def main():
    client = Client()

    DATASET_NAME = "rag_dataset"     # Must exist in LangSmith datasets
    PROJECT_NAME = "RAG-v5-Base-Evaluation"

    print(f"Starting evaluation on dataset: {DATASET_NAME}")

    eval_config = RunEvalConfig(
        custom_evaluators=deepeval_run_evaluators
    )

    run = client.run_on_dataset(
        dataset_name=DATASET_NAME,
        llm_or_chain_factory=conversational_rag_chain,
        evaluation=eval_config,
        project_name=PROJECT_NAME,
        input_mapper=lambda x: {"input": x["question"], "chat_history": []},
        concurrency_level=5,
    )

    print("Evaluation complete.")
    print(f"Results URL: {run.url}")

if __name__ == "__main__":
    main()