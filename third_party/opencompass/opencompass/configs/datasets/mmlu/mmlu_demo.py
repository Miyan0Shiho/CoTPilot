from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{input}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

# Only use one subject for demo speed
mmlu_datasets = []
name = 'abstract_algebra'

mmlu_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mmlu_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])'))

mmlu_datasets.append(
    dict(
        abbr=f'lukaemon_mmlu_{name}',
        type=MMLUDataset,
        path='opencompass/mmlu',
        name=name,
        reader_cfg=mmlu_reader_cfg,
        infer_cfg=mmlu_infer_cfg,
        eval_cfg=mmlu_eval_cfg,
    ))
