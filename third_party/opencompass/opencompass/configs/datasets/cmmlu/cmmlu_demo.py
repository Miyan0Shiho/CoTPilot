from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

cmmlu_subject_mapping = {
    'ancient_chinese': '古汉语',
}

QUERY_TEMPLATE = """
你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

cmmlu_all_sets = list(cmmlu_subject_mapping.keys())

cmmlu_datasets = []
for _name in cmmlu_all_sets:
    _ch_name = cmmlu_subject_mapping[_name]
    prompt_prefix = f'请回答以下关于{_ch_name}的单项选择题, '
    cmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    cmmlu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)答案\s*:\s*[\W]*([A-D])[\W]*',
        )
    )
    cmmlu_datasets.append(
        dict(
            type=CMMLUDataset,
            path='opencompass/cmmlu',
            name=_name,
            abbr=f'cmmlu-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=cmmlu_infer_cfg,
            eval_cfg=cmmlu_eval_cfg,
        ))

del _name, _ch_name
