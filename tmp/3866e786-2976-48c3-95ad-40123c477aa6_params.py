datasets = [
    [
        dict(
            abbr='openai_humaneval',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.HumanEvalEvaluator'),
                k=[
                    1,
                    10,
                    100,
                ],
                pred_postprocessor=dict(
                    type='opencompass.datasets.humaneval_postprocess_v2'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{prompt}\nLet's think step by step.",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path=
            '/Users/liuminxuan/Desktop/CoTdev/CoT-Pilot/workspace/baseline/cot/eval_data.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'prompt',
                ],
                output_column='task_id',
                train_split='test'),
            type='opencompass.datasets.JsonlDataset'),
    ],
]
models = [
    dict(
        batch_size=4,
        key='EMPTY',
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(round=[
            dict(api_role='user', role='HUMAN'),
            dict(api_role='assistant', role='BOT'),
        ]),
        openai_api_base='http://localhost:11434/v1',
        openai_extra_kwargs=dict(),
        path='qwen3:0.6b',
        query_per_second=1,
        temperature=0.0,
        type='opencompass.models.OpenAISDK'),
]
work_dir = '/Users/liuminxuan/Desktop/CoTdev/CoT-Pilot/workspace/baseline/cot/20260225_205526'
