datasets = [
    [
        dict(
            abbr='lukaemon_mmlu_abstract_algebra',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator'),
                pred_postprocessor=dict(
                    answer_pattern='(?i)ANSWER\\s*:\\s*([A-D])',
                    type=
                    'opencompass.utils.text_postprocessors.match_answer_pattern'
                )),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD.  before answering.\n\n{input}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nCombine the task of simplifying text with the goal of rewriting a complex sentence.",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='abstract_algebra',
            path=
            '/Users/liuminxuan/Desktop/CoTdev/CoT-Pilot/workspace/opt/eval/eval_data.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
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
        openai_extra_kwargs=dict(stop=[
            '<|im_end|>',
            '<|endoftext|>',
            'User:',
            'Observation:',
            'Input:',
        ]),
        path='qwen3:0.6b',
        query_per_second=1,
        temperature=0.0,
        type='opencompass.models.OpenAISDK'),
]
work_dir = '/Users/liuminxuan/Desktop/CoTdev/CoT-Pilot/workspace/opt/eval/20260225_180446'
