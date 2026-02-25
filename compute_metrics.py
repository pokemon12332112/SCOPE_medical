import pandas as pd
def compute_performance_using_generated_reports():
    from tools.metrics.metrics import compute_all_scores, compute_chexbert_details_scores
    mimic_cxr_generated_path = ''
    mimic_abn_generated_path = ''
    twoview_cxr_generated_path = ''
    args = {
        'chexbert_path': "",
        'bert_path': "",
        'radgraph_path': "",
    }
    for generated_path in [mimic_cxr_generated_path, mimic_abn_generated_path, twoview_cxr_generated_path]:
        data = pd.read_csv(generated_path)
        gts, gens = data['labels'].tolist(), data['report'].tolist()
        scores = compute_all_scores(gts, gens, args)
        print(scores)
compute_performance_using_generated_reports()