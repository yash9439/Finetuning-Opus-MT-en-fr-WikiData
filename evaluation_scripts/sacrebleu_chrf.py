import json
from evaluate import load

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def mt_eval(data):
    # Load metrics
    sacrebleu = load("sacrebleu")
    chrf = load("chrf")

    for i in range(len(data)):
        data[i]["sacrebleu_score"] = sacrebleu.compute(predictions=[data[i]["output"]], references=[data[i]["gold"]])['score']
        data[i]["chrf_score"] = chrf.compute(predictions=[data[i]["output"]], references=[data[i]["gold"]])['score']

    return data

def main():
    # Load data
    baseline = load_json('../data-files/opus_baseline_comet_score.json')
    prediction = load_json('../data-files/opus_finetuned_comet_score.json')

    # Calculate metrics
    baseline = mt_eval(baseline)
    prediction = mt_eval(prediction)

    # Save results
    save_json(baseline, '../data-files/opus_baseline_results.json')
    save_json(prediction, '../data-files/opus_finetuned_results.json')

if __name__ == "__main__":
    main()
