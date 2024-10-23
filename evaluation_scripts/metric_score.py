import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_average(scores, key):
    return sum(item[key] for item in scores) / len(scores)

def print_averages(label, scores):
    average_sacrebleu = calculate_average(scores, "sacrebleu_score")
    average_chrf = calculate_average(scores, "chrf_score")
    average_comet = calculate_average(scores, "comet_score")

    print(f"{label} : Average sacrebleu : {average_sacrebleu}")
    print(f"{label} : Average chrf : {average_chrf}")
    print(f"{label} : Average comet : {average_comet}")

def main():
    baseline = load_json('../data-files/opus_baseline_results.json')
    prediction = load_json('../data-files/opus_finetuned_results.json')

    print_averages("baseline", baseline)
    print_averages("finetuned", prediction)

if __name__ == "__main__":
    main()