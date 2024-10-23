from comet import download_model, load_from_checkpoint
import pandas as pd
from tqdm import tqdm
import json

def calculate_comet_score(data, model_path="Unbabel/wmt20-comet-qe-da", batch_size=64):
    # Download and load model if string is provided
    if isinstance(model_path, str) and not model_path.endswith('.ckpt'):
        model_path = download_model(model_path)
    model = load_from_checkpoint(model_path)

    # Process data in batches
    for i in tqdm(range(0, len(data), batch_size)):
        # Get current batch
        batch = data[i:i + batch_size]

        # Prepare batch in COMET format
        comet_batch = [{
            "src": item['src'],
            "mt": item['output']
        } for item in batch]

        # Calculate COMET scores for the batch
        batch_output = model.predict(comet_batch, batch_size=batch_size, gpus=1)

        # Add scores back to the original data
        for j, score in enumerate(batch_output['scores']):
            data[i + j]['comet_score'] = score

    return data


def main():
    with open('../data-files/opus_baselines.json', 'r') as file:
        baseline = json.load(file)

    with open('../data-files/opus_finetuned.json', 'r') as file:
        prediction = json.load(file)

    # Calculate COMET scores
    baseline_updated_data = calculate_comet_score(baseline)
    prediction_updated_data = calculate_comet_score(prediction)


    # Save the list of dictionaries to a JSON file
    with open("../data-files/opus_baseline_comet_score.json", "w", encoding='utf-8') as json_file:
        json.dump(baseline_updated_data, json_file, indent=4, ensure_ascii=False)


    with open("../data-files/opus_finetuned_comet_score.json", "w", encoding='utf-8') as json_file:
        json.dump(prediction_updated_data, json_file, indent=4, ensure_ascii=False)

    print("JSON files created successfully.")


if __name__ == "__main__":
    main()