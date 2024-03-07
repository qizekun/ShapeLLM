import json
import argparse
import jsonlines
from tqdm import tqdm
import concurrent.futures
from llava.eval.gpt_eval import gpt_get_average_score


def main(args):
    ans_lines = args.answers_file.readlines()
    gt_lines = args.gt_file.readlines()
    assert len(ans_lines) == len(gt_lines)

    model_name = args.model
    output_file = args.output_file
    open(output_file, 'w').write("")

    ans_dict = {
        "General Visual Recognition": [],
        "Knowledge": [],
        "Language Generation": [],
        "Spatial Recognition": [],
        "Embodied Interaction": [],
        "Overall": [],
    }

    with tqdm(total=len(ans_lines), desc="Processing tasks", unit="task") as pbar:
        def process_task(i):
            model_output = json.loads(ans_lines[i])
            gt = json.loads(gt_lines[i])
            question = model_output['prompt']
            model_ans = model_output['text']
            gt_ans = gt['text']
            category = gt['category']

            score = gpt_get_average_score(question, category, model_ans, gt_ans, model=model_name, times=args.times)
            score = round(score, 1)
            ans_dict[category].append(score)
            ans_dict["Overall"].append(score)

            data = {
                "question_id": gt['question_id'],
                "answer_id": model_output['answer_id'],
                "question": question,
                "answer_model": model_ans,
                "answer_label": gt_ans,
                "model_id": model_name,
                "score": score,
            }
            jsonlines.open(output_file, mode='a').write(data)
            pbar.update(1)
            return f"{gt['question_id']} {score}%"

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_task, i) for i in range(len(ans_lines))]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print("Task completed:", result)
                except Exception as e:
                    print("Task encountered an exception:", e)

    for category in ans_dict:
        print(f"{category} Acc: {round(sum(ans_dict[category]) / len(ans_dict[category]), 1)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=argparse.FileType('r'), default="tables/answer.jsonl")
    parser.add_argument("--gt-file", type=argparse.FileType('r'), default="tables/gt.jsonl")
    parser.add_argument("--output-file", type=str, default="tables/result.jsonl")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--times", type=int, default=5)
    args = parser.parse_args()

    main(args)
