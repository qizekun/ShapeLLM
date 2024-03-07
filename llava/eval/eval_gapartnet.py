import re
import json
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
import concurrent.futures


def get_iou(bbox1, bbox2):

    def calculate_volume(bbox):
        min_coords = np.min(bbox, axis=0)
        max_coords = np.max(bbox, axis=0)
        side_lengths = np.maximum(0.0, max_coords - min_coords)
        volume = np.prod(side_lengths)
        return volume

    def calculate_intersection(bbox1, bbox2):
        min_coords = np.maximum(np.min(bbox1, axis=0), np.min(bbox2, axis=0))
        max_coords = np.minimum(np.max(bbox1, axis=0), np.max(bbox2, axis=0))
        side_lengths = np.maximum(0.0, max_coords - min_coords)
        intersection_volume = np.prod(side_lengths)
        return intersection_volume

    volume_bbox1 = calculate_volume(bbox1)
    volume_bbox2 = calculate_volume(bbox2)

    intersection_volume = calculate_intersection(bbox1, bbox2)
    union_volume = volume_bbox1 + volume_bbox2 - intersection_volume
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0

    return iou


def main(args):
    ans_lines = args.answers_file.readlines()
    gt_lines = args.gt_file.readlines()
    assert len(ans_lines) == len(gt_lines)

    output_file = args.output_file
    open(output_file, 'w').write("")

    ans_dict = {
        # "Bucket": [],
        # "CoffeeMachine": [],
        # "Printer": [],
        # "Camera": [],
        # "Toaster": [],
        "StorageFurniture": [],
        "Toilet": [],
        "Box": [],
        "WashingMachine": [],
        "Dishwasher": [],
        "Microwave": [],
        "Overall": [],
    }
    pattern = r"\[\[.*\], \[.*\], \[.*\], \[.*\], \[.*\], \[.*\], \[.*\], \[.*\]\]"

    with tqdm(total=len(ans_lines), desc="Processing tasks", unit="task") as pbar:
        def process_task(i):
            model_output = json.loads(ans_lines[i])
            gt = json.loads(gt_lines[i])
            question = model_output['prompt']
            model_ans = model_output['text']
            gt_ans = gt['text']
            gt_bboxes = gt['bboxes']
            category = gt['category']

            bboxes = []
            for pred in model_ans.split("]]"):
                pred = re.findall(pattern, pred + ']]')
                if len(pred) > 0:
                    bbox = json.loads(pred[0])
                    bboxes.append(bbox)

            iou_list = []
            acc_list = []
            for j in range(len(bboxes)):
                iou = get_iou(bboxes[j], gt_bboxes[j])
                iou_list.append(iou)
                if iou > 0.25:
                    acc_list.append(1)
                else:
                    acc_list.append(0)
            miou = round(sum(iou_list) / len(gt_bboxes) * 100, 2)
            acc = round(sum(acc_list) / len(gt_bboxes) * 100, 2)

            ans_dict[category].append(acc)
            ans_dict["Overall"].append(acc)

            data = {
                "question_id": gt['question_id'],
                "answer_id": model_output['answer_id'],
                "question": question,
                "answer_model": model_ans,
                "answer_label": gt_ans,
                "mIoU": miou,
                "acc": acc,
            }
            jsonlines.open(output_file, mode='a').write(data)
            pbar.update(1)
            return f"{gt['question_id']} {miou}%"

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_task, i) for i in range(len(ans_lines))]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print("Task completed:", result)
                except Exception as e:
                    print("Task encountered an exception:", e)

    category_acc_list = []
    for category in ans_dict:
        acc = round(sum(ans_dict[category]) / len(ans_dict[category]), 1)
        category_acc_list.append(acc)
        print(f"{category} Acc: {acc}%")
    print(f"Mean Acc: {round(sum(category_acc_list) / len(category_acc_list), 1)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=argparse.FileType('r'), default="tables/answer.jsonl")
    parser.add_argument("--gt-file", type=argparse.FileType('r'), default="tables/gt.jsonl")
    parser.add_argument("--output-file", type=str, default="tables/result.jsonl")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--times", type=int, default=5)
    args = parser.parse_args()

    main(args)
