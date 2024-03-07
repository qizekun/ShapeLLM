import argparse
from tqdm import tqdm

import torch
from transformers import StoppingCriteria
from torch.utils.data import DataLoader
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.eval.data import ObjectPointCloudDataset
from llava.eval.evaluator import start_evaluation
from llava.mm_utils import tokenizer_point_token, get_model_name_from_path
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if
                            type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def init_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv


def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None  # * load point cloud only
    )
    print("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    if model.config.mm_use_pt_start_end:
        qs = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_POINT_TOKEN + '\n' + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_point_token(prompt, tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    responses = []

    for batch in tqdm(dataloader):
        points = batch["point_clouds"].cuda().to(model.dtype)  # * tensor of B, N, C(3)
        object_ids = batch["object_ids"]  # * list of string

        pts_tensor = points.to(model.device, dtype=torch.float16)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                points=pts_tensor,
                do_sample=True if args.temperature > 0 and args.num_beams == 1 else False,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = [outputs.strip()]

        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_path, "evaluation")

    # * output file 
    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need inferencing
        # * load annotation files
        with open(args.anno_path, 'r') as fp:
            annos = json.load(fp)

        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
        dataloader = get_dataloader(dataset, 1, args.shuffle, args.num_workers)

        model, tokenizer, conv = init_model(args)

        # * convert annos file from [{"object_id": }] to {"object_id": }
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir,
                                   args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_eval:
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file,
                         eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True,
                         num_workers=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    # * dataset type
    parser.add_argument("--data_path", type=str, default="playground/data/shapellm/cap3d_pcs", required=False)
    parser.add_argument("--anno_path", type=str, default="playground/data/eval/objaverse/PointLLM_brief_description_val_200_GT.json",
                        required=False)
    parser.add_argument("--pointnum", type=int, default=10000)
    parser.add_argument("--use_color", action="store_true", default=True)

    # * data loader, shuffle, num_workers
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613",
                        choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview"],
                        help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"],
                        help="Type of the task to evaluate.")

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)
