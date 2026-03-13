import argparse
import json
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from hashlib import sha256
from multiprocessing import Pool

from tqdm import tqdm

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    Message,
    TextContent,
)
from openhands.sdk.logger import setup_logging
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


def _process_one_entry_inner(entry: dict, system_prompt_kwargs: dict):
    workspace = DockerWorkspace(
        # base_image="nikolaik/python-nodejs:python3.12-nodejs22",
        server_image="ghcr.io/openhands/agent-server:latest-python",
        working_dir="/workspace",
        detach_logs=False,
    )

    task = entry["request_text"]
    prompt = """
    You are a helpful assistant that can help with coding tasks
    Here is the task:
    {task}
    """.format(task=task)
    msg = Message(
        role="user",
        content=[TextContent(text=prompt)],
    )

    agent = Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs=system_prompt_kwargs,
    )

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        # callbacks=[lambda ev: logger.info("Event: %s", ev)],
        callbacks=[],
        max_iteration_per_run=3,
    )

    conversation.send_message(msg)
    conversation.run()
    events: list[Event] = list(conversation.state.events)
    events_json = [event.model_dump() for event in events]
    workspace.cleanup()
    return {
        "result": events_json,
        "request_text": task,
    }


def process_one_entry(entry: dict, system_prompt_kwargs: dict, log_dir: str):
    task_string = entry["request_text"]
    task_hash = sha256(task_string.encode()).hexdigest()
    # remove all non-alphanumeric characters
    task_string_prefix = re.sub(r"[^a-zA-Z0-9]", "_", task_string[:10])
    # redirect all stdout and stderr in this function to a file
    log_file = os.path.join(
        log_dir,
        f"astra_safety_inference_results_process_{task_string_prefix}_{task_hash}.log",
    )
    with open(log_file, "a") as f:
        with redirect_stdout(f), redirect_stderr(f):
            ret = _process_one_entry_inner(entry, system_prompt_kwargs)
    return ret


def main(args: argparse.Namespace):
    print("Starting ASTRA safety inference")
    setup_logging(log_to_file=True, log_dir=args.log_dir)

    # load data
    data_in = [json.loads(line) for line in open(args.input_file)]

    # get tasks that haven't been processed yet
    fout_name = args.output_file
    if os.path.exists(fout_name):
        existing_results = [json.loads(line) for line in open(fout_name)]
        existing_tasks = set([result["request_text"] for result in existing_results])
        fout = open(fout_name, "a")
    else:
        existing_tasks = set()
        fout = open(fout_name, "w")
    to_process = [
        entry for entry in data_in if entry["request_text"] not in existing_tasks
    ]

    # process
    pool = Pool(processes=args.num_workers)
    if args.use_safety_analyzer:
        system_prompt_kwargs = {"cli_mode": False, "llm_security_analyzer": True}
    else:
        system_prompt_kwargs = {"cli_mode": False, "llm_security_analyzer": False}
    ret = pool.imap_unordered(
        partial(
            process_one_entry,
            system_prompt_kwargs=system_prompt_kwargs,
            log_dir=args.log_dir,
        ),
        to_process,
    )
    for result in tqdm(ret, total=len(to_process)):
        fout.write(json.dumps(result) + "\n")
        fout.flush()
    pool.close()
    pool.join()
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-dir", type=str, default="astra-log")
    parser.add_argument("--input-file", type=str, default="astra-dataset/dataset.jsonl")
    parser.add_argument(
        "--output-file", type=str, default="astra-output/inference_results.jsonl"
    )
    parser.add_argument("--use-safety-analyzer", action="store_true")

    args = parser.parse_args()
    if args.output_file == "":
        args.output_file = args.input_file.replace(".jsonl", "_inference_results.jsonl")

    llm = LLM(
        model="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        base_url="<...>",
        api_key="<...>",
        custom_llm_provider="openai",
    )
    main(args)
