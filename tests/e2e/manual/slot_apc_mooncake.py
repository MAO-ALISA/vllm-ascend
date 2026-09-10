# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual acceptance against two running Mooncake Hybrid P/D servers.

Run this script directly, not through pytest. It sends real inference traffic,
but never resets a server's cache. Unique cache salts isolate each experiment.
Use a single P/D DP replica first, then repeat with deployment routing.
"""

import argparse
import json
import math
import uuid
from urllib.request import Request, urlopen


def complete(url, model, tokens, salt, output_length, api_key, params=None):
    payload = dict(
        model=model,
        prompt=tokens,
        cache_salt=salt,
        max_tokens=output_length,
        temperature=0,
        ignore_eos=True,
        logprobs=1,
        return_token_ids=True,
        stream=False,
    )
    if params is not None:
        payload["kv_transfer_params"] = params
    request = Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def compare(cold, actual):
    expected, result = cold["choices"][0], actual["choices"][0]
    assert expected["token_ids"] and expected["token_ids"] == result["token_ids"], "Generated tokens differ"
    expected_probs = expected["logprobs"]["token_logprobs"]
    result_probs = result["logprobs"]["token_logprobs"]
    assert len(expected_probs) == len(result_probs)
    for left, right in zip(expected_probs, result_probs):
        assert math.isfinite(left) and math.isfinite(right) and abs(left - right) <= 0.05, "Log probabilities differ"


def run(args):
    boundaries = args.boundaries
    if boundaries is None:
        boundaries = sorted(
            {
                128,
                129,
                255,
                256,
                511,
                512,
                513,
                *(args.block_size * ratio + delta for ratio in (4, 128) for delta in (-1, 0, 1)),
            }
        )
    for end in boundaries:
        assert end > 0
        tokens = [100 + i % 97 for i in range(end + 1)]
        salt = uuid.uuid4().hex
        cold = complete(args.decode_url, args.model, tokens, salt + "-reference", args.output_length, args.api_key)
        for warm in (False, True):
            case_salt = salt + ("-warm" if warm else "-cold")
            if warm:
                # Divergent tails prevent a warmup from supplying the entire
                # prompt. 256 common tokens allow the speculative peek check.
                seed = tokens[: min(end, 256)] + [301, 302, 303]
                for url in (args.prefill_url, args.decode_url):
                    complete(url, args.model, seed, case_salt, 1, args.api_key)
            prefill = complete(
                args.prefill_url,
                args.model,
                tokens,
                case_salt,
                1,
                args.api_key,
                {"do_remote_decode": True, "do_remote_prefill": False},
            )
            params = prefill["kv_transfer_params"]
            assert params["slot_apc"]["version"] == 1
            assert params["slot_apc"]["num_tokens"] == end
            if warm and end >= 256:
                assert prefill["usage"]["prompt_tokens_details"]["cached_tokens"] >= 128, "P local APC did not hit"
            remote = complete(args.decode_url, args.model, tokens, case_salt, args.output_length, args.api_key, params)
            compare(cold, remote)
            # This fresh P request may hit APC completely. D still receives an
            # ACK-only task when its own local hit covers the remote prefix.
            repeated_prefill = complete(
                args.prefill_url,
                args.model,
                tokens,
                case_salt,
                1,
                args.api_key,
                {"do_remote_decode": True, "do_remote_prefill": False},
            )
            repeated = complete(
                args.decode_url,
                args.model,
                tokens,
                case_salt,
                args.output_length,
                args.api_key,
                repeated_prefill["kv_transfer_params"],
            )
            compare(cold, repeated)
            print(
                json.dumps(
                    dict(
                        block_size=args.block_size,
                        end=end,
                        warm=warm,
                        usage=remote["usage"],
                        repeated_usage=repeated["usage"],
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True, help="P server root URL, without /v1")
    parser.add_argument("--decode-url", required=True, help="D server root URL, without /v1")
    parser.add_argument("--model", required=True, help="Identical served model name on P and D")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument(
        "--block-size",
        type=int,
        choices=[32, 64, 128],
        default=128,
        help="P/D physical page size used to select boundary cases; does not change server configuration",
    )
    parser.add_argument("--boundaries", type=int, nargs="+")
    run(parser.parse_args())
